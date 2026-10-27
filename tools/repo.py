from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
import tomllib
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlparse

type Suite = Literal["smoke", "full", "coverage"]
type SnapshotOperation = Literal["prepare", "environment", "smoke", "full", "coverage"]
BOOTSTRAP_PYTHON = "3.12.13"
FULL_SUITE_SHARDS = 4
GATE_TESTS = "tests/test_repo_gates.py"
# The prepared identity adds pyproject.toml, uv.lock, .python-version, uv, and the runtime.
GATE_INPUTS = (
    ".gitignore",
    "Makefile",
    "examples",
    "tests/conftest.py",
    "tests/fixtures/fake_gh.py",
    "tests/fixtures/wheel_consumer.py",
    GATE_TESTS,
    "tools",
)


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Snapshot:
    tree_oid: str
    commit_oid: str | None
    requested_target: str


@dataclass(frozen=True)
class SnapshotTask:
    snapshot: Snapshot
    root: Path
    store: Path
    operation: SnapshotOperation


@dataclass(frozen=True)
class Runtime:
    implementation: str
    version: tuple[int, int, int]
    abi: str
    platform: str


@dataclass(frozen=True)
class PreparedEnvironment:
    identity: str
    python: Path
    purelib: Path
    runtime: Runtime


@dataclass(frozen=True)
class CapturedEnvironment:
    environment: PreparedEnvironment
    lock_bytes: bytes


@dataclass(frozen=True)
class GenerationSeal:
    generation: Path
    sha256: str


@dataclass(frozen=True)
class CheckReport:
    snapshot: Snapshot
    suite: Suite
    environment: PreparedEnvironment
    elapsed: float

    @property
    def summary(self) -> str:
        identity = self.snapshot.commit_oid or self.snapshot.tree_oid
        return f"Passed {self.suite} for {identity} in {self.elapsed:.3f}s"


@dataclass(frozen=True)
class PrRequest:
    head: str
    base: str
    title_file: Path
    body_file: Path
    repository: str = ""
    remote: str = "origin"


@dataclass(frozen=True)
class PrReceipt:
    commit_oid: str
    url: str


@dataclass(frozen=True)
class MergeRequest:
    pr: int
    remote: str = "origin"


@dataclass(frozen=True)
class PullRequest:
    head: str
    head_oid: str
    base: str
    url: str


def _run(
    args: list[str | Path],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    data: bytes | None = None,
) -> bytes:
    with tempfile.TemporaryFile() as source:
        source.write(data or b"")
        source.seek(0)
        result = subprocess.run(args, cwd=cwd, env=env, stdin=source, capture_output=True)
    if result.returncode:
        detail = (result.stdout + result.stderr).decode(errors="replace").strip()
        raise GateError(f"{shlex.join(map(str, args))} failed\n{detail}")
    return result.stdout


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return _run(["git", "--no-replace-objects", *args], repo, env=env).decode().strip()


def _child_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if (
            not key.startswith(("GIT_", "PYTHON", "PYTEST_", "UV_", "COVERAGE_"))
            or key in ("UV_OFFLINE", "UV_CACHE_DIR")
        )
        and key != "VIRTUAL_ENV"
    }


def capture_target(repo: Path, target: str) -> Snapshot:
    if target != "index":
        commit = _git(repo, "rev-parse", "--verify", "--end-of-options", f"{target}^{{commit}}")
        return Snapshot(_git(repo, "rev-parse", f"{commit}^{{tree}}"), commit, target)
    index = Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir"))
    with tempfile.TemporaryDirectory(prefix="snapshot-index-", dir=git_dir) as directory:
        copied = Path(directory) / "index"
        if index.exists():
            shutil.copyfile(index, copied)
        env = os.environ | {"GIT_INDEX_FILE": str(copied)}
        tree = _git(repo, "write-tree", env=env)
    return Snapshot(tree, None, target)


def _export(repo: Path, snapshot: Snapshot, destination: Path) -> None:
    listing = _run(
        ["git", "--no-replace-objects", "ls-tree", "-rz", "--full-tree", snapshot.tree_oid], repo
    )
    objects = []
    for entry in listing.split(b"\0")[:-1]:
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        parts = tuple(os.fsdecode(raw_path).split("/"))
        if any(part in ("", ".", "..", ".git") for part in parts):
            raise GateError(f"Unsupported tracked path {os.fsdecode(raw_path)}")
        if kind != "blob" or mode not in ("100644", "100755", "120000"):
            raise GateError(
                f"Unsupported tracked mode {mode} at {'/'.join(parts)}; gitlinks are not supported"
            )
        objects.append((parts, mode, oid))
    blobs = _run(
        ["git", "--no-replace-objects", "cat-file", "--batch"],
        repo,
        data="".join(f"{oid}\n" for _, _, oid in objects).encode(),
    )
    entries = {}
    offset = 0
    for parts, mode, _ in objects:
        if parts in entries:
            raise GateError(f"Duplicate tracked path {'/'.join(parts)}")
        end = blobs.index(b"\n", offset)
        size = int(blobs[offset:end].rsplit(b" ", 1)[1])
        entries[parts] = (mode, blobs[end + 1 : end + 1 + size])
        offset = end + 2 + size
    _validate_links(entries)
    for parts, (mode, content) in entries.items():
        path = destination.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            path.symlink_to(os.fsdecode(content))
        else:
            path.write_bytes(content)
            path.chmod(0o755 if mode == "100755" else 0o644)


def _validate_links(entries: dict[tuple[str, ...], tuple[str, bytes]]) -> None:
    directories = {parts[:end] for parts in entries for end in range(len(parts))}
    if collision := directories.intersection(entries):
        raise GateError(f"Tracked file used as a directory: {sorted(collision)}")
    portable_paths = {}
    for parts in sorted(directories | entries.keys()):
        key = _portable_key(parts)
        if key in portable_paths and portable_paths[key] != parts:
            raise GateError(
                f"Tracked paths collide under portable naming: "
                f"{'/'.join(portable_paths[key])!r} and {'/'.join(parts)!r}"
            )
        portable_paths[key] = parts
    targets = {}
    for parts, (mode, content) in entries.items():
        if mode == "120000":
            if not content or b"\0" in content or content.startswith(b"/"):
                raise GateError(
                    f"Symlink target must be a nonempty relative path: {'/'.join(parts)}"
                )
            targets[parts] = tuple(os.fsdecode(content).split("/"))

    def walk(
        parts: tuple[str, ...], parent: tuple[str, ...], active: set[tuple[str, ...]]
    ) -> tuple[tuple[str, ...], bool]:
        nonlocal expansions
        dangling = False
        for part in parts:
            if not dangling and parent in entries and parent not in targets:
                raise GateError(f"Symlink traverses a regular file: {'/'.join(parent)}")
            if part in ("", "."):
                continue
            if part == "..":
                if not parent:
                    raise GateError("Symlink escapes the snapshot root")
                parent = parent[:-1]
                continue
            parent += (part,)
            original = portable_paths.get(_portable_key(parent))
            if original is not None and original != parent:
                raise GateError(
                    f"Symlink target aliases a tracked path: "
                    f"{'/'.join(parent)!r} and {'/'.join(original)!r}"
                )
            if not dangling and parent in targets:
                if parent in active:
                    raise GateError(f"Symlink cycle at {'/'.join(parent)}")
                expansions += 1
                if expansions > 64:
                    raise GateError("Symlink traversal exceeds 64 expansions")
                parent, dangling = walk(targets[parent], parent[:-1], active | {parent})
            elif parent not in directories and parent not in entries:
                dangling = True
        return parent, dangling

    for parts, target in targets.items():
        expansions = 0
        walk(target, parts[:-1], {parts})


def _portable_key(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", part).casefold() for part in parts)


def _runtime(python: Path, cwd: Path, env: dict[str, str]) -> Runtime:
    output = _run(
        [
            python,
            "-B",
            "-I",
            "-S",
            "-c",
            "import json, sys, sysconfig; print(json.dumps([sys.implementation.name, "
            "list(sys.version_info[:3]), sysconfig.get_config_var('SOABI'), "
            "sysconfig.get_platform()]))",
        ],
        cwd,
        env=env,
    )
    implementation, version, abi, platform = json.loads(output)
    return Runtime(implementation, tuple(version), abi, platform)


def _configuration(root: Path) -> tuple[bytes, bytes, bytes]:
    files = tuple(
        (root / name).read_bytes() for name in ("pyproject.toml", "uv.lock", ".python-version")
    )
    project = tomllib.loads(files[0].decode())
    lock = tomllib.loads(files[1].decode())
    if "workspace" in project.get("tool", {}).get("uv", {}):
        raise GateError("Workspace dependencies are not supported by repository verification")
    for package in lock["package"]:
        source = package.get("source", {})
        match source:
            case {"registry": str(registry)} if len(source) == 1 and registry.strip():
                continue
            case {"virtual": "."} if (
                len(source) == 1
                and package["name"] == re.sub(r"[-_.]+", "-", project["project"]["name"]).lower()
            ):
                continue
            case _:
                raise GateError(
                    f"Unsupported dependency source for {package['name']}: {source!r}; "
                    "only a registry or the virtual root project is supported"
                )
    return files


def _snapshot_python(root: Path, env: dict[str, str], *, install_python: bool) -> Path:
    pin = (root / ".python-version").read_bytes()
    if not re.fullmatch(rb"\d+\.\d+\.\d+\s*", pin):
        raise GateError(".python-version must select an exact Python patch version")
    version = pin.decode().strip()
    find = [
        "uv",
        "python",
        "find",
        version,
        "--system",
        "--managed-python",
        "--no-project",
        "--no-config",
        "--offline",
        "--no-python-downloads",
    ]
    found = subprocess.run(find, cwd=root, env=env, stdin=subprocess.DEVNULL, capture_output=True)
    # A satisfied install still rewrites the shared libpython, which a full disk leaves empty.
    if found.returncode and install_python:
        _run(["uv", "python", "install", version, "--no-config"], root, env=env)
    return Path((_run(find, root, env=env) if found.returncode else found.stdout).decode().strip())


def _identity(root: Path, env: dict[str, str]) -> tuple[str, Runtime]:
    files = _configuration(root)
    runtime = Runtime(
        sys.implementation.name,
        tuple(sys.version_info[:3]),
        sysconfig.get_config_var("SOABI"),
        sysconfig.get_platform(),
    )
    if ".".join(map(str, runtime.version)) != files[2].decode().strip():
        raise GateError("Selected Python does not match .python-version")
    digest = hashlib.sha256()
    for content in (
        *files,
        b"dev",
        _run(["uv", "--version"], root, env=env),
        json.dumps(asdict(runtime)).encode(),
    ):
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest(), runtime


def _environment_path(repo: Path) -> Path:
    return Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "prepared"


def _prepared(identity: str, generation: Path, runtime: Runtime) -> PreparedEnvironment:
    python = generation / "bin" / "python"
    libraries = (
        generation / "lib" / f"python{runtime.version[0]}.{runtime.version[1]}" / "site-packages"
    )
    return PreparedEnvironment(identity, python, libraries, runtime)


def _sync(root: Path, prepared: PreparedEnvironment, env: dict[str, str], *, check: bool) -> None:
    args = [
        "uv",
        "sync",
        "--locked",
        "--no-install-project",
        "--no-editable",
        "--no-default-groups",
        "--group",
        "dev",
        "--python",
        str(prepared.python),
        "--no-python-downloads",
        "--no-config",
    ]
    if check:
        args.extend(["--check", "--offline"])
    output = _run(
        args, root, env=env | {"UV_PROJECT_ENVIRONMENT": str(prepared.python.parent.parent)}
    )
    if output:
        print(output.decode(), end="")


def _inspect_installation(root: Path, prepared: PreparedEnvironment, env: dict[str, str]) -> None:
    if _runtime(prepared.python, root, env) != prepared.runtime:
        raise GateError("Prepared environment uses a different Python runtime")
    for metadata in prepared.purelib.glob("*.dist-info"):
        distribution = importlib.metadata.Distribution.at(metadata)
        directory_name, _, directory_version = metadata.name.removesuffix(".dist-info").rpartition(
            "-"
        )
        fields = distribution.metadata
        name = re.sub(r"[-_.]+", "-", fields.get("Name", "")).lower()
        expected = (re.sub(r"[-_.]+", "-", directory_name).lower(), directory_version)
        if (name, fields.get("Version")) != expected:
            raise GateError(f"Distribution metadata disagrees with {metadata.name}")
        with (metadata / "RECORD").open(newline="") as record:
            rows = csv.reader(record, strict=True)
            try:
                for row in rows:
                    if len(row) != 3 or not row[0]:
                        raise GateError(f"Invalid RECORD row for {name}: {row!r}")
                    path, digest, size = row
                    installed = distribution.locate_file(path)
                    if not installed.is_file():
                        raise GateError(f"Installed file {path} is missing from {name}")
                    if size:
                        if not re.fullmatch(r"[0-9]+", size):
                            raise GateError(f"Invalid RECORD size for {name} at {path}: {size!r}")
                        try:
                            expected_size = int(size)
                        except ValueError as error:
                            raise GateError(
                                f"Invalid RECORD size for {name} at {path}: {error}"
                            ) from error
                        if installed.stat().st_size != expected_size:
                            raise GateError(f"Installed file size mismatch for {name} at {path}")
                    if digest:
                        algorithm, separator, expected_digest = digest.partition("=")
                        if not separator or not expected_digest:
                            raise GateError(f"Invalid RECORD hash for {name} at {path}: {digest!r}")
                        try:
                            with installed.open("rb") as stream:
                                actual = hashlib.file_digest(stream, algorithm).digest()
                        except (ValueError, TypeError) as error:
                            raise GateError(
                                f"Invalid RECORD hash for {name} at {path}: {error}"
                            ) from error
                        if (
                            base64.urlsafe_b64encode(actual).rstrip(b"=").decode()
                            != expected_digest
                        ):
                            raise GateError(f"Installed file hash mismatch for {name} at {path}")
            except csv.Error as error:
                raise GateError(
                    f"Invalid RECORD for {name} at line {rows.line_num}: {error}"
                ) from error
        direct = distribution.read_text("direct_url.json")
        if direct and json.loads(direct).get("dir_info", {}).get("editable"):
            raise GateError(f"Editable distribution {name} is not supported")
        if name == "quant-research":
            raise GateError(
                "The prepared environment must not contain the application distribution"
            )
    _sync(root, prepared, env, check=True)


def _generation_fingerprint(prepared: PreparedEnvironment) -> str:
    generation = prepared.python.parent.parent
    digest = hashlib.sha256()

    def field(content: bytes) -> None:
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    field(b"repo-generation-v1")
    field(prepared.identity.encode())
    field(json.dumps(asdict(prepared.runtime), sort_keys=True).encode())
    selected_python = Path(sys._base_executable).resolve(strict=True)
    runtime_aliases = {
        "python",
        "python3",
        f"python{prepared.runtime.version[0]}.{prepared.runtime.version[1]}",
    }
    paths = [generation]
    try:
        if not stat.S_ISDIR(generation.lstat().st_mode):
            raise GateError(f"Prepared generation is not a directory: {generation}")
        for path in paths:
            if stat.S_ISDIR(path.lstat().st_mode):
                paths.extend(path.iterdir())
        for path in sorted(paths, key=lambda path: os.fsencode(path.relative_to(generation))):
            mode = path.lstat().st_mode
            field(os.fsencode(path.relative_to(generation)))
            field(stat.S_IMODE(mode).to_bytes(4, "big"))
            if stat.S_ISDIR(mode):
                field(b"directory")
            elif stat.S_ISREG(mode):
                field(b"file")
                field(str(path.stat().st_size).encode())
                with path.open("rb") as stream:
                    field(hashlib.file_digest(stream, "sha256").digest())
            elif stat.S_ISLNK(mode):
                field(b"symlink")
                target = Path(os.readlink(path))
                field(os.fsencode(os.readlink(path)))
                resolved = path.resolve()
                if resolved.is_relative_to(generation):
                    parent = path.parent
                    if target.is_absolute():
                        if not target.is_relative_to(generation):
                            raise GateError(
                                f"Unsupported external prepared symlink: {path} -> {target}"
                            )
                        parent, target = generation, target.relative_to(generation)
                    for part in target.parts:
                        parent = (parent / part).resolve()
                        if not parent.is_relative_to(generation):
                            raise GateError(
                                f"Prepared symlink traverses outside generation: {path}"
                            )
                else:
                    if not (
                        path.parent == generation / "bin"
                        and path.name in runtime_aliases
                        and resolved == selected_python
                        and stat.S_ISREG(resolved.lstat().st_mode)
                    ):
                        raise GateError(
                            f"Unsupported external prepared symlink: {path} -> {resolved}"
                        )
                    field(b"selected-runtime")
                    field(os.fsencode(resolved))
                    field(stat.S_IMODE(resolved.stat().st_mode).to_bytes(4, "big"))
                    with resolved.open("rb") as stream:
                        field(hashlib.file_digest(stream, "sha256").digest())
            else:
                raise GateError(f"Unsupported prepared file type: {path}")
    except (OSError, RuntimeError) as error:
        raise GateError(f"Cannot fingerprint prepared generation {generation}: {error}") from error
    return digest.hexdigest()


def _read_generation(pointer: Path, identity: str) -> GenerationSeal:
    try:
        value = json.loads(pointer.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"Cannot read prepared generation receipt {pointer}: {error}") from error
    match value:
        case {
            "format": "repo-prepared-v2",
            "identity": expected,
            "generation": str(generation),
            "sha256": str(digest),
        } if (
            expected == identity
            and Path(generation).is_absolute()
            and re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            return GenerationSeal(Path(generation), digest)
        case _:
            raise GateError(
                f"Prepared generation receipt {pointer} is unsealed or invalid; "
                "prepare a fresh generation"
            )


def _check_environment(
    root: Path, prepared: PreparedEnvironment, env: dict[str, str], seal: GenerationSeal
) -> None:
    observed = _generation_fingerprint(prepared)
    if observed != seal.sha256:
        raise GateError(
            f"Prepared generation fingerprint mismatch at {seal.generation}: "
            f"expected {seal.sha256}, observed {observed}"
        )
    if _runtime(prepared.python, root, env) != prepared.runtime:
        raise GateError("Prepared environment uses a different Python runtime")
    _sync(root, prepared, env, check=True)


def _snapshot_operation(task: SnapshotTask) -> PreparedEnvironment:
    root, store = task.root, task.store
    env = _child_environment()
    if task.operation != "prepare":
        env |= {"UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never"}
    identity, runtime = _identity(root, env)
    if task.operation == "prepare":
        generation = store / "generations" / uuid.uuid4().hex
        generation.parent.mkdir(parents=True, exist_ok=True)
        _run(
            ["uv", "venv", "--python", sys.executable, "--no-python-downloads", generation],
            root,
            env=env,
        )
        prepared = _prepared(identity, generation, runtime)
        _sync(root, prepared, env, check=False)
        _inspect_installation(root, prepared, env)
        seal = GenerationSeal(generation, _generation_fingerprint(prepared))
        pointer = store / f"{identity}.json"
        temporary = store / f"{identity}-{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "format": "repo-prepared-v2",
                    "identity": identity,
                    "generation": str(seal.generation),
                    "sha256": seal.sha256,
                }
            )
        )
        temporary.replace(pointer)
        print(f"Prepared {identity} at {generation}")
    else:
        pointer = store / f"{identity}.json"
        seal = _read_generation(pointer, identity)
        prepared = _prepared(identity, seal.generation, runtime)
        _check_environment(root, prepared, env, seal)
        if task.operation != "environment":
            receipt = (
                _gate_receipt(root, store, identity, env) if task.operation != "coverage" else None
            )
            arguments = ["_suite", task.operation, *([str(receipt)] if receipt else [])]
            _isolated(prepared, root, "tools.repo", arguments, env)
    return prepared


def _gate_receipt(root: Path, store: Path, identity: str, env: dict[str, str]) -> Path | None:
    if not (root / GATE_TESTS).is_file():
        return None
    files = sorted(
        path
        for name in GATE_INPUTS
        for path in (root / name, *(root / name).rglob("*"))
        if path.is_file()
    )
    digest = hashlib.sha256()
    for content in (
        identity.encode(),
        _run(["git", "--version"], root, env=env),
        *(
            part
            for path in files
            for part in (os.fsencode(path.relative_to(root)), path.read_bytes())
        ),
    ):
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return store / "gate-passes" / digest.hexdigest()


def _snapshot_request(request: Path, result: Path) -> None:
    match json.loads(request.read_text()):
        case {
            "protocol": "repo-snapshot-v1",
            "operation": ("prepare" | "environment" | "smoke" | "full" | "coverage") as operation,
            "root": str(root),
            "store": str(store),
            "snapshot": {
                "tree_oid": str(tree),
                "commit_oid": (str() | None) as commit,
                "requested_target": str(target),
            },
        } if Path(root).is_absolute() and Path(store).is_absolute():
            task = SnapshotTask(Snapshot(tree, commit, target), Path(root), Path(store), operation)
        case _:
            raise GateError("Unsupported snapshot request; expected repo-snapshot-v1")
    prepared = _snapshot_operation(task)
    result.write_text(
        json.dumps({"protocol": "repo-snapshot-v1", "environment": asdict(prepared)}, default=str)
    )


def _execute_snapshot(
    repo: Path, snapshot: Snapshot, operation: SnapshotOperation
) -> CapturedEnvironment:
    env = _child_environment()
    if operation != "prepare":
        env |= {"UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never"}
    with (
        tempfile.TemporaryDirectory(prefix="repo-snapshot-") as directory,
        tempfile.TemporaryDirectory(prefix="repo-task-") as communication,
    ):
        root, invocation = Path(directory), Path(communication)
        _export(repo, snapshot, root)
        if not (root / "tools/__init__.py").is_file():
            raise GateError("Snapshot requires the tools/__init__.py package marker")
        lock_bytes = (root / "uv.lock").read_bytes()
        python = _snapshot_python(root, env, install_python=operation == "prepare")
        request, result = invocation / "request.json", invocation / "result.json"
        task = SnapshotTask(snapshot, root, _environment_path(repo), operation)
        request.write_text(
            json.dumps({"protocol": "repo-snapshot-v1", **asdict(task)}, default=str)
        )
        worker = subprocess.run(
            [python, "-B", "-I", "-S", root / "tools/repo.py", "_snapshot", request, result],
            cwd=root,
            env=env,
            stderr=subprocess.PIPE,
        )
        if worker.returncode:
            raise GateError(
                f"Snapshot {operation} failed with exit {worker.returncode}\n"
                f"{worker.stderr.decode(errors='replace').strip()}"
            )
        match json.loads(result.read_text()):
            case {
                "protocol": "repo-snapshot-v1",
                "environment": {
                    "identity": str(identity),
                    "python": str(executable),
                    "purelib": str(purelib),
                    "runtime": {
                        "implementation": str(implementation),
                        "version": [int(major), int(minor), int(patch)],
                        "abi": str(abi),
                        "platform": str(platform),
                    },
                },
            }:
                return CapturedEnvironment(
                    PreparedEnvironment(
                        identity,
                        Path(executable),
                        Path(purelib),
                        Runtime(implementation, (major, minor, patch), abi, platform),
                    ),
                    lock_bytes,
                )
            case _:
                raise GateError("Unsupported snapshot result; expected repo-snapshot-v1")


def prepare(repo: Path, snapshot: Snapshot) -> PreparedEnvironment:
    return _execute_snapshot(repo, snapshot, "prepare").environment


def install_hooks(repo: Path, snapshot: Snapshot) -> None:
    common = Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    installed = common / "repo-gates"
    with tempfile.TemporaryDirectory(prefix="repo-install-", dir=common) as directory:
        staging = Path(directory)
        source = staging / "source"
        source.mkdir()
        _export(repo, snapshot, source)
        runner = source / "tools/repo.py"
        _run(
            [
                "uv",
                "run",
                "--no-project",
                "--no-config",
                "--offline",
                "--no-python-downloads",
                "--python",
                BOOTSTRAP_PYTHON,
                "python",
                "-B",
                "-I",
                "-S",
                "-c",
                "import pathlib, sys; compile(pathlib.Path(sys.argv[1]).read_bytes(), "
                "sys.argv[1], 'exec')",
                runner,
            ],
            source,
            env=_child_environment(),
        )
        contents = {"repo.py": (runner.read_bytes(), 0o644)}
        for name, command in (
            ("pre-commit", "smoke-index"),
            ("pre-merge-commit", "smoke-index"),
            ("post-rewrite", "post-rewrite"),
        ):
            contents[name] = (
                (
                    "#!/bin/sh\n"
                    "exec uv run --no-project --no-config --offline --no-python-downloads "
                    f"--python {BOOTSTRAP_PYTHON} python -B -I -S "
                    f'"$(dirname "$0")/repo.py" {command} "$@"\n'
                ).encode(),
                0o755,
            )
        for name, (content, mode) in contents.items():
            candidate = staging / name
            candidate.write_bytes(content)
            candidate.chmod(mode)
        with (common / "repo-gates.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            retired = [name for name in ("pre-push",) if (installed / name).exists()]
            backups = {}
            for name in [*contents, *retired]:
                previous = installed / name
                if previous.exists():
                    backup = staging / f"{name}.previous"
                    shutil.copy2(previous, backup)
                    backups[name] = backup
            installed.mkdir(exist_ok=True)
            published = []
            try:
                for name in contents:
                    (staging / name).replace(installed / name)
                    published.append(name)
                for name in retired:
                    (installed / name).unlink()
                    published.append(name)
                _git(repo, "config", "--local", "core.hooksPath", str(installed))
            except (OSError, GateError):
                for name in reversed(published):
                    if name in backups:
                        backups[name].replace(installed / name)
                    else:
                        (installed / name).unlink()
                raise
    print(f"Installed repository gates at {installed}")


def _isolated_command(
    prepared: PreparedEnvironment,
    root: Path,
    module: str,
    args: list[str],
    paths: tuple[Path, ...] | None = None,
) -> list[str | Path]:
    import_paths = (root, root / "src") if paths is None else paths
    tool_imports = {"pytest": "import pytest; ", "coverage": "import pytest, coverage; "}.get(
        module, ""
    )
    bootstrap = (
        "import runpy, sys; "
        f"sys.path.append({str(prepared.purelib)!r}); "
        f"{tool_imports}"
        f"sys.path[-1:-1] = {list(map(str, import_paths))!r}; "
        f"sys.argv = {[module, *args]!r}; "
        f"runpy.run_module({module!r}, run_name='__main__')"
    )
    return [prepared.python, "-B", "-I", "-S", "-c", bootstrap]


def _isolated(
    prepared: PreparedEnvironment,
    root: Path,
    module: str,
    args: list[str],
    env: dict[str, str],
    *,
    paths: tuple[Path, ...] | None = None,
) -> None:
    result = subprocess.run(
        _isolated_command(prepared, root, module, args, paths), cwd=root, env=env
    )
    if result.returncode:
        raise GateError(f"{module} {' '.join(args)} failed with exit {result.returncode}")


def _sharded_pytest(
    prepared: PreparedEnvironment,
    root: Path,
    arguments: list[str],
    env: dict[str, str],
    directory: Path,
) -> None:
    listing = subprocess.run(
        _isolated_command(prepared, root, "pytest", [*arguments, "--collect-only"]),
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    if listing.returncode:
        raise GateError(
            f"pytest collection failed with exit {listing.returncode}\n"
            f"{listing.stdout}{listing.stderr}"
        )
    total = len(listing.stdout.partition("\n\n")[0].splitlines())
    shards = min(FULL_SUITE_SHARDS, total)
    logs = [directory / f"shard-{index}.log" for index in range(shards)]
    failures = [directory / f"shard-{index}.failures" for index in range(shards)]
    processes = []
    for index, (log, failure_log) in enumerate(zip(logs, failures, strict=True)):
        failure_log.touch()
        shard = [
            *arguments,
            "-rfE",
            "-p",
            "tools.shards",
            f"--repo-shard={index}/{shards}/{total}",
            f"--repo-failures={failure_log}",
            f"--basetemp={directory / f'shard-{index}'}",
        ]
        with log.open("w") as output:
            processes.append(
                subprocess.Popen(
                    _isolated_command(prepared, root, "pytest", shard),
                    cwd=root,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            )
    failed: list[str] = []
    printed = [0] * shards
    while True:
        finished = all(process.poll() is not None for process in processes)
        for index, failure_log in enumerate(failures):
            lines = failure_log.read_text().split("\n")[:-1]
            for line in lines[printed[index] :]:
                failed.append(f"shard {index}: {line}")
                print(failed[-1], flush=True)
            printed[index] = len(lines)
        if finished:
            break
        time.sleep(0.5)
    for log in logs:
        print(log.read_text(), end="", flush=True)
    codes = [process.returncode for process in processes]
    if any(codes):
        raise GateError("\n".join([*failed, f"pytest shards failed with exits {codes}"]))


def verify(repo: Path, snapshot: Snapshot, suite: Suite) -> CheckReport:
    start = time.monotonic()
    try:
        prepared = _execute_snapshot(repo, snapshot, suite).environment
    except (GateError, FileNotFoundError) as error:
        recovery = shlex.join(["make", "deps", f"TARGET={snapshot.requested_target}"])
        raise GateError(f"{error}\nPrepare this snapshot with: {recovery}") from error
    report = CheckReport(snapshot, suite, prepared, time.monotonic() - start)
    print(report.summary)
    return report


def _verify_wheel(root: Path, prepared: PreparedEnvironment, env: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="repo-wheel-") as directory:
        stage = Path(directory)
        _isolated(
            prepared, root, "hatchling", ["build", "-t", "wheel", "-d", str(stage)], env, paths=()
        )
        wheels = list(stage.glob("*.whl"))
        if len(wheels) != 1:
            raise GateError("Wheel build must produce exactly one wheel")
        wheel = wheels[0]
        consumer = stage / "consumer"
        consumer.mkdir()
        shutil.copytree(root / "examples/development-package", consumer / "package")
        shutil.copyfile(root / "tests/fixtures/wheel_consumer.py", consumer / "wheel_consumer.py")
        _isolated(prepared, consumer, "wheel_consumer", [str(wheel)], env, paths=(wheel, consumer))
        for command in ("inspect", "verify"):
            _isolated(
                prepared, consumer, "quant_research", [command, "package"], env, paths=(wheel,)
            )


def _run_suite(suite: Suite, receipt: Path | None = None) -> None:
    root = Path.cwd()
    python = Path(sys.executable)
    runtime = Runtime(
        sys.implementation.name,
        tuple(sys.version_info[:3]),
        sysconfig.get_config_var("SOABI"),
        sysconfig.get_platform(),
    )
    prepared = _prepared("", python.parent.parent, runtime)
    env = dict(os.environ) | {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    paths = [name for name in ("tools", "src", "tests", "examples") if (root / name).is_dir()]
    arguments = ["-q", "-p", "no:cacheprovider"]
    if suite == "smoke":
        arguments.extend(["-m", "smoke"])
    skip_gates = receipt is not None and receipt.exists()
    if skip_gates:
        arguments.append(f"--ignore={GATE_TESTS}")
    print(f"Suite {suite}: ruff check, ruff format, pytest {' '.join(arguments)}", flush=True)

    def ruff(*command: str) -> None:
        result = subprocess.run([python.parent / "ruff", *command, *paths], cwd=root, env=env)
        if result.returncode:
            raise GateError(f"ruff {' '.join(command)} failed with exit {result.returncode}")

    def run_pytest() -> None:
        with tempfile.TemporaryDirectory(prefix="repo-pytest-") as directory:
            basetemp = f"--basetemp={Path(directory) / 'run'}"
            if suite == "coverage":
                _isolated(
                    prepared, root, "coverage", ["run", "-m", "pytest", *arguments, basetemp], env
                )
                _isolated(prepared, root, "coverage", ["report"], env)
            elif suite == "full":
                _sharded_pytest(prepared, root, arguments, env, Path(directory))
            else:
                _isolated(prepared, root, "pytest", [*arguments, basetemp], env)

    steps = {
        "ruff check": lambda: ruff("check"),
        "ruff format": lambda: ruff("format", "--check"),
        "pytest": run_pytest,
    }
    if suite != "smoke" and "build-system" in tomllib.loads((root / "pyproject.toml").read_text()):
        steps["wheel"] = lambda: _verify_wheel(root, prepared, env)
    failures = {}
    for name, step in steps.items():
        try:
            step()
        except GateError as error:
            failures[name] = error
    if skip_gates:
        print(
            f"Skipped {GATE_TESTS}: a full suite already passed the same gate inputs "
            f"(receipt {receipt.name[:12]})"
        )
    if failures:
        print(
            f"Suite {suite} failed {len(failures)} of {len(steps)} steps:",
            *(f"{name}: {error}" for name, error in failures.items()),
            sep="\n",
            flush=True,
        )
        raise GateError(f"Suite {suite} failed: {', '.join(failures)}")
    if not skip_gates and receipt is not None and suite == "full":
        receipt.parent.mkdir(exist_ok=True)
        receipt.touch()


def _github_repository(url: str) -> str:
    if url.startswith("git@github.com:"):
        path = url.removeprefix("git@github.com:")
    else:
        parsed = urlparse(url)
        if parsed.hostname != "github.com" or parsed.scheme not in ("https", "ssh"):
            raise GateError("The PR push remote must identify a github.com repository")
        path = parsed.path.lstrip("/")
    path = path.removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path):
        raise GateError("Invalid GitHub repository in push remote")
    return path


def _github_branch(repo: Path, repository: str, branch: str) -> str:
    result = json.loads(
        _run(
            [
                "gh",
                "api",
                f"repos/{repository}/branches/{quote(branch, safe='')}",
                "--hostname",
                "github.com",
            ],
            repo,
        )
    )
    match result:
        case {"name": name, "commit": {"sha": str(oid)}} if name == branch and re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", oid
        ):
            return oid
        case _:
            raise GateError(f"Invalid GitHub branch lookup for {repository}:{branch}")


def _pr_url(value: str, repository: str) -> bool:
    return (
        re.fullmatch(
            rf"https://github\.com/{re.escape(repository)}/pull/[1-9][0-9]*", value, re.IGNORECASE
        )
        is not None
    )


def _open_prs(repo: Path, repository: str, head: str, base: str) -> list[str]:
    existing = json.loads(
        _run(
            [
                "gh",
                "api",
                f"repos/{repository}/pulls",
                "--hostname",
                "github.com",
                "--method",
                "GET",
                "--raw-field",
                "state=open",
                "--raw-field",
                f"head={repository.split('/', 1)[0]}:{head}",
                "--raw-field",
                f"base={base}",
                "--paginate",
                "--slurp",
            ],
            repo,
        )
    )
    if (
        not isinstance(existing, list)
        or not existing
        or any(not isinstance(page, list) for page in existing)
    ):
        raise GateError("Invalid GitHub duplicate PR lookup response")
    duplicates = []
    for item in (item for page in existing for item in page):
        match item:
            case {
                "html_url": str(url),
                "base": {"ref": str(item_base)},
                "head": {"ref": str(item_head), "repo": {"full_name": str(name)}},
            } if (
                url.strip()
                and item_head.strip()
                and item_base.strip()
                and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name)
            ):
                if name.lower() == repository.lower() and item_head == head and item_base == base:
                    if not _pr_url(url, repository):
                        raise GateError("Invalid GitHub duplicate PR URL")
                    duplicates.append(url)
            case _:
                raise GateError("Invalid GitHub duplicate PR lookup response")
    return duplicates


def create_pr(repo: Path, request: PrRequest) -> PrReceipt:
    title = request.title_file.read_text(encoding="utf-8").strip()
    body = request.body_file.read_bytes()
    if (
        not title
        or not title[0].isupper()
        or "\n" in title
        or "\0" in title
        or re.match(r"\w+(\([^)]*\))?!?:", title)
    ):
        raise GateError("PR title must start with a capital letter and have no type prefix")
    if not body.decode("utf-8").strip():
        raise GateError("PR body must be nonempty UTF-8 text")
    _git(repo, "check-ref-format", f"refs/heads/{request.head}")
    remote_url = _git(repo, "remote", "get-url", "--push", "--all", request.remote)
    repository = _github_repository(remote_url)
    if request.repository and request.repository.lower() != repository.lower():
        raise GateError("Requested repository does not match the push remote")
    qualified = f"github.com/{repository}"
    _run(["gh", "auth", "status", "--hostname", "github.com"], repo)
    try:
        metadata = json.loads(
            _run(
                ["gh", "repo", "view", qualified, "--json", "nameWithOwner,defaultBranchRef"], repo
            )
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise GateError(f"Invalid GitHub repository lookup response: {error}") from error
    match metadata:
        case {"nameWithOwner": str(name)} if name.strip():
            pass
        case _:
            raise GateError("Invalid GitHub repository lookup response")
    if name.lower() != repository.lower():
        raise GateError("GitHub repository lookup does not match the push remote")
    base = request.base
    if not base:
        match metadata:
            case {"defaultBranchRef": {"name": str(default)}} if default.strip():
                base = default
            case _:
                raise GateError("Invalid GitHub default branch lookup response")
    _git(repo, "check-ref-format", f"refs/heads/{base}")
    if base == request.head:
        raise GateError("PR base and head must differ")
    snapshot = capture_target(repo, f"refs/heads/{request.head}")
    base_oid = _github_branch(repo, repository, base)
    graph_env = _child_environment() | {"GIT_NO_LAZY_FETCH": "1", "GIT_GRAFT_FILE": os.devnull}
    try:
        if _git(repo, "rev-parse", "--is-shallow-repository", env=graph_env) != "false":
            raise GateError("Shallow history is incomplete")
        for oid in (base_oid, snapshot.commit_oid):
            if _git(repo, "cat-file", "-t", oid, env=graph_env) != "commit":
                raise GateError(f"Missing commit object {oid}")
        commits = int(
            _git(repo, "rev-list", "--count", f"{base_oid}..{snapshot.commit_oid}", env=graph_env)
        )
    except (GateError, ValueError) as error:
        raise GateError(
            f"Cannot compare local PR history against base {base_oid}: {error}\n"
            f"Obtain complete history for that base and head from {remote_url}, then retry"
        ) from error
    if not commits:
        raise GateError(f"PR head {request.head} has no commits beyond base {base} ({base_oid})")
    duplicates = _open_prs(repo, repository, request.head, base)
    if duplicates:
        raise GateError(f"An open PR already exists: {duplicates[0]}")
    with tempfile.TemporaryDirectory(prefix="repo-pr-") as directory:
        frozen_body = Path(directory) / "body.txt"
        frozen_body.write_bytes(body)
        if capture_target(repo, f"refs/heads/{request.head}").commit_oid != snapshot.commit_oid:
            raise GateError("Local head moved after capture; retry with its current commit")
        _git(repo, "push", "--", remote_url, f"{snapshot.commit_oid}:refs/heads/{request.head}")
        remote_head = _github_branch(repo, repository, request.head)
        if remote_head != snapshot.commit_oid:
            raise GateError("Remote head moved after push; no PR was created")
        outcome = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                qualified,
                "--base",
                base,
                "--head",
                request.head,
                "--title",
                title,
                "--body-file",
                frozen_body,
            ],
            cwd=repo,
            capture_output=True,
        )
    detail = (
        f"gh pr create exited {outcome.returncode}\n"
        + (outcome.stdout + outcome.stderr).decode(errors="replace").strip()
    )
    printed = {
        line.strip()
        for line in outcome.stdout.decode(errors="replace").splitlines()
        if _pr_url(line.strip(), repository)
    }
    url = printed.pop() if len(printed) == 1 else ""
    if not url:
        try:
            candidates = _open_prs(repo, repository, request.head, base)
        except (GateError, OSError, UnicodeError, json.JSONDecodeError) as error:
            raise GateError(
                f"PR creation outcome unknown for {repository} {request.head} into {base}; "
                f"recovery lookup failed: {error}\n{detail}"
            ) from error
        if len(candidates) != 1:
            raise GateError(
                f"PR creation outcome unknown for {repository} {request.head} into {base}; "
                f"found {len(candidates)} matching PRs\n{detail}"
            )
        url = candidates[0]
    try:
        created = json.loads(
            _run(
                [
                    "gh",
                    "pr",
                    "view",
                    url,
                    "--repo",
                    qualified,
                    "--json",
                    "headRefOid,headRefName,headRepository,baseRefName,url",
                ],
                repo,
            )
        )
        match created:
            case {
                "headRefOid": str(head),
                "url": created_url,
                "headRefName": head_name,
                "headRepository": {"nameWithOwner": str(head_repository)},
                "baseRefName": base_name,
            } if (
                created_url == url
                and _pr_url(created_url, repository)
                and head_name == request.head
                and head_repository.lower() == repository.lower()
                and base_name == base
            ):
                if head != snapshot.commit_oid:
                    raise GateError(f"unverified head {head}")
            case _:
                raise GateError("invalid GitHub PR lookup identity")
    except (GateError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"PR {url} exists; verification incomplete: {error}\n{detail}") from error
    if outcome.returncode:
        raise GateError(
            f"PR {url} exists at verified head {snapshot.commit_oid}; creation incomplete\n{detail}"
        )
    try:
        _run(["gh", "pr", "edit", url, "--repo", qualified, "--add-assignee", "@me"], repo)
    except (GateError, OSError) as error:
        raise GateError(
            f"PR {url} exists at verified head {snapshot.commit_oid}; "
            f"assignment incomplete: {error}"
        ) from error
    return PrReceipt(snapshot.commit_oid, url)


def _pull_request(repo: Path, qualified: str, repository: str, number: int) -> PullRequest:
    view = json.loads(
        _run(
            [
                "gh",
                "pr",
                "view",
                str(number),
                "--repo",
                qualified,
                "--json",
                "number,state,isDraft,headRefName,headRefOid,headRepository,baseRefName,url",
            ],
            repo,
        )
    )
    match view:
        case {"state": str(state)} if state != "OPEN":
            raise GateError(f"PR {number} is {state.lower()}, not open")
        case {"isDraft": True}:
            raise GateError(f"PR {number} is a draft; mark it ready for review first")
        case {"headRepository": {"nameWithOwner": str(owner)}} if (
            owner.lower() != repository.lower()
        ):
            raise GateError(
                f"PR {number} comes from {owner}; make merge needs a branch of {repository}"
            )
        case {
            "number": found,
            "isDraft": False,
            "headRefName": str(head),
            "headRefOid": str(oid),
            "baseRefName": str(base),
            "url": str(url),
        } if (
            found == number
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid)
            and _pr_url(url, repository)
        ):
            for branch in (head, base):
                _git(repo, "check-ref-format", f"refs/heads/{branch}")
            return PullRequest(head, oid, base, url)
        case _:
            raise GateError(f"Invalid GitHub lookup for PR {number}")


def merge_pr(repo: Path, request: MergeRequest) -> None:
    common = Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    with (common / "repo-merge.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            print(f"Waiting for the merge lock held by {lock.read().strip()}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        lock.truncate(0)
        lock.write(f"PR {request.pr} in {repo} since {time.strftime('%Y-%m-%d %H:%M:%S %z')}\n")
        lock.flush()
        remote_url = _git(repo, "remote", "get-url", "--push", "--all", request.remote)
        repository = _github_repository(remote_url)
        qualified = f"github.com/{repository}"
        pull = _pull_request(repo, qualified, repository, request.pr)
        if _git(repo, "branch", "--show-current") != pull.head:
            raise GateError(f"Check out {pull.head} before merging PR {request.pr}")
        if _git(repo, "status", "--porcelain", "--untracked-files=no"):
            raise GateError("Commit or discard changes to tracked files before make merge")
        local = _git(repo, "rev-parse", "HEAD")
        if local != pull.head_oid:
            raise GateError(
                f"Local {pull.head} is at {local} but PR {request.pr} is at {pull.head_oid}; "
                "push or pull first"
            )
        upstream = f"refs/remotes/{request.remote}/{pull.base}"
        fetch = ["fetch", request.remote, f"+refs/heads/{pull.base}:{upstream}"]
        _git(repo, *fetch)
        while True:
            if not _ancestor(repo, upstream, "HEAD"):
                print(f"Merging {request.remote}/{pull.base} into {pull.head}", flush=True)
                try:
                    _git(repo, "merge", "--no-ff", "--no-edit", "--no-verify", upstream)
                except GateError as error:
                    # A merge that never started leaves nothing to abort.
                    subprocess.run(["git", "merge", "--abort"], cwd=repo, capture_output=True)
                    raise GateError(
                        f"{error}\nMerge {request.remote}/{pull.base} into {pull.head} by hand, "
                        "resolve, commit, push, and rerun make merge"
                    ) from error
                merged = _git(repo, "rev-parse", "HEAD")
                _git(repo, "push", "--", remote_url, f"{merged}:refs/heads/{pull.head}")
                if _github_branch(repo, repository, pull.head) != merged:
                    raise GateError(f"Remote {pull.head} moved after pushing {merged}")
                print(f"Pushed {merged} to {pull.head}", flush=True)
            changed = _run(
                ["git", "diff", "--name-only", "--no-renames", "-z", f"{upstream}...HEAD"], repo
            )
            documentation = all(
                path.startswith(".claude/")
                or (path.endswith(".md") and (path.startswith("docs/") or "/" not in path))
                for path in os.fsdecode(changed).split("\0")[:-1]
            )
            report = verify(
                repo, capture_target(repo, "HEAD"), "smoke" if documentation else "full"
            )
            verified = report.snapshot.commit_oid
            _git(repo, *fetch)
            if _ancestor(repo, upstream, verified):
                break
            print(f"{pull.base} moved during the run; merging it again", flush=True)
        current = _pull_request(repo, qualified, repository, request.pr)
        if current.head_oid != verified:
            raise GateError(f"PR {request.pr} moved on GitHub after {verified} passed")
        if current.base != pull.base:
            raise GateError(
                f"PR {request.pr} was retargeted from {pull.base} to {current.base} "
                f"after {verified} passed; rerun make merge"
            )
        _run(["gh", "pr", "comment", pull.url, "--repo", qualified, "--body", report.summary], repo)
        _run(
            [
                "gh",
                "pr",
                "merge",
                pull.url,
                "--repo",
                qualified,
                "--merge",
                "--match-head-commit",
                verified,
            ],
            repo,
        )
        result = json.loads(
            _run(
                ["gh", "pr", "view", pull.url, "--repo", qualified, "--json", "state,mergeCommit"],
                repo,
            )
        )
        match result:
            case {"state": "MERGED", "mergeCommit": {"oid": str(commit)}}:
                print(f"Merged {pull.url} at {verified} as {commit}")
            case _:
                raise GateError(f"gh pr merge returned, but {pull.url} reports {result!r}")


def _ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "--no-replace-objects", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo,
        capture_output=True,
    )
    if result.returncode not in (0, 1):
        raise GateError(f"Cannot compare {ancestor} with {descendant}\n{result.stderr.decode()}")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "deps",
            "hooks",
            "smoke-index",
            "verify",
            "coverage",
            "post-rewrite",
            "pr",
            "merge",
            "run",
            "example",
            "_suite",
            "_snapshot",
        ],
    )
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "_suite":
        _run_suite(args.arguments[0], *map(Path, args.arguments[1:]))
        return
    if args.command == "_snapshot":
        if len(args.arguments) != 2:
            raise GateError("The private snapshot command requires request and result paths")
        _snapshot_request(*map(Path, args.arguments))
        return
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    match args.command:
        case "deps":
            prepare(repo, capture_target(repo, os.environ.get("TARGET") or "index"))
        case "hooks":
            install_hooks(repo, capture_target(repo, os.environ.get("TARGET") or "index"))
        case "smoke-index":
            verify(repo, capture_target(repo, "index"), "smoke")
        case "verify" | "coverage":
            suite = "full" if args.command == "verify" else "coverage"
            verify(repo, capture_target(repo, os.environ.get("TARGET") or "HEAD"), suite)
        case "post-rewrite":
            if args.arguments[:1] == ["rebase"]:
                try:
                    verify(repo, capture_target(repo, "HEAD"), "smoke")
                except GateError as error:
                    raise GateError(
                        f"{error}\nThe rebased branch fails the smoke suite. "
                        "Fix it and commit before make merge."
                    ) from error
        case "run" | "example":
            try:
                arguments = (
                    shlex.split(os.environ.get("ARGS", ""))
                    if args.command == "run"
                    else [os.environ.get("DESTINATION") or "local-runs"]
                )
            except ValueError as error:
                raise GateError(f"Invalid ARGS: {error}") from error
            try:
                captured = _execute_snapshot(repo, capture_target(repo, "index"), "environment")
            except (GateError, FileNotFoundError) as error:
                raise GateError(
                    f"{error}\nIf staged dependency metadata is invalid or stale, correct it, "
                    "run uv lock, and stage pyproject.toml, uv.lock and .python-version first.\n"
                    "Prepare this snapshot with: make deps TARGET=index"
                ) from error
            env = _child_environment() | {"UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never"}
            module = "quant_research" if args.command == "run" else "examples.save_development"
            with tempfile.TemporaryDirectory(prefix="repo-authoring-") as directory:
                if args.command == "example":
                    lock = Path(directory) / "uv.lock"
                    lock.write_bytes(captured.lock_bytes)
                    arguments.extend(["--lock", str(lock)])
                _isolated(captured.environment, repo, module, arguments, env)
        case "pr":
            receipt = create_pr(
                repo,
                PrRequest(
                    head=os.environ.get("HEAD", ""),
                    base=os.environ.get("BASE", ""),
                    title_file=Path(os.environ.get("TITLE_FILE", "")),
                    body_file=Path(os.environ.get("BODY_FILE", "")),
                    repository=os.environ.get("REPOSITORY", ""),
                    remote=os.environ.get("REMOTE") or "origin",
                ),
            )
            print(f"Created {receipt.url} at commit {receipt.commit_oid}")
        case "merge":
            number = os.environ.get("PR", "")
            if not re.fullmatch(r"[1-9][0-9]*", number):
                raise GateError("make merge requires PR=<pull request number>")
            merge_pr(repo, MergeRequest(int(number), os.environ.get("REMOTE") or "origin"))


if __name__ == "__main__":
    try:
        main()
    except (
        GateError,
        OSError,
        UnicodeError,
        tomllib.TOMLDecodeError,
        json.JSONDecodeError,
    ) as error:
        sys.exit(f"Repository gate failed: {error}")
