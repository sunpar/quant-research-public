from __future__ import annotations

import base64
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from quant_research.packages import open_package
from tools import repo as gates

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
PASSING = "import pytest\n\n\n@pytest.mark.smoke\ndef test_value():\n    assert True\n"
FAILING = PASSING.replace("assert True", "assert 1 == 2")


def command(root: Path, *args: str | Path, env=None, success=True, timeout=120):
    result = subprocess.run(
        args, cwd=root, env=env, text=True, capture_output=True, timeout=timeout
    )
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    return result


def git(root: Path, *args: str, env=None) -> str:
    return command(root, "git", *args, env=env).stdout.strip()


def write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def wheel(
    destination: Path,
    name: str,
    version: str,
    files: dict[str, bytes],
    tag="py3-none-any",
    record_fields=None,
):
    dist = f"{name.replace('-', '_')}-{version}.dist-info"
    files |= {
        f"{dist}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n".encode(),
    }
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for path, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        fields = [path, f"sha256={digest}", str(len(content))]
        if path == "gate_dep.py" and record_fields:
            fields = record_fields(content, fields)
        writer.writerow(fields)
    writer.writerow((f"{dist}/RECORD", "", ""))
    files[f"{dist}/RECORD"] = record.getvalue().encode()
    output = destination / f"{name.replace('-', '_')}-{version}-{tag}.whl"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            info = zipfile.ZipInfo(path, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o755 if path.endswith("/scripts/ruff") else 0o644) << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED)
    return output


def wheel_dependency(root: Path, version: str) -> Path:
    return wheel(
        root,
        "gate_dep",
        version,
        {
            "gate_dep.py": f"VERSION = {version!r}\n".encode(),
            "examples/__init__.py": b'raise RuntimeError("Dependency examples imported")\n',
            "examples/save_development.py": b'raise RuntimeError("Dependency author imported")\n',
            f"gate_dep-{version}.dist-info/METADATA": (
                f"Metadata-Version: 2.1\nName: gate-dep\nVersion: {version}\n"
            ).encode(),
        },
    )


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for key in os.environ:
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")


@pytest.fixture(scope="session")
def wheelhouse(tmp_path_factory):
    destination = Path(tempfile.mkdtemp(prefix="repo-wheel-index-")).resolve()
    names = (
        "pytest",
        "ruff",
        "coverage",
        "iniconfig",
        "packaging",
        "pluggy",
        "pygments",
        "hatchling",
        "pathspec",
        "tomlkit",
        "trove-classifiers",
    )
    for name in names:
        distribution = importlib.metadata.distribution(name)
        version = distribution.version
        files = {}
        for entry in distribution.files:
            path = Path(str(entry))
            if path.suffix == ".pyc" or path.name in (
                "RECORD",
                "INSTALLER",
                "REQUESTED",
                "direct_url.json",
            ):
                continue
            if ".." in path.parts:
                if name == "ruff" and path.name == "ruff":
                    files[f"ruff-{version}.data/scripts/ruff"] = distribution.locate_file(
                        entry
                    ).read_bytes()
                continue
            files[str(path)] = distribution.locate_file(entry).read_bytes()
        tag = next(
            line[5:]
            for line in distribution.read_text("WHEEL").splitlines()
            if line.startswith("Tag: ")
        )
        wheel(destination, name, version, files, tag)
    wheel_dependency(destination, "1.0")
    wheel_dependency(destination, "2.0")
    for path in destination.glob("*.whl"):
        folder = destination / path.name.split("-")[0].replace("_", "-")
        folder.mkdir(exist_ok=True)
        path.replace(folder / path.name)
    for folder in destination.iterdir():
        (folder / "index.html").write_text(
            "\n".join(
                f'<a href="{path.name}">{path.name}</a>' for path in sorted(folder.glob("*.whl"))
            )
        )
    yield destination
    shutil.rmtree(destination)


@pytest.fixture(scope="session")
def template(tmp_path_factory, wheelhouse):
    root = Path(tempfile.mkdtemp(prefix="repo-fixture-")).resolve()
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env |= {"GIT_CONFIG_GLOBAL": os.devnull, "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never"}
    git(root, "init", "-b", "main", env=env)
    git(root, "config", "user.name", "Gate Test", env=env)
    git(root, "config", "user.email", "gate@example.invalid", env=env)
    for name in (
        "Makefile",
        ".python-version",
        ".gitignore",
        "tools/__init__.py",
        "tools/repo.py",
        "tools/shards.py",
    ):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write(
        root,
        "pyproject.toml",
        textwrap.dedent("""\
        [project]
        name = "gate-fixture"
        version = "0.1.0"
        requires-python = ">=3.12,<3.13"
        dependencies = []
        [dependency-groups]
        dev = ["pytest", "ruff", "iniconfig", "packaging", "pluggy", "pygments", "gate-dep==1.0"]
        [tool.uv]
        package = false
        [tool.pytest.ini_options]
        testpaths = ["tests"]
        addopts = "--import-mode=importlib"
        markers = ["smoke: fixture smoke"]
        [tool.ruff]
        target-version = "py312"
        line-length = 100
        [tool.ruff.lint]
        select = ["E", "F", "I", "UP", "B"]
        [[tool.uv.index]]
        default = true
        """).replace(
            "package = false",
            f"package = false\nenvironments = [\"sys_platform == '{sys.platform}'\"]",
        )
        + f'url = "{wheelhouse.as_uri()}"\n',
    )
    write(root, "tests/test_fixture.py", PASSING)
    command(root, "uv", "lock", "--offline", "--python", "3.12.13", env=env)
    git(root, "add", ".", env=env)
    command(root, "make", "setup", "TARGET=index", env=env)
    git(root, "commit", "--no-verify", "-m", "Prepare fixture", env=env)
    yield root
    shutil.rmtree(root)


@pytest.fixture
def repository(tmp_path, template):
    root = tmp_path / "repository with spaces"
    shutil.copytree(template, root, ignore=shutil.ignore_patterns("prepared", "__pycache__"))
    (root / ".git/prepared").mkdir()
    for pointer in (template / ".git/prepared").glob("*.json"):
        shutil.copyfile(pointer, root / ".git/prepared" / pointer.name)
    git(root, "config", "--local", "core.hooksPath", str(root / ".git/repo-gates"))
    return root


@pytest.fixture
def alternate_python(repository):
    result = subprocess.run(
        ["uv", "python", "find", "3.13.12", "--no-python-downloads", "--offline"],
        cwd=repository,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        pytest.skip("Python 3.13.12 is not installed; alternate-runtime coverage is unavailable")
    return result.stdout.strip()


def index_path(root: Path) -> Path:
    return Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "index"))


def tree_bytes(root: Path):
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def replace_installed_file(path: Path, content: bytes):
    replacement = path.with_name(f".{path.name}.replacement")
    replacement.write_bytes(content)
    replacement.chmod(path.stat().st_mode)
    replacement.replace(path)


def test_partial_staging_uses_index_without_mutating_it(repository):
    root = repository
    write(root, "tests/test_fixture.py", FAILING)
    git(root, "add", "tests/test_fixture.py")
    write(root, "tests/test_fixture.py", PASSING)
    before = index_path(root).read_bytes()
    tracked = {name: (root / name).read_bytes() for name in git(root, "ls-files").splitlines()}
    result = command(root, "make", "smoke-index", success=False)
    assert "assert 1 == 2" in result.stdout
    assert index_path(root).read_bytes() == before
    assert (root / "tests/test_fixture.py").read_text() == PASSING
    assert {name: (root / name).read_bytes() for name in tracked} == tracked
    command(root, "git", "commit", "-m", "Reject staged failure", success=False)
    assert git(root, "rev-list", "--count", "HEAD") == "1"
    git(root, "add", "tests/test_fixture.py")
    write(root, "tests/test_fixture.py", FAILING)
    before = index_path(root).read_bytes()
    tracked = {name: (root / name).read_bytes() for name in tracked}
    command(root, "make", "smoke-index")
    assert index_path(root).read_bytes() == before
    assert {name: (root / name).read_bytes() for name in tracked} == tracked
    git(root, "commit", "--allow-empty", "-m", "Accept staged repair")
    assert (root / "tests/test_fixture.py").read_text() == FAILING


@pytest.mark.smoke
def test_installed_hooks_ignore_working_runner_and_make_errors(repository):
    root = repository
    installed = tree_bytes(root / ".git/repo-gates")
    write(root, "note.txt", "staged change\n")
    git(root, "add", "note.txt", "tools/repo.py", "Makefile")
    captured = gates.capture_target(root, "index")
    runner, makefile = (root / "tools/repo.py").read_bytes(), (root / "Makefile").read_bytes()
    write(root, "tools/repo.py", "invalid Python (\n")
    write(root, "Makefile", "$(error invalid working Makefile)\n")
    before = {name: (root / name).read_bytes() for name in git(root, "ls-files").splitlines()}
    git(root, "commit", "-m", "Verify staged tooling")
    assert git(root, "rev-parse", "HEAD^{tree}") == captured.tree_oid
    assert {name: (root / name).read_bytes() for name in before} == before
    git(root, "add", "tools/repo.py", "Makefile")
    (root / "tools/repo.py").write_bytes(runner)
    (root / "Makefile").write_bytes(makefile)
    staged = gates.capture_target(root, "index")
    head = git(root, "rev-parse", "HEAD")
    result = command(root, "git", "commit", "-m", "Reject invalid staged runner", success=False)
    assert "SyntaxError" in result.stdout + result.stderr
    assert git(root, "rev-parse", "HEAD") == head
    assert gates.capture_target(root, "index") == staged
    assert (root / "tools/repo.py").read_bytes() == runner
    assert tree_bytes(root / ".git/repo-gates") == installed


@pytest.mark.parametrize("index_kind", ["custom-split", "commit-only"])
def test_staged_tooling_divergence_through_special_index_hooks(repository, tmp_path, index_kind):
    root = repository
    write(root, "note.txt", "selected change\n")
    git(root, "add", "note.txt")
    env = dict(os.environ)
    if index_kind == "custom-split":
        git(root, "update-index", "--split-index")
        selected = tmp_path / "custom split index"
        shutil.copyfile(index_path(root), selected)
        env["GIT_INDEX_FILE"] = str(selected)
        original = index_path(root).read_bytes()
        index_bytes = selected.read_bytes()
    else:
        write(root, "tests/test_fixture.py", FAILING)
        git(root, "add", "tests/test_fixture.py")
        held = git(root, "rev-parse", ":tests/test_fixture.py")
    write(root, "tools/repo.py", "invalid working Python (\n")
    write(root, "Makefile", "$(error invalid working Makefile)\n")
    if index_kind == "custom-split":
        command(root, root / ".git/repo-gates/pre-commit", env=env)
        assert selected.read_bytes() == index_bytes
        assert index_path(root).read_bytes() == original
        command(root, "git", "commit", "-m", "Use custom split index", env=env)
        assert index_path(root).read_bytes() == original
    else:
        git(root, "commit", "--only", "note.txt", "-m", "Use temporary commit index")
        assert git(root, "rev-parse", ":tests/test_fixture.py") == held
        assert git(root, "diff", "--cached", "--name-only") == "tests/test_fixture.py"
    assert git(root, "show", "HEAD:note.txt") == "selected change"
    assert (root / "tools/repo.py").read_text() == "invalid working Python (\n"
    assert (root / "Makefile").read_text() == "$(error invalid working Makefile)\n"


def test_custom_split_index_and_partial_path_commit(repository, tmp_path, monkeypatch):
    root = repository
    git(root, "update-index", "--split-index")
    custom = tmp_path / "custom index"
    shutil.copyfile(index_path(root), custom)
    env = os.environ | {"GIT_INDEX_FILE": str(custom)}
    write(root, "tests/test_fixture.py", FAILING)
    git(root, "add", "tests/test_fixture.py", env=env)
    write(root, "tests/test_fixture.py", PASSING)
    before = custom.read_bytes()
    normal = index_path(root).read_bytes()
    command(root, "make", "smoke-index", env=env, success=False)
    assert custom.read_bytes() == before
    assert index_path(root).read_bytes() == normal
    monkeypatch.setenv("GIT_INDEX_FILE", str(custom))
    captured = gates.capture_target(root, "index")
    assert (
        git(root, "show", f"{captured.tree_oid}:tests/test_fixture.py", env=env) == FAILING.strip()
    )
    monkeypatch.delenv("GIT_INDEX_FILE")
    write(root, "note.txt", "new\n")
    git(root, "add", "note.txt")
    write(root, "tests/test_fixture.py", FAILING)
    command(
        root,
        "git",
        "commit",
        "--only",
        "tests/test_fixture.py",
        "-m",
        "Reject temporary index",
        success=False,
    )
    write(root, "tests/test_fixture.py", PASSING.replace("assert True", "assert 1 == 1"))
    command(
        root, "git", "commit", "--only", "tests/test_fixture.py", "-m", "Accept temporary index"
    )
    assert git(root, "diff", "--cached", "--name-only") == "note.txt"


@pytest.mark.smoke
def test_raw_export_preserves_modes_attributes_filters_and_symlinks(repository, tmp_path):
    root = repository
    write(root, ".gitattributes", "hidden.txt export-ignore\nfiltered.txt filter=replace\n")
    write(root, "hidden.txt", "visible to verification\n")
    write(root, "filtered.txt", "raw Git bytes\n")
    executable = write(root, "executable", "#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    (root / "link").symlink_to("hidden.txt")
    write(root, "directory/nested/value", "inside\n")
    write(root, "directory/nested/other", "shared ancestor\n")
    targets = {
        "directory-link": "directory/nested",
        "chain": "directory-link/value",
        "parent-after-link": "directory-link/../../hidden.txt",
        "repeated-link": "directory-link/../../directory-link/value",
        "dangling": "missing/../also-missing/value",
        "dangling-regular": "missing/../executable/child",
        "directory/dangling": "nested/absent/",
        "raw-target": "directory//nested/./value",
    }
    for name, target in targets.items():
        (root / name).symlink_to(target)
    (root / "raw-bytes").symlink_to(os.fsdecode(b"directory/\xff"))
    git(root, "add", ".")
    git(root, "config", "filter.replace.smudge", "sed s/raw/filtered/")
    destination = tmp_path / "export"
    destination.mkdir()
    gates._export(root, gates.capture_target(root, "index"), destination)
    assert (destination / "hidden.txt").read_bytes() == b"visible to verification\n"
    assert (destination / "filtered.txt").read_bytes() == b"raw Git bytes\n"
    assert (destination / "executable").stat().st_mode & 0o111 == 0o111
    assert (destination / "link").is_symlink()
    assert (destination / "link").readlink() == Path("hidden.txt")
    for name, target in targets.items():
        assert os.readlink(destination / name) == target
    assert os.fsencode(os.readlink(destination / "raw-bytes")) == b"directory/\xff"
    assert (destination / "chain").read_text() == "inside\n"
    assert (destination / "parent-after-link").read_bytes() == b"visible to verification\n"
    assert (destination / "repeated-link").read_text() == "inside\n"
    assert not (destination / "dangling").exists()
    git(
        root,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{git(root, 'rev-parse', 'HEAD')},submodule",
    )
    with pytest.raises(gates.GateError, match="gitlinks"):
        gates.verify(root, gates.capture_target(root, "index"), "smoke")


@pytest.mark.parametrize(
    "links",
    [
        {"link": b"../outside"},
        {"link": b"/outside"},
        {"link": b"../outside/../snapshot/file"},
        {"link": b"alias/../../outside", "alias": b"directory"},
        {"link": b"alias/value", "alias": b"../outside"},
        {"link": b"alias", "alias": b"link"},
        {"link": b"link"},
        {"link": b"file/../file"},
        {"link": b"file/"},
        {"link": b"file/."},
        {"link": b"alias/../file", "alias": b"file"},
        {"link": b""},
        {"link": b"file\0ignored"},
        {"Alias": b".", "escape": b"alias/.."},
        {"\u00e9": b".", "escape": b"e\xcc\x81/.."},
        {"e\u0301": b".", "escape": b"\xc3\xa9/.."},
        {"link": b"DIRECTORY/value"},
        {"link": b"directory/VALUE"},
        {"link": b"FILE/.."},
        {"Alias": b".", "directory/nested": b"../alias/..", "link": b"directory/nested"},
        {"\u00e9": b"directory", "nested": b"e\xcc\x81/value", "link": b"nested"},
        {f"link{number}": f"link{number + 1}".encode() for number in range(66)},
    ],
)
def test_export_rejects_unsafe_link_components_before_writing(tmp_path, links):
    root, destination = tmp_path / "repository", tmp_path / "export"
    root.mkdir()
    destination.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "core.precomposeUnicode", "false")
    write(root, "file", "regular file\n")
    write(root, "directory/value", "inside\n")
    git(root, "add", ".")
    for name, target in links.items():
        blob = (
            subprocess.run(
                ["git", "hash-object", "-w", "--stdin"],
                cwd=root,
                input=target,
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        git(root, "update-index", "--add", "--cacheinfo", f"120000,{blob},{name}")
    with pytest.raises(gates.GateError, match="Symlink"):
        gates._export(root, gates.capture_target(root, "index"), destination)
    assert not list(destination.iterdir())


@pytest.mark.parametrize(
    ("left", "left_mode", "right", "right_mode"),
    [
        ("alias", "100644", "alias", "40000"),
        ("alias", "120000", "alias", "40000"),
        ("alias", "100644", "alias", "100644"),
        ("File.txt", "100644", "file.txt", "100644"),
        ("\u00e9", "100644", "e\u0301", "100644"),
        ("A", "100644", "a", "40000"),
        ("\u00e9", "120000", "e\u0301", "40000"),
        ("A", "40000", "a", "40000"),
        ("\u00e9", "40000", "e\u0301", "40000"),
    ],
)
def test_export_rejects_colliding_git_entries(tmp_path, left, left_mode, right, right_mode):
    root, destination = tmp_path / "repository", tmp_path / "export"
    root.mkdir()
    destination.mkdir()
    git(root, "init", "-b", "main")

    def store(object_type, content):
        return (
            subprocess.run(
                ["git", "hash-object", "-t", object_type, "--literally", "-w", "--stdin"],
                cwd=root,
                input=content,
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )

    blob = bytes.fromhex(store("blob", b"value"))
    entries = b""
    for name, mode, child_name in ((left, left_mode, "x"), (right, right_mode, "y")):
        oid = (
            bytes.fromhex(store("tree", f"100644 {child_name}\0".encode() + blob))
            if mode == "40000"
            else blob
        )
        entries += f"{mode} {name}\0".encode() + oid
    tree = store("tree", entries)
    with pytest.raises(gates.GateError, match="directory|Duplicate|portable"):
        gates._export(root, gates.Snapshot(tree, None, "index"), destination)
    assert not list(destination.iterdir())


def test_dangling_link_validation_ignores_outside_filesystem_links(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-b", "main")
    (root / "link").symlink_to("missing/value")
    git(root, "add", "link")
    snapshot = gates.capture_target(root, "index")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.mkfifo(outside / "value")
    for number, target in enumerate((outside, tmp_path / "absent")):
        (root / "missing").symlink_to(target)
        destination = tmp_path / f"export-{number}"
        destination.mkdir()
        gates._export(root, snapshot, destination)
        assert os.readlink(destination / "link") == "missing/value"
        assert not (destination / "missing").exists()
        (root / "missing").unlink()


@pytest.mark.parametrize(
    "name",
    ["pyproject.toml", "uv.lock", ".python-version", "tools/repo.py", "tests/test_fixture.py"],
)
def test_escaping_source_links_never_open_external_fifo(repository, tmp_path, name):
    sentinel = tmp_path / "external-fifo"
    os.mkfifo(sentinel)
    path = repository / name
    path.unlink()
    path.symlink_to(sentinel)
    git(repository, "add", name)
    result = command(repository, "make", "smoke-index", success=False, timeout=10)
    assert "Symlink target must be a nonempty relative path" in result.stderr


def test_linked_worktree_uses_common_preparation_and_installed_hooks(repository, tmp_path):
    linked = tmp_path / "linked worktree with spaces"
    git(repository, "worktree", "add", "-b", "linked", str(linked))
    command(linked, "make", "hooks")
    assert Path(git(linked, "config", "core.hooksPath")) == repository / ".git/repo-gates"
    command(linked, "make", "smoke-index")
    write(linked, "note.txt", "linked\n")
    git(linked, "add", "note.txt")
    write(linked, "tools/repo.py", "invalid Python (\n")
    write(linked, "Makefile", "$(error invalid working Makefile)\n")
    git(linked, "commit", "-m", "Verify linked worktree")
    assert (linked / "tools/repo.py").read_text() == "invalid Python (\n"
    assert git(linked, "show", "HEAD:note.txt") == "linked"


def change_dependency(root: Path, version: str):
    project = root / "pyproject.toml"
    project.write_text(project.read_text().replace("gate-dep==1.0", f"gate-dep=={version}"))
    with tempfile.TemporaryDirectory(prefix="repo-fixture-lock-") as directory:
        staging = Path(directory)
        for name in ("pyproject.toml", "uv.lock", ".python-version"):
            shutil.copyfile(root / name, staging / name)
        command(
            staging,
            "uv",
            "lock",
            "--offline",
            "--python",
            (root / ".python-version").read_text().strip(),
        )
        shutil.copyfile(staging / "uv.lock", root / "uv.lock")
    git(root, "add", "pyproject.toml", "uv.lock", ".python-version")


def fixture_dependency(root, wheelhouse, tmp_path, files, record_fields=None):
    index = tmp_path / "package-index"
    shutil.copytree(wheelhouse, index)
    distribution = "gate_dep-3.0.dist-info"
    wheel(
        index / "gate-dep",
        "gate_dep",
        "3.0",
        {
            "gate_dep.py": b"VERSION = '3.0'\n",
            f"{distribution}/METADATA": b"Metadata-Version: 2.1\nName: gate-dep\nVersion: 3.0\n",
            **files,
        },
        record_fields=record_fields,
    )
    write(index, "gate-dep/index.html", '<a href="gate_dep-3.0-py3-none-any.whl">gate-dep</a>')
    project = root / "pyproject.toml"
    project.write_text(project.read_text().replace(wheelhouse.as_uri(), index.as_uri()))
    change_dependency(root, "3.0")
    return gates.capture_target(root, "index")


def test_real_dependency_versions_select_separate_prepared_generations(repository):
    root = repository
    original = gates.verify(root, gates.capture_target(root, "HEAD"), "full")
    change_dependency(root, "2.0")
    changed = gates.capture_target(root, "index")
    with pytest.raises(gates.GateError, match="make deps TARGET=index"):
        gates.verify(root, changed, "smoke")
    prepared = gates.prepare(root, changed)
    assert prepared.identity != original.environment.identity
    assert prepared.python != original.environment.python
    write(
        root,
        "tests/test_fixture.py",
        PASSING.replace(
            "    assert True", "    import gate_dep\n\n    assert gate_dep.VERSION == '2.0'"
        ),
    )
    command(root, PYTHON.parent / "ruff", "format", "tests")
    git(root, "add", "tests")
    git(root, "restore", "--source=HEAD", "--worktree", "pyproject.toml", "uv.lock")
    assert gates.verify(root, gates.capture_target(root, "index"), "smoke").environment == prepared
    assert (
        gates.verify(root, gates.capture_target(root, "HEAD"), "full").environment
        == original.environment
    )


def test_changed_environment_fails_without_repair(repository):
    root = repository
    prepared = gates.prepare(root, gates.capture_target(root, "index"))
    shutil.rmtree(prepared.purelib / "gate_dep-1.0.dist-info")
    (prepared.purelib / "gate_dep.py").unlink()
    before = tree_bytes(prepared.python.parent.parent)
    with pytest.raises(gates.GateError, match="Prepare this snapshot"):
        gates.verify(root, gates.capture_target(root, "index"), "full")
    assert tree_bytes(prepared.python.parent.parent) == before


@pytest.mark.parametrize("installed", ["module", "executable"])
def test_seal_detects_same_length_corruption(repository, installed, tmp_path):
    prepared = gates.prepare(repository, gates.capture_target(repository, "index"))
    path = (
        prepared.purelib / "gate_dep.py"
        if installed == "module"
        else prepared.python.parent / "ruff"
    )
    metadata = next(
        prepared.purelib.glob(
            "gate_dep-*.dist-info" if installed == "module" else "ruff-*.dist-info"
        )
    )
    record = (metadata / "RECORD").read_bytes()
    original = path.read_bytes()
    mode = path.stat().st_mode
    sibling = tmp_path / "shared-installed-file"
    sibling.hardlink_to(path)
    replace_installed_file(path, original[:-1] + bytes([original[-1] ^ 1]))
    assert path.stat().st_size == len(original)
    assert path.stat().st_mode == mode
    assert sibling.read_bytes() == original
    assert (metadata / "RECORD").read_bytes() == record
    with pytest.raises(gates.GateError, match="Prepare this snapshot") as error:
        gates.verify(repository, gates.capture_target(repository, "index"), "full")
    assert "fingerprint mismatch" in str(error.value)
    assert path.read_bytes() != original
    assert (metadata / "RECORD").read_bytes() == record


@pytest.mark.parametrize(
    "record_case",
    [
        "short-row",
        "long-row",
        "malformed-csv",
        "empty-path",
        "invalid-size",
        "negative-size",
        "wrong-size",
        "unknown-hash",
        "malformed-hash",
        "wrong-hash",
    ],
)
def test_invalid_installed_record_reports_distribution_and_path(
    repository, record_case, monkeypatch
):
    row = ["gate_dep.py", "", ""]
    match record_case:
        case "short-row":
            row.pop()
        case "long-row":
            row.append("unexpected")
        case "empty-path":
            row[0] = ""
        case "invalid-size":
            row[2] = "not-a-size"
        case "negative-size":
            row[2] = "-1"
        case "wrong-size":
            row[2] = "0"
        case "unknown-hash":
            row[1] = "unknown_algorithm=AAAA"
        case "malformed-hash":
            row[1] = "sha256"
        case "wrong-hash":
            row[1] = "sha256=AAAA"
    output = io.StringIO()
    if record_case == "malformed-csv":
        output.write('"gate_dep.py,unterminated')
    else:
        csv.writer(output).writerow(row)
    snapshot = gates.capture_target(repository, "index")
    store = repository / ".git/prepared"
    before = tree_bytes(store)
    installed = []
    sync = gates._sync

    def install_invalid(root, prepared, env, *, check):
        sync(root, prepared, env, check=check)
        if not check:
            record = prepared.purelib / "gate_dep-1.0.dist-info/RECORD"
            replace_installed_file(record, output.getvalue().encode())
            installed.append(record)

    monkeypatch.setattr(gates, "_sync", install_invalid)
    with tempfile.TemporaryDirectory(prefix="repo-admission-") as directory:
        exported = Path(directory)
        gates._export(repository, snapshot, exported)
        with pytest.raises(gates.GateError, match="gate-dep") as error:
            gates._snapshot_operation(gates.SnapshotTask(snapshot, exported, store, "prepare"))
    assert "gate_dep.py" in str(error.value) or "RECORD" in str(error.value)
    assert installed[0].read_bytes() == output.getvalue().encode()
    assert {name: (store / name).read_bytes() for name in before} == before
    gates.verify(repository, snapshot, "smoke")


@pytest.mark.parametrize("fields", ["hash", "size", "neither", "sha512"])
def test_installed_record_accepts_independent_optional_fields(
    repository, wheelhouse, tmp_path, fields
):
    def record_fields(content, row):
        if fields in ("size", "neither"):
            row[1] = ""
        if fields in ("hash", "neither"):
            row[2] = ""
        if fields == "sha512":
            row[1] = (
                "sha512="
                + base64.urlsafe_b64encode(hashlib.sha512(content).digest()).rstrip(b"=").decode()
            )
        return row

    snapshot = fixture_dependency(repository, wheelhouse, tmp_path, {}, record_fields)
    prepared = gates.prepare(repository, snapshot)
    record = prepared.purelib / "gate_dep-3.0.dist-info/RECORD"
    before = record.read_bytes()
    row = next(
        row for row in csv.reader(io.StringIO(record.read_text())) if row[0] == "gate_dep.py"
    )
    assert bool(row[1]) == (fields in ("hash", "sha512"))
    assert bool(row[2]) == (fields in ("size", "sha512"))
    gates.verify(repository, snapshot, "smoke")
    assert record.read_bytes() == before


def test_stale_lock_and_missing_interpreter_stop_preparation_or_verification(repository):
    root = repository
    project = root / "pyproject.toml"
    project.write_text(project.read_text().replace("gate-dep==1.0", "gate-dep==2.0"))
    git(root, "add", "pyproject.toml")
    with pytest.raises(gates.GateError, match="locked"):
        gates.prepare(root, gates.capture_target(root, "index"))
    git(root, "restore", "--source=HEAD", "--staged", "--worktree", "pyproject.toml")
    write(root, ".python-version", "3.99.1\n")
    git(root, "add", ".python-version")
    with pytest.raises(gates.GateError, match="make deps TARGET=index"):
        gates.verify(root, gates.capture_target(root, "index"), "full")


def test_alternate_runtime_has_its_own_environment(repository, alternate_python):
    root = repository
    write(root, ".python-version", "3.13.12\n")
    project = root / "pyproject.toml"
    project.write_text(project.read_text().replace(">=3.12,<3.13", ">=3.13,<3.14"))
    change_dependency(root, "1.0")
    prepared = gates.prepare(root, gates.capture_target(root, "index"))
    result = gates.verify(root, gates.capture_target(root, "index"), "full")
    assert result.environment == prepared
    assert prepared.runtime.version == (3, 13, 12)


def test_preparation_installs_python_only_without_a_managed_pin(repository, tmp_path, monkeypatch):
    root = repository
    pin = (root / ".python-version").read_text().strip()
    run, installs = gates._run, []

    def record_install(args, cwd, **kwargs):
        if args[:3] == ["uv", "python", "install"]:
            installs.append(args[3])
            return b""
        return run(args, cwd, **kwargs)

    monkeypatch.setattr(gates, "_run", record_install)
    gates.prepare(root, gates.capture_target(root, "index"))
    assert installs == []
    write(root, ".python-version", "3.99.1\n")
    git(root, "add", ".python-version")
    with pytest.raises(gates.GateError, match="uv python find 3.99.1 --system --managed-python"):
        gates.prepare(root, gates.capture_target(root, "index"))
    assert installs == ["3.99.1"]
    git(root, "restore", "--source=HEAD", "--staged", "--worktree", ".python-version")
    unmanaged = tmp_path / "unmanaged"
    unmanaged.mkdir()
    (unmanaged / "python3.12").symlink_to(Path(sys._base_executable).resolve())
    monkeypatch.setenv("PATH", f"{unmanaged}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    with pytest.raises(gates.GateError, match=f"uv python find {pin} --system --managed-python"):
        gates.prepare(root, gates.capture_target(root, "index"))
    assert installs == ["3.99.1", pin]


def test_installation_is_explicit_and_failed_refresh_preserves_complete_gates(repository):
    root = repository
    installed = root / ".git/repo-gates"
    shutil.rmtree(installed)
    git(root, "config", "--unset", "core.hooksPath")
    result = command(root, "make", "deps", "TARGET=HEAD", success=False)
    assert "Run make hooks or make setup" in result.stderr
    command(root, "make", "setup", "TARGET=HEAD")
    assert sorted(path.name for path in installed.iterdir()) == [
        "post-rewrite",
        "pre-commit",
        "pre-merge-commit",
        "repo.py",
    ]
    stale = write(installed, "pre-push", "#!/bin/sh\nexit 1\n")
    stale.chmod(0o755)
    before = tree_bytes(installed)
    modes = {path.name: path.stat().st_mode for path in installed.iterdir()}
    command(root, "make", "hooks", "TARGET=HEAD")
    assert tree_bytes(installed) == {
        name: content for name, content in before.items() if name != "pre-push"
    }
    stale = write(installed, "pre-push", "#!/bin/sh\nexit 1\n")
    stale.chmod(0o755)
    runner = (root / "tools/repo.py").read_text()
    write(root, "tools/repo.py", "invalid selected runner (\n")
    git(root, "add", "tools/repo.py")
    write(root, "tools/repo.py", runner)
    command(root, "make", "hooks", "TARGET=index", success=False)
    assert tree_bytes(installed) == before
    write(root, "tools/repo.py", runner.replace("Installed repository gates", "Refreshed gates"))
    git(root, "add", "tools/repo.py")
    config = (root / ".git/config").read_bytes()
    lock = write(root, ".git/config.lock", "another writer\n")
    command(root, "make", "hooks", "TARGET=index", success=False)
    assert tree_bytes(installed) == before
    assert {path.name: path.stat().st_mode for path in installed.iterdir()} == modes
    assert (root / ".git/config").read_bytes() == config
    lock.unlink()
    command(root, "make", "hooks", "TARGET=index")
    refreshed = tree_bytes(installed)
    assert refreshed["repo.py"] != before["repo.py"]
    assert "pre-push" not in refreshed
    git(root, "restore", "--source=HEAD", "--staged", "--worktree", "tools/repo.py")
    command(root, "make", "verify", "TARGET=HEAD")
    assert tree_bytes(installed) == refreshed


@pytest.mark.parametrize("fail_second", [False, True])
def test_concurrent_linked_installers_publish_or_restore_complete_predecessor(
    repository, tmp_path, fail_second
):
    root = repository
    installed = root / ".git/repo-gates"
    original = (root / "tools/repo.py").read_text()
    targets, installers, expected = {}, {}, {}
    for writer in ("first", "second"):
        variant = original.replace(
            '"#!/bin/sh\\n"', f'"#!/bin/sh\\nGATE_INSTALLED_REVISION={writer}\\n"'
        ).replace("Installed repository gates", f"Installed {writer} repository gates")
        installers[writer] = write(tmp_path, f"{writer}.py", variant)
        write(root, "tools/repo.py", variant)
        git(root, "add", "tools/repo.py")
        targets[writer] = git(
            root,
            "commit-tree",
            gates.capture_target(root, "index").tree_oid,
            "-p",
            "HEAD",
            "-m",
            f"Select {writer} installer",
        )
        command(root, "make", "hooks", f"TARGET={targets[writer]}")
        expected[writer] = (
            tree_bytes(installed),
            {path.name: path.stat().st_mode for path in installed.iterdir()},
        )
    git(root, "restore", "--source=HEAD", "--staged", "--worktree", "tools/repo.py")
    command(root, "make", "hooks", "TARGET=HEAD")
    linked = tmp_path / "linked"
    git(root, "worktree", "add", "-b", "second-installer", str(linked), "HEAD")
    events = tmp_path / "events"
    os.mkfifo(events)
    for writer in targets:
        os.mkfifo(tmp_path / f"release-{writer}")
    observer = textwrap.dedent("""\
        import fcntl
        import os
        import runpy
        from pathlib import Path

        flock = fcntl.flock
        def observe(descriptor, operation):
            try:
                flock(descriptor, operation | fcntl.LOCK_NB)
            except BlockingIOError:
                with Path(os.environ["GATE_EVENTS"]).open("w") as stream:
                    stream.write(os.environ["GATE_WRITER"] + ":blocked\\n")
                flock(descriptor, operation)
        fcntl.flock = observe
        runpy.run_path(os.environ["GATE_INSTALLER"], run_name="__main__")
        """)
    for worktree in (root, linked):
        write(worktree, "tools/repo.py", observer)
    real_git = shutil.which("git")
    git_script = write(
        tmp_path,
        "git.py",
        textwrap.dedent(f"""\
        import os
        import sys
        from pathlib import Path

        arguments = sys.argv[1:]
        if arguments[:1] == ["--no-replace-objects"]:
            arguments = arguments[1:]
        if arguments[:3] == ["config", "--local", "core.hooksPath"]:
            writer = os.environ["GATE_WRITER"]
            with Path(os.environ["GATE_EVENTS"]).open("w") as stream:
                stream.write(writer + ":config\\n")
            with Path(os.environ["GATE_RELEASE"]).open("rb", buffering=0) as stream:
                assert stream.read(1) == b"x"
            if os.environ["GATE_FAIL"] == writer:
                sys.exit("Injected installer config failure")
        os.execv({real_git!r}, [{real_git!r}, *sys.argv[1:]])
        """),
    )
    fake_git = write(
        tmp_path,
        "bin/git",
        f"#!/bin/sh\nexec {shlex.quote(str(PYTHON))} -B -I -S "
        f'{shlex.quote(str(git_script))} "$@"\n',
    )
    fake_git.chmod(0o755)
    processes = {}
    with os.fdopen(os.open(events, os.O_RDWR), "rb", buffering=0) as stream:

        def event():
            assert select.select([stream], [], [], 30)[0], "Installer barrier did not arrive"
            return stream.readline().decode().strip()

        try:
            for writer, worktree in (("first", root), ("second", linked)):
                env = os.environ | {
                    "PATH": f"{fake_git.parent}{os.pathsep}{os.environ['PATH']}",
                    "GATE_WRITER": writer,
                    "GATE_INSTALLER": str(installers[writer]),
                    "GATE_EVENTS": str(events),
                    "GATE_RELEASE": str(tmp_path / f"release-{writer}"),
                    "GATE_FAIL": "second" if fail_second else "",
                }
                processes[writer] = subprocess.Popen(
                    ["make", "hooks", f"TARGET={targets[writer]}"],
                    cwd=worktree,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                assert event() == f"{writer}:{'config' if writer == 'first' else 'blocked'}"
            (tmp_path / "release-first").write_bytes(b"x")
            output, _ = processes["first"].communicate(timeout=30)
            assert processes["first"].returncode == 0, output
            assert event() == "second:config"
            (tmp_path / "release-second").write_bytes(b"x")
            output, _ = processes["second"].communicate(timeout=30)
            assert (processes["second"].returncode != 0) == fail_second, output
            if fail_second:
                assert "Injected installer config failure" in output
        finally:
            for process in processes.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=30)
    retained = "first" if fail_second else "second"
    assert (
        tree_bytes(installed),
        {path.name: path.stat().st_mode for path in installed.iterdir()},
    ) == expected[retained]
    assert git(root, "config", "core.hooksPath") == str(installed)
    command(linked, "make", "verify", f"TARGET={targets[retained]}")


def test_snapshot_imports_ignore_worktree_pythonpath_and_pth(
    repository, wheelhouse, tmp_path, monkeypatch, capfd
):
    root = repository
    marker = tmp_path / "pth-ran"
    injected = tmp_path / "injected"
    injected.mkdir()
    write(injected, "gate_value.py", "VALUE = True\n")
    snapshot = fixture_dependency(
        root,
        wheelhouse,
        tmp_path,
        {
            "inject.pth": (
                f"{injected}\nimport pathlib; pathlib.Path({str(marker)!r}).touch()\n"
            ).encode(),
        },
    )
    prepared = gates.prepare(root, snapshot)
    write(root, "gate_value.py", "VALUE = False\n")
    write(
        root,
        "tests/test_fixture.py",
        PASSING.replace("    assert True", "    from gate_value import VALUE\n\n    assert VALUE"),
    )
    git(root, "add", "gate_value.py", "tests/test_fixture.py")
    write(root, "gate_value.py", "VALUE = True\n")
    monkeypatch.setenv("PYTHONPATH", str(injected))
    before = tree_bytes(prepared.python.parent.parent)
    with pytest.raises(gates.GateError, match="failed"):
        gates.verify(root, gates.capture_target(root, "index"), "full")
    assert "assert VALUE" in capfd.readouterr().out
    assert not marker.exists()
    assert tree_bytes(prepared.python.parent.parent) == before


def test_captured_packages_and_modules_precede_prepared_dependency_names(
    repository, template, wheelhouse, tmp_path, capfd
):
    root = repository
    template_before = tree_bytes(template / ".git/prepared")
    marker = tmp_path / "dependency-tools-ran"
    snapshot = fixture_dependency(
        root,
        wheelhouse,
        tmp_path,
        {
            "tools/__init__.py": (
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
            ).encode(),
            "tools/repo.py": b'raise SystemExit("dependency runner executed")\n',
            "gate_value.py": b"VALUE = False\n",
            "src_value.py": b"VALUE = False\n",
        },
    )
    prepared = gates.prepare(root, snapshot)
    write(root, "gate_value.py", "VALUE = True\n")
    write(root, "src/src_value.py", "VALUE = True\n")
    write(
        root,
        "tests/test_fixture.py",
        PASSING.replace(
            "    assert True",
            "    import gate_value\n    import src_value\n\n"
            "    assert gate_value.VALUE and src_value.VALUE",
        ),
    )
    git(root, "add", "gate_value.py", "src", "tests")
    before = tree_bytes(prepared.python.parent.parent)
    gates.verify(root, gates.capture_target(root, "index"), "smoke")
    write(root, "tests/test_fixture.py", FAILING)
    git(root, "add", "tests")
    with pytest.raises(gates.GateError, match="failed"):
        gates.verify(root, gates.capture_target(root, "index"), "smoke")
    assert "assert 1 == 2" in capfd.readouterr().out
    assert not marker.exists()
    assert tree_bytes(prepared.python.parent.parent) == before
    assert tree_bytes(template / ".git/prepared") == template_before


@pytest.mark.parametrize(
    ("suite", "shadow"),
    [
        ("smoke-index", "pytest.py"),
        ("verify", "src/_pytest/__init__.py"),
        ("coverage", "src/coverage.py"),
        ("coverage", "pytest/__init__.py"),
    ],
)
def test_verification_tools_ignore_captured_shadows(repository, tmp_path, suite, shadow):
    root = repository
    if suite == "coverage":
        project = root / "pyproject.toml"
        project.write_text(project.read_text().replace("dev = [", 'dev = ["coverage", '))
        change_dependency(root, "1.0")
        command(root, "make", "deps", "TARGET=index")
    marker = tmp_path / "shadow-ran"
    write(
        root,
        shadow,
        f"from pathlib import Path\n\nPath(\n    {json.dumps(str(marker))},\n).touch()\n"
        "raise SystemExit(0)\n",
    )
    write(root, "gate_value.py", "VALUE = 'captured root'\n")
    write(root, "src/src_value.py", 'VALUE = "captured src"\n')
    failing = PASSING.replace(
        "    assert True",
        "    import gate_value\n    import src_value\n\n"
        '    assert gate_value.VALUE == "captured root"\n'
        '    assert src_value.VALUE == "captured src"\n'
        '    assert 1 == 2, "deliberate captured failure"',
    )
    write(root, "tests/test_fixture.py", failing)
    git(root, "add", ".")
    index = index_path(root).read_bytes()
    result = command(root, "make", suite, "TARGET=index", success=False)
    assert "deliberate captured failure" in result.stdout + result.stderr
    assert not marker.exists()
    assert index_path(root).read_bytes() == index
    write(root, "tests/test_fixture.py", failing.replace("assert 1 == 2", "assert 1 == 1"))
    git(root, "add", "tests/test_fixture.py")
    result = command(root, "make", suite, "TARGET=index")
    assert "1 passed" in result.stdout
    assert not marker.exists()


@pytest.mark.parametrize("shadow", ["pytest.py", "src/pytest/__init__.py"])
def test_commit_runs_locked_pytest_despite_shadows(repository, shadow):
    root = repository
    write(root, shadow, "raise SystemExit(0)\n")
    write(root, "tests/test_fixture.py", FAILING)
    git(root, "add", ".")
    before = git(root, "rev-parse", "HEAD")
    result = command(root, "git", "commit", "-m", "Reject failing staged assertion", success=False)
    assert "assert 1 == 2" in result.stdout + result.stderr
    assert git(root, "rev-parse", "HEAD") == before


def test_missing_captured_tools_marker_cannot_run_dependency_suite(
    repository, template, wheelhouse, tmp_path
):
    root = repository
    template_before = tree_bytes(template / ".git/prepared")
    marker = tmp_path / "dependency-tools-ran"
    snapshot = fixture_dependency(
        root,
        wheelhouse,
        tmp_path,
        {
            "tools/__init__.py": (
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
            ).encode(),
            "tools/repo.py": b"raise SystemExit(0)\n",
        },
    )
    gates.prepare(root, snapshot)
    (root / "tools/__init__.py").unlink()
    write(root, "tests/test_fixture.py", FAILING)
    git(root, "add", "-u")
    tree = gates.capture_target(root, "index").tree_oid
    oid = git(root, "commit-tree", tree, "-p", "HEAD", "-m", "Omit captured package marker")
    git(root, "branch", "unsupported", oid)
    git(root, "restore", "--source=HEAD", "--staged", "--worktree", ".")
    result = command(root, "make", "verify", "TARGET=unsupported", success=False)
    assert "tools/__init__.py" in result.stdout + result.stderr
    assert not marker.exists()
    assert tree_bytes(template / ".git/prepared") == template_before


@pytest.fixture
def remote(repository, tmp_path):
    remote = tmp_path / "remote.git"
    command(tmp_path, "git", "init", "--bare", str(remote))
    git(repository, "remote", "add", "origin", str(remote))
    git(repository, "push", "--no-verify", "origin", "main")
    return remote


def failing_commit(root: Path) -> str:
    write(root, "tests/test_fixture.py", FAILING)
    git(root, "add", "tests/test_fixture.py")
    tree = gates.capture_target(root, "index").tree_oid
    commit = git(root, "commit-tree", tree, "-p", "HEAD", "-m", "Fixture failing commit")
    git(root, "restore", "--source=HEAD", "--staged", "--worktree", "tests/test_fixture.py")
    return commit


def test_verified_branches_own_policy_runtime_and_dependencies(repository, alternate_python):
    root = repository
    runner = root / "tools/repo.py"
    original = runner.read_text()
    marker = (
        '    print("Policy strict runtime=" + ".".join(map(str, sys.version_info[:3]))\n'
        '          + " dependency=" + importlib.metadata.version("gate-dep"), flush=True)\n'
    )
    runner.write_text(
        original.replace(
            "def _run_suite(suite: Suite, receipt: Path | None = None) -> None:\n",
            "def _run_suite(suite: Suite, receipt: Path | None = None) -> None:\n" + marker,
        )
    )
    write(
        root,
        "tests/test_fixture.py",
        PASSING.replace(
            "    assert True", "    import gate_dep\n\n    assert gate_dep.VERSION == '1.0'"
        ),
    )
    command(root, PYTHON.parent / "ruff", "format", "tools", "tests")
    git(root, "add", "tools", "tests")
    git(root, "switch", "-c", "strict-policy")
    git(root, "commit", "-m", "Select strict policy")
    strict_oid = git(root, "rev-parse", "HEAD")
    strict_runner = runner.read_text()
    strict = gates.verify(root, gates.capture_target(root, strict_oid), "full")
    runner.write_text(
        strict_runner.replace("Policy strict runtime=", "Policy relaxed runtime=").replace(
            '        source = package.get("source", {})\n',
            '        source = package.get("source", {})\n'
            '        if source == {"editable": "."} and '
            'package["name"] == project["project"]["name"]:\n'
            "            continue\n",
        )
        + "\ntype WorkerRuntime[T = str] = list[T]\n"
    )
    project = root / "pyproject.toml"
    project.write_text(
        project.read_text()
        .replace(">=3.12,<3.13", ">=3.13,<3.14")
        .replace("package = false", "package = true")
        .replace('target-version = "py312"', 'target-version = "py313"')
    )
    write(root, ".python-version", "3.13.12\n")
    change_dependency(root, "2.0")
    write(
        root,
        "tests/test_fixture.py",
        PASSING.replace(
            "    assert True", "    import gate_dep\n\n    assert gate_dep.VERSION == '2.0'"
        ),
    )
    command(root, PYTHON.parent / "ruff", "format", "tools", "tests")
    relaxed_runner = runner.read_text()
    runner.write_text(strict_runner)
    git(root, "add", "tools", "tests")
    with pytest.raises(gates.GateError, match="Unsupported dependency source"):
        gates.prepare(root, gates.capture_target(root, "index"))
    runner.write_text(relaxed_runner)
    git(root, "add", "tools/repo.py")
    tree = gates.capture_target(root, "index").tree_oid
    relaxed_oid = git(root, "commit-tree", tree, "-p", strict_oid, "-m", "Select relaxed policy")
    git(root, "branch", "relaxed-policy", relaxed_oid)
    write(root, "tests/test_fixture.py", FAILING)
    git(root, "add", "tests")
    bad_tree = gates.capture_target(root, "index").tree_oid
    bad_oid = git(root, "commit-tree", bad_tree, "-p", relaxed_oid, "-m", "Fail relaxed suite")
    git(root, "branch", "failing-policy", bad_oid)
    git(root, "restore", "--source=HEAD", "--staged", "--worktree", ".")
    missing = command(root, "make", "verify", "TARGET=relaxed-policy", success=False)
    assert "make deps TARGET=relaxed-policy" in missing.stdout + missing.stderr
    relaxed = gates.prepare(root, gates.capture_target(root, relaxed_oid))
    assert relaxed.runtime.version == (3, 13, 12)
    assert relaxed.identity != strict.environment.identity
    pointer = root / ".git/prepared" / f"{relaxed.identity}.json"
    original_pointer = pointer.read_bytes()
    pointer.write_text(
        json.dumps(
            json.loads(original_pointer)
            | {"generation": str(strict.environment.python.parent.parent)}
        )
    )
    mismatch = command(root, "make", "verify", "TARGET=relaxed-policy", success=False)
    assert "Unsupported external prepared symlink" in mismatch.stdout + mismatch.stderr
    pointer.write_bytes(original_pointer)
    output = "".join(
        result.stdout + result.stderr
        for result in (
            command(root, "make", "verify", f"TARGET={target}")
            for target in ("strict-policy", "relaxed-policy")
        )
    )
    assert f"Passed full for {strict_oid}" in output
    assert f"Passed full for {relaxed_oid}" in output
    assert "Policy strict runtime=3.12.13 dependency=1.0" in output
    assert "Policy relaxed runtime=3.13.12 dependency=2.0" in output
    assert git(root, "branch", "--show-current") == "strict-policy"
    result = command(root, "make", "verify", "TARGET=failing-policy", success=False)
    assert "assert 1 == 2" in result.stdout + result.stderr


@pytest.mark.parametrize("missing", ["runner", "pin", "private-command", "protocol", "result"])
def test_snapshot_without_worker_contract_cannot_verify(repository, missing):
    root = repository
    if missing in ("runner", "pin"):
        (root / ("tools/repo.py" if missing == "runner" else ".python-version")).unlink()
    else:
        runner = root / "tools/repo.py"
        if missing == "private-command":
            runner.write_text(runner.read_text().replace('"_snapshot"', '"_unavailable"'))
        elif missing == "protocol":
            runner.write_text(
                runner.read_text().replace('"repo-snapshot-v1"', '"repo-snapshot-v2"')
            )
        else:
            runner.write_text("raise SystemExit(0)\n")
    git(root, "add", "-u")
    tree = gates.capture_target(root, "index").tree_oid
    oid = git(root, "commit-tree", tree, "-p", "HEAD", "-m", "Omit snapshot contract")
    git(root, "branch", "unsupported", oid)
    git(root, "restore", "--source=HEAD", "--staged", "--worktree", ".")
    result = command(root, "make", "verify", "TARGET=unsupported", success=False)
    assert "make deps TARGET=unsupported" in result.stdout + result.stderr


def diverge_failing_branch(root: Path) -> None:
    git(root, "switch", "-c", "failing")
    write(root, "tests/test_fixture.py", FAILING)
    git(root, "commit", "--no-verify", "-am", "Fail the fixture test")
    git(root, "switch", "main")
    write(root, "note.txt", "main\n")
    git(root, "add", "note.txt")
    git(root, "commit", "--no-verify", "-m", "Advance main")


def test_merge_commit_hook_blocks_a_merge_whose_result_fails_smoke(repository):
    root = repository
    diverge_failing_branch(root)
    head = git(root, "rev-parse", "HEAD")
    result = command(root, "git", "merge", "--no-edit", "failing", success=False)
    assert "assert 1 == 2" in result.stdout + result.stderr
    assert git(root, "rev-parse", "HEAD") == head


def test_rebase_hook_reports_a_rebased_branch_that_fails_smoke(repository):
    root = repository
    diverge_failing_branch(root)
    git(root, "switch", "failing")
    result = command(root, "git", "rebase", "main")
    output = result.stdout + result.stderr
    assert "assert 1 == 2" in output
    assert "The rebased branch fails the smoke suite" in output
    assert git(root, "rev-parse", "HEAD^") == git(root, "rev-parse", "main")
    amended = command(root, "git", "commit", "--amend", "--no-verify", "-m", "Amend the failure")
    assert "Suite smoke" not in amended.stdout + amended.stderr


@pytest.fixture
def pr_boundary(repository, remote, tmp_path, monkeypatch):
    root = repository
    git(root, "switch", "-c", "feature")
    write(root, "note.txt", "feature\n")
    git(root, "add", "note.txt")
    git(root, "commit", "--no-verify", "-m", "Prepare feature")
    title = write(tmp_path, "title.txt", "Verify proposed snapshots\n")
    body = write(tmp_path, "body.txt", "Keep exact snapshots verified.\n\nChecks passed.\n")
    state_file = tmp_path / "gh-state.json"
    state_file.write_text(
        json.dumps(
            {
                "remote": str(remote),
                "other": git(root, "rev-parse", "main"),
                "title": str(title),
                "body": str(body),
            }
        )
    )
    fake = write(
        tmp_path,
        "bin/gh",
        f"#!/bin/sh\nexec {shlex.quote(str(PYTHON))} -B -I -S "
        f'{shlex.quote(str(ROOT / "tests/fixtures/fake_gh.py"))} "$@"\n',
    )
    fake.chmod(0o755)
    ssh = write(tmp_path, "bin/local-ssh", '#!/bin/sh\nexec git-receive-pack "$FAKE_REMOTE"\n')
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GIT_SSH_COMMAND", shlex.quote(str(ssh)))
    monkeypatch.setenv("GIT_SSH_VARIANT", "ssh")
    monkeypatch.setenv("FAKE_REMOTE", str(remote))
    monkeypatch.setenv("FAKE_GH_STATE", str(state_file))
    git(root, "remote", "set-url", "--push", "origin", "git@github.com:example/research.git")
    return root, remote, state_file, gates.PrRequest("feature", "", title, body)


def gh_calls(state_file: Path):
    calls = state_file.with_suffix(".calls")
    return [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []


def set_gh_state(state_file: Path, **values):
    state_file.write_text(json.dumps(json.loads(state_file.read_text()) | values))


@pytest.mark.parametrize(
    "invalid",
    [
        "lower-title",
        "type-title",
        "breaking-type-title",
        "breaking-scoped-title",
        "nul-title",
        "empty-body",
        "invalid-utf8",
        "head",
        "base",
        "same-branch",
        "repository",
    ],
)
def test_invalid_pr_metadata_never_pushes(pr_boundary, invalid):
    from dataclasses import replace

    root, remote, state_file, request = pr_boundary
    match invalid:
        case "lower-title":
            request.title_file.write_text("fix something")
        case "type-title":
            request.title_file.write_text("Fix: something")
        case "breaking-type-title":
            request.title_file.write_text("Fix!: something")
        case "breaking-scoped-title":
            request.title_file.write_text("Fix(scope)!: something")
        case "nul-title":
            request.title_file.write_text("Verify\0snapshot")
        case "empty-body":
            request.body_file.write_text("   \n")
        case "invalid-utf8":
            request.body_file.write_bytes(b"\xff")
        case "head":
            request = replace(request, head="missing-head")
        case "base":
            request = replace(request, base="missing-base")
        case "same-branch":
            request = replace(request, base="feature")
        case "repository":
            request = replace(request, repository="different/repository")
    before = git(root, "ls-remote", str(remote))
    result = command(
        root,
        "make",
        "pr",
        f"HEAD={request.head}",
        f"BASE={request.base}",
        f"TITLE_FILE={request.title_file}",
        f"BODY_FILE={request.body_file}",
        f"REPOSITORY={request.repository}",
        success=False,
    )
    assert "Repository gate failed" in result.stderr
    assert git(root, "ls-remote", str(remote)) == before
    assert not any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))
    if invalid == "nul-title":
        assert gh_calls(state_file) == []


@pytest.mark.parametrize("failure", ["auth", "repository", "base", "pr-list"])
def test_failed_pr_lookup_never_pushes(pr_boundary, failure):
    root, remote, state_file, request = pr_boundary
    set_gh_state(state_file, fail=failure)
    before = git(root, "ls-remote", str(remote))
    with pytest.raises(gates.GateError, match="Injected"):
        gates.create_pr(root, request)
    assert git(root, "ls-remote", str(remote)) == before
    assert not any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))


@pytest.mark.parametrize(
    ("stage", "response"),
    [
        ("base", None),
        ("base", {"name": "another-branch", "commit": {"sha": "0" * 40}}),
        ("pr-list", None),
        ("pr-list", []),
        ("pr-list", [None]),
        ("pr-list", [[None]]),
        ("pr-list", [[{"html_url": "url"}]]),
        ("pr-list", [[{"html_url": "url", "head": None}]]),
        ("pr-list", [[{"html_url": "url", "head": {"ref": "feature", "repo": None}}]]),
        ("pr-list", [[{"html_url": "url", "head": {"ref": "feature", "repo": {}}}]]),
        (
            "pr-list",
            [[{"html_url": "url", "head": {"ref": "feature", "repo": {"full_name": 1}}}]],
        ),
        (
            "pr-list",
            [[{"html_url": "url", "head": {"ref": "feature", "repo": {"full_name": ""}}}]],
        ),
        (
            "pr-list",
            [[{"html_url": "url", "head": {"ref": "feature", "repo": {"full_name": "bad"}}}]],
        ),
        (
            "pr-list",
            [
                [
                    {
                        "html_url": "url",
                        "head": {"ref": "feature", "repo": {"full_name": "example/research"}},
                    },
                    None,
                ]
            ],
        ),
    ],
)
def test_malformed_pr_lookup_never_pushes(pr_boundary, stage, response):
    root, remote, state_file, request = pr_boundary
    set_gh_state(state_file, outputs={stage: json.dumps(response)})
    before = git(root, "ls-remote", str(remote))
    with pytest.raises(gates.GateError, match="Invalid GitHub"):
        gates.create_pr(root, request)
    assert git(root, "ls-remote", str(remote)) == before
    assert not any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))


@pytest.mark.parametrize(
    "response",
    [
        "command-failure",
        "not-json",
        "null",
        "{}",
        '{"nameWithOwner": null}',
        '{"nameWithOwner": []}',
        '{"nameWithOwner": ""}',
        '{"nameWithOwner": "example/research"}',
        '{"nameWithOwner": "example/research", "defaultBranchRef": null}',
        '{"nameWithOwner": "example/research", "defaultBranchRef": {"name": 1}}',
        '{"nameWithOwner": "example/research", "defaultBranchRef": {"name": ""}}',
    ],
)
def test_invalid_repository_lookup_stops_pr_wrapper_before_push(pr_boundary, response):
    root, remote, state_file, request = pr_boundary
    if response == "command-failure":
        set_gh_state(state_file, fail="repository")
    else:
        set_gh_state(state_file, outputs={"repository": response})
    before = git(root, "ls-remote", str(remote))
    result = command(
        root,
        "make",
        "pr",
        "HEAD=feature",
        f"BASE={request.base}",
        f"TITLE_FILE={request.title_file}",
        f"BODY_FILE={request.body_file}",
        success=False,
    )
    assert "Repository gate failed" in result.stderr
    assert "Traceback" not in result.stderr
    expected = "Injected repository" if response == "command-failure" else "Invalid GitHub"
    assert expected in result.stderr
    assert git(root, "ls-remote", str(remote)) == before
    assert not any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))


def test_explicit_pr_base_does_not_require_default_branch_metadata(pr_boundary):
    root, remote, state_file, request = pr_boundary
    set_gh_state(state_file, outputs={"repository": '{"nameWithOwner": "example/research"}'})
    result = command(
        root,
        "make",
        "pr",
        "HEAD=feature",
        "BASE=main",
        f"TITLE_FILE={request.title_file}",
        f"BODY_FILE={request.body_file}",
    )
    assert "Created https://github.com/example/research/pull/10" in result.stdout
    assert git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == git(
        root, "rev-parse", "feature"
    )
    created = next(call for call in gh_calls(state_file) if call["args"][:2] == ["pr", "create"])
    assert created["args"][created["args"].index("--base") + 1] == "main"


@pytest.mark.parametrize(
    "failure",
    [
        "command",
        "json",
        "null",
        "missing-head",
        "invalid-head",
        "empty-head",
        "wrong-head",
        "missing-url",
        "invalid-url",
        "empty-url",
        "wrong-url",
    ],
)
def test_post_create_lookup_failure_keeps_existing_pr_url(pr_boundary, failure):
    root, remote, state_file, request = pr_boundary
    expected = git(root, "rev-parse", "feature")
    url = "https://github.com/example/research/pull/10"
    response = {
        "url": url,
        "headRefOid": expected,
        "headRefName": "feature",
        "headRepository": {"nameWithOwner": "example/research"},
        "baseRefName": "main",
    }
    match failure:
        case "missing-head":
            del response["headRefOid"]
        case "invalid-head":
            response["headRefOid"] = 1
        case "empty-head":
            response["headRefOid"] = ""
        case "wrong-head":
            response["headRefOid"] = git(root, "rev-parse", "main")
        case "missing-url":
            del response["url"]
        case "invalid-url":
            response["url"] = 1
        case "empty-url":
            response["url"] = ""
        case "wrong-url":
            response["url"] = "https://github.com/example/research/pull/11"
    if failure == "command":
        set_gh_state(state_file, fail="pr-view")
    else:
        output = {"json": "not-json", "null": "null"}.get(failure, json.dumps(response))
        set_gh_state(state_file, outputs={"pr-view": output})
    result = command(
        root,
        "make",
        "pr",
        "HEAD=feature",
        f"TITLE_FILE={request.title_file}",
        f"BODY_FILE={request.body_file}",
        success=False,
    )
    assert f"PR {url} exists; verification incomplete" in result.stderr
    assert "Traceback" not in result.stderr
    if failure == "command":
        assert "Injected pr-view failure" in result.stderr
    if failure == "json":
        assert "Expecting value" in result.stderr
    assert any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))
    assert git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == expected


@pytest.mark.parametrize(
    "failure",
    ["push", "duplicate", "wrong-repository", "move-local", "move-before-create"],
)
def test_pr_checks_push_and_branch_movement_stop_creation(pr_boundary, failure):
    root, remote, state_file, request = pr_boundary
    match failure:
        case "push":
            hook = write(remote, "hooks/pre-receive", "#!/bin/sh\nexit 1\n")
            hook.chmod(0o755)
        case "duplicate":
            set_gh_state(state_file, existing=True)
        case "wrong-repository":
            set_gh_state(state_file, repository="someone/else")
        case _:
            set_gh_state(state_file, mode=failure)
    before = git(root, "ls-remote", str(remote))
    with pytest.raises(gates.GateError):
        gates.create_pr(root, request)
    if failure != "move-before-create":
        assert git(root, "ls-remote", str(remote)) == before
    assert not any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))


def test_pr_creation_reports_unverified_head_after_concurrent_move(pr_boundary):
    root, _, state_file, request = pr_boundary
    set_gh_state(state_file, mode="move-after-create")
    with pytest.raises(
        gates.GateError, match="pull/10 exists; verification incomplete.*unverified head"
    ):
        gates.create_pr(root, request)
    assert any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))


def test_pr_freezes_metadata_and_pushes_captured_commit(pr_boundary):
    root, remote, state_file, request = pr_boundary
    title = "Verify $(touch injected) and `touch injected-too`\n"
    request.title_file.write_text(title)
    body = request.body_file.read_text()
    expected = git(root, "rev-parse", "feature")
    set_gh_state(state_file, mode="mutate-metadata")
    result = command(
        root,
        "make",
        "pr",
        "HEAD=feature",
        f"TITLE_FILE={request.title_file}",
        f"BODY_FILE={request.body_file}",
    )
    assert f"at commit {expected}" in result.stdout
    assert git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == expected
    created = next(call for call in gh_calls(state_file) if call["args"][:2] == ["pr", "create"])
    assert created["body"] == body
    assert created["args"][created["args"].index("--title") + 1] == title.strip()
    assert "--assignee" not in created["args"]
    assert json.loads(state_file.read_text())["assigned"] is True
    assert not (root / "injected").exists()
    assert not (root / "injected-too").exists()


def test_pr_push_uses_validated_destination_after_remote_changes(pr_boundary, tmp_path):
    root, remote, state_file, request = pr_boundary
    alternate = tmp_path / "alternate.git"
    command(tmp_path, "git", "init", "--bare", alternate)
    ssh = tmp_path / "bin/local-ssh"
    ssh.write_text(
        '#!/bin/sh\ncase "$*" in\n'
        f"  *example/other.git*) exec git-receive-pack {shlex.quote(str(alternate))} ;;\n"
        f"  *) exec git-receive-pack {shlex.quote(str(remote))} ;;\nesac\n"
    )
    destination = "git@github.com:example/other.git"
    expected = git(root, "rev-parse", "feature")
    before = git(root, "ls-remote", str(alternate))
    set_gh_state(state_file, mode="move-pushurl", push_url=destination)
    result = command(
        root,
        "make",
        "pr",
        f"HEAD={request.head}",
        f"BASE={request.base}",
        f"TITLE_FILE={request.title_file}",
        f"BODY_FILE={request.body_file}",
    )
    assert git(root, "remote", "get-url", "--push", "origin") == destination
    assert git(root, "ls-remote", str(alternate)) == before
    assert git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == expected
    assert f"at commit {expected}" in result.stdout
    assert any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))


@pytest.mark.parametrize(
    "head_repository", ["fork/research", "example/other", "EXAMPLE/RESEARCH", "later-page"]
)
def test_duplicate_pr_uses_head_repository_identity(pr_boundary, head_repository):
    root, remote, state_file, request = pr_boundary
    set_gh_state(state_file, existing=True, existing_head=head_repository)
    duplicate = head_repository in ("EXAMPLE/RESEARCH", "later-page")
    if head_repository == "later-page":
        entries = [
            {
                "html_url": f"https://github.com/example/research/pull/{number + 1}",
                "base": {"ref": "main"},
                "head": {
                    "ref": "feature",
                    "repo": {"full_name": f"example/fork-{number}"},
                },
            }
            for number in range(30)
        ]
        own = {
            "html_url": "https://github.com/example/research/pull/30",
            "base": {"ref": "main"},
            "head": {"ref": "feature", "repo": {"full_name": "example/research"}},
        }
        set_gh_state(state_file, pages=[entries, [own]])
    before = git(root, "ls-remote", str(remote))
    result = command(
        root,
        "make",
        "pr",
        f"HEAD={request.head}",
        f"BASE={request.base}",
        f"TITLE_FILE={request.title_file}",
        f"BODY_FILE={request.body_file}",
        success=not duplicate,
    )
    if duplicate:
        assert "An open PR already exists" in result.stderr
        assert git(root, "ls-remote", str(remote)) == before
    else:
        assert "Created https://github.com/example/research/pull/10" in result.stdout
        assert git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == git(
            root, "rev-parse", "feature"
        )
    assert any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file)) != duplicate


def test_pr_commands_bind_github_host(pr_boundary, monkeypatch):
    root, remote, state_file, request = pr_boundary
    set_gh_state(state_file, enforce_host=True)
    monkeypatch.setenv("GH_HOST", "enterprise.example.invalid")
    result = command(
        root,
        "make",
        "pr",
        f"HEAD={request.head}",
        f"BASE={request.base}",
        f"TITLE_FILE={request.title_file}",
        f"BODY_FILE={request.body_file}",
    )
    assert "Created https://github.com/example/research/pull/10" in result.stdout
    calls = gh_calls(state_file)
    assert {tuple(call["args"][:2]) for call in calls if call["args"][0] != "api"} == {
        ("auth", "status"),
        ("repo", "view"),
        ("pr", "create"),
        ("pr", "view"),
        ("pr", "edit"),
    }
    assert any(call["args"][0] == "api" for call in calls)
    assert all(call["host"] == "github.com" for call in calls)
    assert git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == git(
        root, "rev-parse", "feature"
    )


@pytest.fixture
def merge_boundary(pr_boundary):
    root, remote, state_file, _ = pr_boundary
    git(root, "push", str(remote), "feature")
    set_gh_state(
        state_file,
        pull={
            "number": 10,
            "state": "OPEN",
            "isDraft": False,
            "headRefName": "feature",
            "headRepository": {"nameWithOwner": "example/research"},
            "baseRefName": "main",
            "url": "https://github.com/example/research/pull/10",
        },
    )
    return root, remote, state_file


def landing_calls(state_file: Path) -> list[list[str]]:
    return [
        call["args"]
        for call in gh_calls(state_file)
        if call["args"][:2] in (["pr", "comment"], ["pr", "merge"])
    ]


def remote_commit(remote: Path, revision: str) -> str:
    return git(remote, "--git-dir", str(remote), "rev-parse", revision)


def advance_main(root: Path, remote: Path, worktree: Path, name: str, text: str) -> str:
    git(root, "worktree", "add", "--detach", str(worktree), "main")
    write(worktree, name, text)
    git(worktree, "add", name)
    git(worktree, "commit", "--no-verify", "-m", f"Change {name} on main")
    git(worktree, "push", str(remote), "HEAD:main")
    return git(worktree, "rev-parse", "HEAD")


@contextmanager
def started_merge(root: Path):
    process = subprocess.Popen(
        ["make", "merge", "PR=10"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        process.stdout.close()


def read_until(process: subprocess.Popen, marker: bytes, timeout=180) -> bytes:
    output = b""
    deadline = time.monotonic() + timeout
    while marker not in output:
        assert time.monotonic() < deadline and process.poll() is None, output.decode()
        if select.select([process.stdout], [], [], 1)[0]:
            output += os.read(process.stdout.fileno(), 65536)
    return output


def test_merge_posts_the_passing_run_then_merges_the_verified_head(merge_boundary):
    root, remote, state_file = merge_boundary
    head = git(root, "rev-parse", "HEAD")
    result = command(root, "make", "merge", "PR=10", timeout=300)
    comment, merge = landing_calls(state_file)
    assert comment[:3] == ["pr", "comment", "https://github.com/example/research/pull/10"]
    body = comment[comment.index("--body") + 1]
    assert body.startswith(f"Passed full for {head} in ")
    assert body in result.stdout
    assert merge[:3] == ["pr", "merge", "https://github.com/example/research/pull/10"]
    assert merge[merge.index("--match-head-commit") + 1] == head
    assert remote_commit(remote, "main^2") == head
    merged = remote_commit(remote, "main")
    assert f"Merged https://github.com/example/research/pull/10 at {head} as {merged}" in (
        result.stdout
    )


@pytest.mark.parametrize(
    ("paths", "suite"),
    [
        (("docs/guide/notes.md", "README.md", ".claude/rules/notes.txt"), "smoke"),
        (("docs/guide/notes.md", "tests/fixtures/notes.md"), "full"),
        (("docs/guide/image.png",), "full"),
    ],
)
def test_merge_runs_smoke_only_for_documentation(merge_boundary, paths, suite):
    root, remote, state_file = merge_boundary
    git(root, "reset", "--hard", "main")
    for path in paths:
        write(root, path, "notes\n")
    git(root, "add", *paths)
    git(root, "commit", "--no-verify", "-m", "Change documentation")
    git(root, "push", "--force", str(remote), "feature")
    head = git(root, "rev-parse", "HEAD")
    result = command(root, "make", "merge", "PR=10", timeout=300)
    assert f"Passed {suite} for {head}" in result.stdout
    assert f"Suite {suite}:" in result.stdout
    assert remote_commit(remote, "main^2") == head


def test_failing_merge_run_reports_each_failure_live_and_never_merges(
    merge_boundary, tmp_path, monkeypatch
):
    root, remote, state_file = merge_boundary
    release, runs = tmp_path / "release", tmp_path / "runs"
    monkeypatch.setenv("GATE_TEST_RELEASE", str(release))
    monkeypatch.setenv("GATE_TEST_RUNS", str(runs))
    write(
        root,
        "tests/test_fixture.py",
        textwrap.dedent("""\
            import os
            import time
            from pathlib import Path

            import pytest


            @pytest.mark.parametrize("case", range(8))
            def test_value(case):
                if case == 5:
                    deadline = time.monotonic() + 180
                    while not Path(os.environ["GATE_TEST_RELEASE"]).exists():
                        assert time.monotonic() < deadline, "failure was not printed"
                        time.sleep(0.05)
                with Path(os.environ["GATE_TEST_RUNS"]).open("a") as log:
                    log.write(f"{case}\\n")
                assert case != 0, "deliberate early failure"
            """),
    )
    git(root, "commit", "--no-verify", "-am", "Fail one shard early")
    git(root, "push", str(remote), "feature")
    main = remote_commit(remote, "main")
    with started_merge(root) as process:
        early = read_until(process, b"shard 0: FAILED tests/test_fixture.py::test_value[0]\n")
        assert b"Passed" not in early
        release.touch()
        rest, _ = process.communicate(timeout=300)
    output = (early + rest).decode()
    assert process.returncode != 0
    assert sorted(runs.read_text().split()) == [str(case) for case in range(8)]
    summary = output.partition("Suite full failed 1 of 3 steps:\n")[2].splitlines()
    assert summary[:2] == [
        "pytest: shard 0: FAILED tests/test_fixture.py::test_value[0]",
        "pytest shards failed with exits [1, 0, 0, 0]",
    ]
    assert landing_calls(state_file) == []
    assert remote_commit(remote, "main") == main


def test_merge_waits_for_the_merge_lock(merge_boundary):
    root, remote, state_file = merge_boundary
    head = git(root, "rev-parse", "HEAD")
    with (root / ".git/repo-merge.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.write("PR 9 in another worktree since noon\n")
        lock.flush()
        with started_merge(root) as process:
            read_until(process, b"Waiting for the merge lock held by PR 9 in another worktree")
            assert gh_calls(state_file) == []
            fcntl.flock(lock, fcntl.LOCK_UN)
            output, _ = process.communicate(timeout=300)
    assert process.returncode == 0, output.decode()
    assert remote_commit(remote, "main^2") == head
    holder = (root / ".git/repo-merge.lock").read_text()
    assert holder.startswith(f"PR 10 in {git(root, 'rev-parse', '--show-toplevel')} since ")


@pytest.mark.parametrize("fast_forward", [None, "only"])
def test_merge_brings_in_main_and_pushes_it_before_the_suite(
    merge_boundary, tmp_path, fast_forward
):
    root, remote, state_file = merge_boundary
    if fast_forward:
        git(root, "config", "merge.ff", fast_forward)
    feature = git(root, "rev-parse", "HEAD")
    main = advance_main(root, remote, tmp_path / "main", "main.txt", "main\n")
    result = command(root, "make", "merge", "PR=10", timeout=300)
    merged = git(root, "rev-parse", "HEAD")
    assert git(root, "rev-parse", "HEAD^1", "HEAD^2").split() == [feature, main]
    assert remote_commit(remote, "feature") == merged
    assert result.stdout.index(f"Pushed {merged} to feature") < result.stdout.index("Suite full")
    _, merge = landing_calls(state_file)
    assert merge[merge.index("--match-head-commit") + 1] == merged
    assert remote_commit(remote, "main^2") == merged


def test_conflicting_main_aborts_the_merge_and_leaves_the_worktree_clean(merge_boundary, tmp_path):
    root, remote, state_file = merge_boundary
    head = git(root, "rev-parse", "HEAD")
    advance_main(root, remote, tmp_path / "main", "note.txt", "main\n")
    result = command(root, "make", "merge", "PR=10", success=False)
    assert "Merge origin/main into feature by hand" in result.stderr
    assert git(root, "rev-parse", "HEAD") == head
    assert git(root, "status", "--porcelain") == ""
    assert remote_commit(remote, "feature") == head
    assert "Suite" not in result.stdout
    assert landing_calls(state_file) == []


@pytest.mark.parametrize("moved", ["main", "feature"])
def test_merge_reruns_when_main_moves_and_stops_when_the_pr_moves(
    merge_boundary, tmp_path, monkeypatch, moved
):
    root, remote, state_file = merge_boundary
    monkeypatch.setenv("GATE_TEST_MOVED", str(tmp_path / "moved"))
    monkeypatch.setenv("GATE_TEST_REMOTE", str(remote))
    monkeypatch.setenv("GATE_TEST_BRANCH", moved)
    write(
        root,
        "tests/test_fixture.py",
        textwrap.dedent("""\
            import os
            import subprocess
            from pathlib import Path


            def test_value():
                marker = Path(os.environ["GATE_TEST_MOVED"])
                if marker.exists():
                    return
                marker.touch()
                branch = os.environ["GATE_TEST_BRANCH"]
                git = ["git", "--git-dir", os.environ["GATE_TEST_REMOTE"]]
                identity = ["-c", "user.name=Gate Test", "-c", "user.email=gate@example.invalid"]
                tree = f"{branch}^{{tree}}"
                moved = subprocess.run(
                    [*git, *identity, "commit-tree", tree, "-p", branch, "-m", "Move"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                subprocess.run([*git, "update-ref", f"refs/heads/{branch}", moved], check=True)
            """),
    )
    git(root, "commit", "--no-verify", "-am", "Move a branch during the run")
    git(root, "push", str(remote), "feature")
    head = git(root, "rev-parse", "HEAD")
    result = command(root, "make", "merge", "PR=10", success=moved == "main", timeout=300)
    if moved == "main":
        merged = git(root, "rev-parse", "HEAD")
        assert "main moved during the run; merging it again" in result.stdout
        assert [
            line.split()[3] for line in result.stdout.splitlines() if "Passed full" in line
        ] == [
            head,
            merged,
        ]
        assert git(root, "rev-parse", "HEAD^1") == head
        _, merge = landing_calls(state_file)
        assert merge[merge.index("--match-head-commit") + 1] == merged
        assert remote_commit(remote, "main^2") == merged
    else:
        assert f"PR 10 moved on GitHub after {head} passed" in result.stderr
        assert landing_calls(state_file) == []


def test_merge_stops_when_the_pr_is_retargeted_during_the_run(merge_boundary, monkeypatch):
    root, remote, state_file = merge_boundary
    git(root, "push", str(remote), "main:release")
    monkeypatch.setenv("GATE_TEST_STATE", str(state_file))
    write(
        root,
        "tests/test_fixture.py",
        textwrap.dedent("""\
            import json
            import os
            from pathlib import Path


            def test_value():
                state_file = Path(os.environ["GATE_TEST_STATE"])
                state = json.loads(state_file.read_text())
                state["pull"]["baseRefName"] = "release"
                state_file.write_text(json.dumps(state))
            """),
    )
    git(root, "commit", "--no-verify", "-am", "Retarget the PR during the run")
    git(root, "push", str(remote), "feature")
    head = git(root, "rev-parse", "HEAD")
    result = command(root, "make", "merge", "PR=10", success=False, timeout=300)
    assert f"PR 10 was retargeted from main to release after {head} passed" in result.stderr
    assert landing_calls(state_file) == []


@pytest.mark.parametrize(
    "invalid", ["number", "closed", "draft", "fork", "branch", "dirty", "unpushed"]
)
def test_merge_refuses_a_pull_request_it_cannot_land(merge_boundary, invalid):
    root, remote, state_file = merge_boundary
    pull = json.loads(state_file.read_text())["pull"]
    number = "PR=10"
    match invalid:
        case "number":
            number = "PR=10; touch injected"
        case "closed":
            set_gh_state(state_file, pull=pull | {"state": "CLOSED"})
        case "draft":
            set_gh_state(state_file, pull=pull | {"isDraft": True})
        case "fork":
            set_gh_state(state_file, pull=pull | {"headRepository": {"nameWithOwner": "fork/x"}})
        case "branch":
            git(root, "switch", "main")
        case "dirty":
            write(root, "note.txt", "uncommitted\n")
        case "unpushed":
            git(root, "commit", "--no-verify", "--allow-empty", "-m", "Keep local")
    expected = {
        "number": "make merge requires PR=<pull request number>",
        "closed": "PR 10 is closed, not open",
        "draft": "PR 10 is a draft",
        "fork": "PR 10 comes from fork/x",
        "branch": "Check out feature before merging PR 10",
        "dirty": "Commit or discard changes to tracked files",
        "unpushed": "push or pull first",
    }[invalid]
    result = command(root, "make", "merge", number, success=False)
    assert expected in result.stderr
    assert "Suite" not in result.stdout
    assert landing_calls(state_file) == []
    assert not (root / "injected").exists()


@pytest.mark.smoke
def test_make_metadata_never_becomes_shell_code(repository, tmp_path):
    marker = tmp_path / "injected"
    command(
        repository,
        "make",
        "verify",
        f"TARGET=HEAD; touch {shlex.quote(str(marker))}",
        success=False,
    )
    assert not marker.exists()
    command(
        repository,
        "make",
        "verify",
        f"TARGET=$(shell touch {shlex.quote(str(marker))})",
        success=False,
    )
    assert not marker.exists()


def test_inherited_pytest_options_cannot_hide_a_failing_test(repository, monkeypatch):
    write(repository, "tests/test_fixture.py", FAILING)
    git(repository, "add", "tests/test_fixture.py")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PYTEST_PLUGINS", "not_a_real_plugin")
    with pytest.raises(gates.GateError, match="failed"):
        gates.verify(repository, gates.capture_target(repository, "index"), "full")


def test_full_suite_shards_run_each_collected_test_once(repository, tmp_path, monkeypatch):
    root = repository
    imports, runs = tmp_path / "imports", tmp_path / "runs"
    imports.mkdir()
    monkeypatch.setenv("GATE_TEST_IMPORTS", str(imports))
    monkeypatch.setenv("GATE_TEST_RUNS", str(runs))
    project = root / "pyproject.toml"
    project.write_text(
        project.read_text().replace('addopts = "', "addopts = \"-k 'not deselected' ")
    )
    header = textwrap.dedent("""\
        import os
        import uuid
        from pathlib import Path

        import pytest

        IMPORTS = Path(os.environ["GATE_TEST_IMPORTS"])
        IMPORTS.joinpath(uuid.uuid4().hex).touch()


        """)
    write(
        root,
        "tests/test_fixture.py",
        header
        + textwrap.dedent("""\
            @pytest.mark.parametrize("case", [*range(9), uuid.uuid4().hex])
            def test_value(case):
                with Path(os.environ["GATE_TEST_RUNS"]).open("a") as log:
                    log.write(f"{case}\\n")
                assert isinstance(case, int)


            def test_deselected():
                raise AssertionError
            """),
    )
    git(root, "add", "pyproject.toml", "tests")
    command(root, "make", "deps", "TARGET=index")
    result = command(root, "make", "verify", "TARGET=index", success=False)
    assert "pytest shards failed with exits [0, 1, 0, 0]" in result.stdout
    assert "shard 1: FAILED tests/test_fixture.py::test_value[" in result.stdout
    logged = runs.read_text().splitlines()
    assert sorted(line for line in logged if len(line) == 1) == [str(case) for case in range(9)]
    assert len(logged) == 10
    assert len(list(imports.iterdir())) == gates.FULL_SUITE_SHARDS + 1
    for path in imports.iterdir():
        path.unlink()
    write(
        root,
        "tests/test_fixture.py",
        header
        + textwrap.dedent("""\
            @pytest.mark.parametrize("case", range(len(list(IMPORTS.iterdir()))))
            def test_value(case):
                assert case >= 0
            """),
    )
    git(root, "add", "tests")
    result = command(root, "make", "verify", "TARGET=index", success=False)
    assert "Shard 0 collected 2 tests; the suite collected 1" in result.stdout + result.stderr


def test_lint_failure_still_runs_pytest_and_ends_with_every_failure(repository):
    root = repository
    write(root, "tests/test_fixture.py", "import os\n" + FAILING)
    git(root, "add", "tests/test_fixture.py")
    result = command(root, "make", "verify", "TARGET=index", success=False)
    assert "F401" in result.stdout
    assert "assert 1 == 2" in result.stdout
    summary = result.stdout.partition("Suite full failed 2 of 3 steps:\n")[2].splitlines()
    assert summary == [
        "ruff check: ruff check failed with exit 1",
        "pytest: shard 0: FAILED tests/test_fixture.py::test_value",
        "pytest shards failed with exits [1]",
    ]
    assert "Suite full failed: ruff check, pytest" in result.stderr


def test_suites_skip_gate_tests_whose_inputs_passed(repository, tmp_path, monkeypatch):
    root = repository
    runs = tmp_path / "runs"
    monkeypatch.setenv("GATE_TEST_RUNS", str(runs))
    write(
        root,
        gates.GATE_TESTS,
        textwrap.dedent("""\
            import os
            from pathlib import Path

            import pytest


            @pytest.mark.smoke
            def test_gate():
                with Path(os.environ["GATE_TEST_RUNS"]).open("a") as log:
                    log.write("ran\\n")
            """),
    )

    def check(target, expected_runs, success=True):
        git(root, "add", ".")
        result = command(root, "make", target, "TARGET=index", success=success)
        assert len(runs.read_text().splitlines()) == expected_runs
        return result.stdout

    assert "Skipped" not in check("smoke-index", 1)
    assert "Skipped" not in check("verify", 2)
    for target, suite in (("verify", "full"), ("smoke-index", "smoke")):
        skipped = check(target, 2).splitlines()
        assert skipped[-2].startswith(f"Skipped {gates.GATE_TESTS}: a full suite already passed")
        assert skipped[-1].startswith(f"Passed {suite} for ")
    write(root, "src/application.py", "VALUE = 1\n")
    assert "Skipped" in check("verify", 2)
    write(root, "examples/notes.txt", "a gate input\n")
    write(root, "tests/test_fixture.py", FAILING)
    check("verify", 3, success=False)
    write(root, "tests/test_fixture.py", PASSING)
    assert "Skipped" not in check("verify", 4)
    change_dependency(root, "2.0")
    command(root, "make", "deps", "TARGET=index")
    assert "Skipped" not in check("verify", 5)


@pytest.mark.parametrize("threshold", [0, 100])
def test_coverage_uses_captured_configuration(repository, tmp_path, monkeypatch, threshold):
    root = repository
    project = root / "pyproject.toml"
    project.write_text(
        project.read_text().replace("dev = [", 'dev = ["coverage", ')
        + '\n[tool.coverage.run]\nsource = ["tools"]\n'
        + f"[tool.coverage.report]\nfail_under = {threshold}\n"
    )
    change_dependency(root, "1.0")
    write(
        root,
        "tests/test_fixture.py",
        PASSING.replace(
            "    assert True",
            "    import gate_dep\n\n"
            "    from tools.repo import _github_repository\n\n"
            '    assert gate_dep.VERSION == "1.0"\n'
            '    assert _github_repository("https://github.com/example/research") '
            '== "example/research"',
        ),
    )
    git(root, "add", "tests")
    command(root, "make", "deps", "TARGET=index")
    external = write(
        tmp_path,
        "external.coveragerc",
        f"[run]\nsource = tools,gate_dep\n[report]\nfail_under = {100 - threshold}\n",
    )
    external_data = tmp_path / "external.coverage"
    monkeypatch.setenv("COVERAGE_RCFILE", str(external))
    monkeypatch.setenv("COVERAGE_FILE", str(external_data))
    result = command(root, "make", "coverage", "TARGET=index", success=threshold == 0)
    assert "tools/repo.py" in result.stdout
    assert "gate_dep.py" not in result.stdout
    assert not external_data.exists()
    if threshold:
        assert "fail-under=100" in result.stdout + result.stderr


def test_verification_requires_preparation_and_preserves_completed_environment(repository):
    root = repository
    store = root / ".git/prepared"
    for pointer in store.glob("*.json"):
        pointer.unlink()
    with pytest.raises(gates.GateError, match="make deps TARGET=HEAD"):
        gates.verify(root, gates.capture_target(root, "HEAD"), "full")
    prepared = gates.prepare(root, gates.capture_target(root, "HEAD"))
    before = tree_bytes(prepared.python.parent.parent)
    command(root, "make", "verify", "TARGET=HEAD")
    assert tree_bytes(prepared.python.parent.parent) == before


@pytest.mark.parametrize("failure", ["remote-head", "pr-create", "pr-view"])
def test_late_pr_failure_reports_incomplete_operation(pr_boundary, failure):
    root, remote, state_file, request = pr_boundary
    set_gh_state(state_file, fail=failure)
    with pytest.raises(gates.GateError, match="Injected") as error:
        gates.create_pr(root, request)
    calls = gh_calls(state_file)
    if failure == "remote-head":
        assert not any(call["args"][:2] == ["pr", "create"] for call in calls)
    if failure == "pr-view":
        assert "https://github.com/example/research/pull/10" in str(error.value)
    assert git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == git(
        root, "rev-parse", "feature"
    )


def test_direct_local_dependencies_are_rejected(repository):
    root = repository
    project = root / "pyproject.toml"
    project.write_text(
        project.read_text()
        + '\n[tool.uv.sources]\ngate-dep = { path = "gate_dep-1.0-py3-none-any.whl" }\n'
    )
    wheel_dependency(root, "1.0")
    command(root, "uv", "lock", "--offline", "--python", "3.12.13")
    git(root, "add", "pyproject.toml", "uv.lock", "gate_dep-1.0-py3-none-any.whl")
    lock = tomllib.loads((root / "uv.lock").read_text())
    assert next(
        package["source"] for package in lock["package"] if package["name"] == "gate-dep"
    ) == {"path": "gate_dep-1.0-py3-none-any.whl"}
    with pytest.raises(gates.GateError, match="Unsupported dependency source"):
        gates.prepare(root, gates.capture_target(root, "index"))


@pytest.mark.parametrize(
    "source",
    [
        '{ archive = "file:///tmp/gate_dep-1.0-py3-none-any.whl" }',
        '{ archive = "https://example.invalid/gate_dep-1.0-py3-none-any.whl" }',
        '{ url = "file:///tmp/gate_dep-1.0-py3-none-any.whl" }',
        '{ url = "http://127.0.0.1:50448/gate_dep-1.0-py3-none-any.whl" }',
        '{ git = "file:///tmp/dependency#commit" }',
        '{ directory = "dependency" }',
        '{ editable = "." }',
        '{ path = "dependency.whl" }',
        "{ workspace = true }",
        '{ virtual = "." }',
        '{ virtual = "dependency" }',
        '{ registry = "" }',
        '{ registry = "  " }',
        "{ registry = 1 }",
        '{ registry = "https://example.invalid/simple", archive = "dependency.whl" }',
        '{ virtual = ".", registry = "https://example.invalid/simple" }',
        '{ unknown = "dependency" }',
        "{}",
        "[]",
        '"registry"',
    ],
)
def test_dependency_source_policy_rejects_unsupported_lock_shapes(tmp_path, source):
    write(tmp_path, "pyproject.toml", '[project]\nname = "root-project"\n')
    write(tmp_path, ".python-version", "3.12.13\n")
    write(tmp_path, "uv.lock", f'[[package]]\nname = "gate-dep"\nsource = {source}\n')
    with pytest.raises(gates.GateError, match="Unsupported dependency source.*gate-dep"):
        gates._configuration(tmp_path)


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("root-project", '{ virtual = "." }'),
        ("gate-dep", '{ registry = "https://example.invalid/simple" }'),
        ("gate-dep", '{ registry = "file:///tmp/local-index" }'),
    ],
)
def test_dependency_source_policy_allows_root_and_registry_sources(tmp_path, name, source):
    project = write(tmp_path, "pyproject.toml", '[project]\nname = "Root_Project"\n')
    pin = write(tmp_path, ".python-version", "3.12.13\n")
    lock = write(tmp_path, "uv.lock", f'[[package]]\nname = "{name}"\nsource = {source}\n')
    assert gates._configuration(tmp_path) == tuple(
        path.read_bytes() for path in (project, lock, pin)
    )


def test_actual_git_dependency_lock_is_rejected_offline(repository, tmp_path):
    dependency = tmp_path / "git-dependency"
    dependency.mkdir()
    git(dependency, "init", "-b", "main")
    git(dependency, "config", "user.name", "Gate Test")
    git(dependency, "config", "user.email", "gate@example.invalid")
    write(dependency, "pyproject.toml", '[project]\nname = "gate-dep"\nversion = "1.0"\n')
    write(dependency, "gate_dep.py", "VERSION = '1.0'\n")
    git(dependency, "add", ".")
    git(dependency, "commit", "-m", "Prepare local dependency")
    project = repository / "pyproject.toml"
    project.write_text(
        project.read_text()
        + f'\n[tool.uv.sources]\ngate-dep = {{ git = "{dependency.as_uri()}" }}\n'
    )
    command(repository, "uv", "lock", "--offline", "--python", "3.12.13")
    lock = tomllib.loads((repository / "uv.lock").read_text())
    source = next(package["source"] for package in lock["package"] if package["name"] == "gate-dep")
    assert source == {"git": f"{dependency.as_uri()}#{git(dependency, 'rev-parse', 'HEAD')}"}
    git(repository, "add", "pyproject.toml", "uv.lock")
    with pytest.raises(gates.GateError, match="Unsupported dependency source.*gate-dep"):
        gates.prepare(repository, gates.capture_target(repository, "index"))


def test_missing_prepared_executable_cannot_use_path_fallback(repository, tmp_path, monkeypatch):
    prepared = gates.prepare(repository, gates.capture_target(repository, "HEAD"))
    (prepared.python.parent / "ruff").unlink()
    marker = tmp_path / "fallback-ran"
    fallback = write(tmp_path, "bin/ruff", f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nexit 0\n")
    fallback.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fallback.parent}{os.pathsep}{os.environ['PATH']}")
    before = tree_bytes(prepared.python.parent.parent)
    with pytest.raises(gates.GateError, match="make deps TARGET=HEAD"):
        gates.verify(repository, gates.capture_target(repository, "HEAD"), "full")
    assert not marker.exists()
    assert tree_bytes(prepared.python.parent.parent) == before


@pytest.fixture
def packaged_repository(repository):
    root = repository
    shutil.copytree(ROOT / "src", root / "src")
    shutil.copytree(ROOT / "examples/development-package", root / "examples/development-package")
    write(
        root,
        "tests/fixtures/wheel_consumer.py",
        (ROOT / "tests/fixtures/wheel_consumer.py").read_text(),
    )
    for name in ("save_development.py", "__init__.py"):
        shutil.copyfile(ROOT / "examples" / name, root / "examples" / name)
    project = root / "pyproject.toml"
    project.write_text(
        project.read_text()
        .replace('name = "gate-fixture"', 'name = "quant-research"')
        .replace("dev = [", 'dev = ["hatchling", ')
        + '\n[project.scripts]\nquant-research = "quant_research.__main__:main"\n'
        + '\n[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        + '\n[tool.hatch.build.targets.wheel]\npackages = ["src/quant_research"]\n'
    )
    change_dependency(root, "1.0")
    git(root, "add", "pyproject.toml", "uv.lock", "src", "examples", "tests/fixtures")
    command(root, "make", "deps", "TARGET=index")
    return root


def test_full_gate_builds_and_exercises_wheel_without_source_or_install(packaged_repository):
    root = packaged_repository
    prepared = gates._execute_snapshot(
        root, gates.capture_target(root, "index"), "environment"
    ).environment
    before = tree_bytes(prepared.python.parent.parent)
    result = command(root, "make", "verify", "TARGET=index")
    assert "Wheel import and portable save/open consumer passed" in result.stdout
    assert "Verified " in result.stdout
    assert '"disclosure_coverage": "unknown"' in result.stdout
    assert not list(prepared.purelib.glob("quant_research*"))
    assert tree_bytes(prepared.python.parent.parent) == before


@pytest.mark.parametrize("missing", ["packages.py", "__main__.py"])
def test_full_gate_rejects_incomplete_wheel_after_passing_source_tests(
    packaged_repository, missing
):
    root = packaged_repository
    project = root / "pyproject.toml"
    project.write_text(project.read_text() + f'exclude = ["src/quant_research/{missing}"]\n')
    git(root, "add", "pyproject.toml")
    command(root, "make", "deps", "TARGET=index")
    result = command(root, "make", "verify", "TARGET=index", success=False)
    assert "1 passed" in result.stdout
    assert (
        "No module named quant_research" in result.stderr
        or "No module named 'quant_research" in result.stderr
    )
    assert "Passed full" not in result.stdout


def test_make_run_uses_prepared_dependencies_checkout_source_and_literal_args(packaged_repository):
    root = packaged_repository
    name = "literal $(shell touch injected) `touch injected-too`"
    shutil.copytree(root / "examples/development-package", root / name)
    result = command(root, "make", "run", f"ARGS=inspect {shlex.quote(name)}")
    assert '"disclosure_coverage": "unknown"' in result.stdout
    assert not (root / "injected").exists()
    assert not (root / "injected-too").exists()
    command(
        root,
        "make",
        "run",
        "ARGS=verify examples/development-package; touch injected",
        success=False,
    )
    assert not (root / "injected").exists()
    command(root, "make", "run", "ARGS='unfinished", success=False)
    source = root / "src/quant_research/__main__.py"
    source.write_text('print("current checkout source")\n')
    result = command(root, "make", "run", "ARGS=inspect examples/development-package")
    assert "current checkout source" in result.stdout


def test_make_example_runs_python_save_consumer_with_literal_destination(packaged_repository):
    root = packaged_repository
    name = "new packages $(shell touch injected) `touch injected-too`"
    prepared = gates._execute_snapshot(
        root, gates.capture_target(root, "index"), "environment"
    ).environment
    assert (prepared.purelib / "examples/__init__.py").is_file()
    before = tree_bytes(prepared.python.parent.parent)
    result = command(root, "make", "example", f"DESTINATION={name}")
    saved = Path(result.stdout.splitlines()[-1])
    assert saved.is_relative_to(root / name)
    verified = command(root, "make", "run", f"ARGS=verify {shlex.quote(str(saved))}")
    assert "Verified " in verified.stdout
    assert not (root / "injected").exists()
    assert not (root / "injected-too").exists()
    assert tree_bytes(prepared.python.parent.parent) == before


def test_make_run_selects_index_dependencies_and_working_source(packaged_repository):
    root = packaged_repository
    change_dependency(root, "2.0")
    command(root, "make", "deps", "TARGET=index")
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        write(root, name, "unstaged invalid metadata\n")
    write(
        root,
        "src/quant_research/__main__.py",
        'import gate_dep\nprint("working source dependency=" + gate_dep.VERSION)\n',
    )
    result = command(root, "make", "run", "ARGS=ignored")
    assert "working source dependency=2.0" in result.stdout


@pytest.mark.parametrize("damage", ["policy", "stale-lock", "unprepared"])
def test_make_commands_reject_invalid_or_unprepared_index_with_good_working_metadata(
    packaged_repository, damage
):
    root = packaged_repository
    original = {name: (root / name).read_bytes() for name in ("pyproject.toml", "uv.lock")}
    if damage == "unprepared":
        change_dependency(root, "2.0")
    else:
        project = root / "pyproject.toml"
        project.write_text(
            project.read_text() + "\n[tool.uv.workspace]\nmembers = []\n"
            if damage == "policy"
            else project.read_text().replace("gate-dep==1.0", "gate-dep==2.0")
        )
        git(root, "add", "pyproject.toml")
    for name, content in original.items():
        (root / name).write_bytes(content)
    for invocation in (("run", "ARGS=verify examples/development-package"), ("example",)):
        result = command(root, "make", *invocation, success=False)
        output = result.stdout + result.stderr
        assert "make deps TARGET=index" in output
        if damage == "policy":
            assert "Workspace dependencies" in output
        if damage != "unprepared":
            assert "run uv lock, and stage" in output
        assert not (root / "local-runs").exists()


def test_make_run_reports_malformed_args_before_environment_work(packaged_repository):
    root = packaged_repository
    write(root, ".python-version", "invalid staged Python pin\n")
    git(root, "add", ".python-version")
    result = command(root, "make", "run", "ARGS='unfinished", success=False)
    assert "Invalid ARGS" in result.stderr
    assert "exact Python patch" not in result.stderr
    assert "make deps" not in result.stderr


@pytest.mark.parametrize("symlink", [False, True])
def test_make_example_keeps_environment_and_lock_from_one_capture(
    packaged_repository, tmp_path, symlink
):
    root = packaged_repository
    lock = root / "uv.lock"
    captured_lock = lock.read_bytes()
    if symlink:
        lock.unlink()
        lock.symlink_to("locks/authoring.lock")
        write(root, "locks/authoring.lock", captured_lock.decode())
        git(root, "add", "uv.lock", "locks")
    lock_entry = "locks/authoring.lock" if symlink else "uv.lock"
    later_lock = write(tmp_path, "later.lock", captured_lock.decode() + "\n# later index\n")
    later_oid = git(root, "hash-object", "-w", str(later_lock))
    lock.write_bytes(b"unrelated unstaged lock\n")
    author = root / "examples/save_development.py"
    author.write_text(author.read_text() + "\n# Working authoring revision.\n")
    marker = tmp_path / "index-moved"
    executable = shutil.which("uv")
    forwarder = write(
        tmp_path,
        "uv_proxy.py",
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"marker = Path({str(marker)!r})\n"
        "if sys.argv[1:] == ['--version'] and not marker.exists():\n"
        f"    subprocess.run(['git', '-C', {str(root)!r}, 'update-index', '--cacheinfo', "
        f"'100644,{later_oid},{lock_entry}'], check=True)\n"
        "    marker.touch()\n"
        f"os.execv({executable!r}, [{executable!r}, *sys.argv[1:]])\n",
    )
    proxy = write(
        tmp_path / "bin",
        "uv",
        f'#!/bin/sh\nexec {shlex.quote(str(PYTHON))} -B -I -S {shlex.quote(str(forwarder))} "$@"\n',
    )
    proxy.chmod(0o755)
    result = command(
        root,
        "make",
        "example",
        f"DESTINATION={tmp_path / 'authored'}",
        env=os.environ | {"PATH": f"{proxy.parent}:{os.environ['PATH']}"},
    )
    saved = open_package(Path(result.stdout.splitlines()[-1]))
    assert marker.exists()
    assert git(root, "show", f":{lock_entry}").encode() != captured_lock.rstrip()
    assert saved.read_artifact("uv_lock") == captured_lock
    assert saved.read_artifact("implementation") == author.read_bytes()
    assert saved.record.selected_trial.specification.environment.python_version == "3.12.13"


def test_sealed_generation_rejects_coherent_pytest_replacement(repository, tmp_path, capfd):
    root = repository
    prepared = gates.prepare(root, gates.capture_target(root, "index"))
    gates.verify(root, gates.capture_target(root, "index"), "smoke")
    assert "1 passed" in capfd.readouterr().out
    write(root, "tests/test_fixture.py", FAILING)
    git(root, "add", "tests")
    with pytest.raises(gates.GateError):
        gates.verify(root, gates.capture_target(root, "index"), "smoke")
    assert "assert 1 == 2" in capfd.readouterr().out
    marker = tmp_path / "replacement-ran"
    module = prepared.purelib / "pytest/__main__.py"
    original = module.read_bytes()
    sibling = tmp_path / "shared-entrypoint"
    sibling.hardlink_to(module)
    replace_installed_file(
        module,
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise SystemExit(0)\n".encode(),
    )
    record = next(prepared.purelib.glob("pytest-*.dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(record.read_text())))
    row = next(row for row in rows if row[0] == "pytest/__main__.py")
    row[1] = (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(module.read_bytes()).digest())
        .rstrip(b"=")
        .decode()
    )
    row[2] = str(module.stat().st_size)
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    replace_installed_file(record, output.getvalue().encode())
    before = tree_bytes(prepared.python.parent.parent)
    with pytest.raises(gates.GateError, match="fingerprint mismatch") as error:
        gates.verify(root, gates.capture_target(root, "index"), "smoke")
    assert "make deps TARGET=index" in str(error.value)
    assert not marker.exists()
    assert sibling.read_bytes() == original
    assert tree_bytes(prepared.python.parent.parent) == before


@pytest.mark.parametrize("relation", ["equal", "ancestor"])
def test_pr_rejects_head_without_contributing_commits(pr_boundary, relation):
    root, remote, state_file, request = pr_boundary
    if relation == "equal":
        git(root, "update-ref", "refs/heads/feature", git(root, "rev-parse", "main"))
    else:
        command(root, "git", "--git-dir", remote, "fetch", str(root), "feature:main")
        git(root, "update-ref", "refs/heads/feature", git(root, "rev-parse", "main"))
    before = git(root, "ls-remote", str(remote))
    with pytest.raises(gates.GateError, match="no commits"):
        gates.create_pr(root, request)
    assert git(root, "ls-remote", str(remote)) == before
    assert not any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))


def test_pr_same_head_into_other_base_is_allowed(pr_boundary):
    root, remote, state_file, request = pr_boundary
    set_gh_state(state_file, existing=True, existing_base="release")
    receipt = gates.create_pr(root, request)
    assert receipt.commit_oid == git(root, "rev-parse", "feature")
    assert (
        git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == receipt.commit_oid
    )
    assert json.loads(state_file.read_text())["assigned"] is True


def test_pr_assignment_failure_retains_verified_identity_and_blocks_recreation(pr_boundary):
    root, remote, state_file, request = pr_boundary
    set_gh_state(state_file, fail="pr-edit")
    with pytest.raises(gates.GateError, match="pull/10.*assignment incomplete") as error:
        gates.create_pr(root, request)
    head = git(root, "rev-parse", "feature")
    assert head in str(error.value)
    assert json.loads(state_file.read_text())["created"]["head"]["sha"] == head
    assert git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == head
    calls = [call["args"][:2] for call in gh_calls(state_file)]
    assert calls.index(["pr", "view"]) < calls.index(["pr", "edit"])
    with pytest.raises(gates.GateError, match="open PR already exists.*pull/10"):
        gates.create_pr(root, request)
    assert json.loads(state_file.read_text())["create_attempts"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "added-file",
        "added-directory",
        "removed-file",
        "file-mode",
        "directory-mode",
        "file-to-directory",
        "internal-link",
        "external-link",
        "runtime-link",
        "runtime-executable",
        "fifo",
    ],
)
def test_seal_covers_membership_types_modes_and_links(repository, tmp_path, change):
    root = repository
    prepared = gates.prepare(root, gates.capture_target(root, "index"))
    generation = prepared.python.parent.parent
    path = prepared.purelib / "gate_dep.py"
    sibling = tmp_path / "hardlinked-module"
    sibling.hardlink_to(path)
    original, original_mode = sibling.read_bytes(), sibling.stat().st_mode
    marker = tmp_path / "prepared-interpreter-ran"
    match change:
        case "added-file":
            write(prepared.purelib, "unrecorded.py", "raise SystemExit(0)\n")
        case "added-directory":
            (generation / "empty-directory").mkdir()
        case "removed-file":
            path.unlink()
        case "file-mode":
            replace_installed_file(path, path.read_bytes())
            path.chmod(0o700)
        case "directory-mode":
            prepared.purelib.chmod(0o700)
        case "file-to-directory":
            path.unlink()
            path.mkdir()
        case "internal-link":
            path.unlink()
            path.symlink_to("_virtualenv.py")
        case "external-link":
            (generation / "outside").symlink_to(tmp_path)
        case "runtime-link":
            alias = prepared.python.parent / "python3"
            alias.unlink()
            alias.symlink_to(prepared.python)
        case "runtime-executable":
            executable = write(
                tmp_path,
                "replacement-python",
                f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nexit 1\n",
            )
            executable.chmod(0o755)
            prepared.python.unlink()
            prepared.python.symlink_to(executable)
        case "fifo":
            os.mkfifo(generation / "pipe")
    pointer = root / ".git/prepared" / f"{prepared.identity}.json"
    before = pointer.read_bytes()
    result = command(root, "make", "smoke-index", success=False, timeout=30)
    assert "fingerprint" in result.stderr
    assert "make deps TARGET=index" in result.stderr
    assert "Suite smoke" not in result.stdout
    assert not marker.exists()
    assert pointer.read_bytes() == before
    assert sibling.read_bytes() == original
    assert sibling.stat().st_mode == original_mode


@pytest.mark.parametrize("invalid", ["legacy", "format", "identity", "digest", "relative", "json"])
def test_unsealed_receipt_requires_fresh_preparation(repository, invalid):
    root = repository
    prepared = gates.prepare(root, gates.capture_target(root, "index"))
    pointer = root / ".git/prepared" / f"{prepared.identity}.json"
    value = json.loads(pointer.read_text())
    match invalid:
        case "legacy":
            value = {"generation": value["generation"]}
        case "format":
            value["format"] = "future"
        case "identity":
            value["identity"] = "0" * 64
        case "digest":
            value["sha256"] = "invalid"
        case "relative":
            value["generation"] = "relative/generation"
    pointer.write_text("not-json" if invalid == "json" else json.dumps(value))
    old_pointer = pointer.read_bytes()
    old_generation = tree_bytes(prepared.python.parent.parent)
    result = command(root, "make", "smoke-index", success=False)
    assert "receipt" in result.stderr
    assert "make deps TARGET=index" in result.stderr
    assert "Suite smoke" not in result.stdout
    assert pointer.read_bytes() == old_pointer
    refreshed = gates.prepare(root, gates.capture_target(root, "index"))
    assert refreshed.python != prepared.python
    gates.verify(root, gates.capture_target(root, "index"), "smoke")
    assert tree_bytes(prepared.python.parent.parent) == old_generation


@pytest.mark.parametrize("fault", ["metadata", "editable", "application", "external-link", "fifo"])
def test_failed_initial_admission_preserves_published_generation(
    repository, monkeypatch, tmp_path, fault
):
    root = repository
    snapshot = gates.capture_target(root, "index")
    store = root / ".git/prepared"
    before = tree_bytes(store)
    sync = gates._sync

    def install_invalid(root, prepared, env, *, check):
        sync(root, prepared, env, check=check)
        if not check:
            metadata = prepared.purelib / "gate_dep-1.0.dist-info"
            match fault:
                case "metadata":
                    replace_installed_file(metadata / "METADATA", b"Name: gate-dep\nVersion: 9.0\n")
                case "editable":
                    write(metadata, "direct_url.json", json.dumps({"dir_info": {"editable": True}}))
                case "application":
                    metadata.rename(prepared.purelib / "quant_research-1.0.dist-info")
                    metadata = prepared.purelib / "quant_research-1.0.dist-info"
                    replace_installed_file(
                        metadata / "METADATA", b"Name: quant-research\nVersion: 1.0\n"
                    )
                    replace_installed_file(
                        metadata / "RECORD", b"quant_research-1.0.dist-info/METADATA,,\n"
                    )
                case "external-link":
                    (prepared.purelib / "outside").symlink_to(tmp_path)
                case "fifo":
                    os.mkfifo(prepared.purelib / "pipe")

    monkeypatch.setattr(gates, "_sync", install_invalid)
    expected = {
        "metadata": "Distribution metadata",
        "editable": "Editable distribution",
        "application": "application distribution",
        "external-link": "external prepared symlink",
        "fifo": "file type",
    }
    with tempfile.TemporaryDirectory(prefix="repo-admission-") as directory:
        exported = Path(directory)
        gates._export(root, snapshot, exported)
        with pytest.raises(gates.GateError, match=expected[fault]):
            gates._snapshot_operation(gates.SnapshotTask(snapshot, exported, store, "prepare"))
    assert {name: (store / name).read_bytes() for name in before} == before
    gates.verify(root, snapshot, "smoke")


def test_generation_links_and_selected_runtime_have_stable_content_identity(tmp_path, monkeypatch):
    generation = tmp_path / "generation"
    (generation / "bin").mkdir(parents=True)
    executable = write(tmp_path, "runtime", "first interpreter bytes")
    executable.chmod(0o755)
    (generation / "bin/python").symlink_to(executable)
    (generation / "bin/python3").symlink_to("python")
    (generation / "internal").symlink_to("bin")
    (generation / "dangling").symlink_to("missing")
    monkeypatch.setattr(sys, "_base_executable", str(executable))
    prepared = gates._prepared(
        "fixture", generation, gates.Runtime("cpython", (3, 12, 13), "abi", "platform")
    )
    before = gates._generation_fingerprint(prepared)
    assert before == gates._generation_fingerprint(prepared)
    outside = tmp_path / "outside-alias"
    outside.symlink_to(generation / "bin")
    for target in (outside, Path("../generation/bin")):
        link = generation / "reentry"
        link.symlink_to(target)
        with pytest.raises(gates.GateError, match="external prepared symlink|outside generation"):
            gates._generation_fingerprint(prepared)
        link.unlink()
    assert before == gates._generation_fingerprint(prepared)
    replace_installed_file(executable, b"other interpreter bytes")
    assert gates._generation_fingerprint(prepared) != before


@pytest.mark.parametrize(
    "history", ["missing-base", "missing-parent", "shallow", "replace-ref", "legacy-graft"]
)
def test_pr_history_rejection_does_not_publish(pr_boundary, history):
    root, remote, state_file, request = pr_boundary
    expected = git(root, "rev-parse", "feature")
    base = git(root, "rev-parse", "main")
    if history == "missing-base":
        set_gh_state(
            state_file, outputs={"base": json.dumps({"name": "main", "commit": {"sha": "1" * 40}})}
        )
    elif history == "missing-parent":
        tree = git(root, "rev-parse", "feature^{tree}")
        raw = (
            f"tree {tree}\nparent {'1' * 40}\n"
            "author Gate Test <gate@example.invalid> 1 +0000\n"
            "committer Gate Test <gate@example.invalid> 1 +0000\n\nMissing history\n"
        )
        missing = (
            gates._run(
                ["git", "hash-object", "-t", "commit", "-w", "--stdin"], root, data=raw.encode()
            )
            .decode()
            .strip()
        )
        git(root, "update-ref", "refs/heads/feature", missing)
    elif history == "shallow":
        write(root, ".git/shallow", expected + "\n")
    elif history == "legacy-graft":
        command(root, "git", "--git-dir", remote, "fetch", str(root), "feature:main")
        git(root, "update-ref", "refs/heads/feature", base)
        write(root, ".git/info/grafts", expected + "\n")
    else:
        git(root, "update-ref", "refs/heads/feature", base)
        git(root, "replace", base, expected)
    before = git(root, "ls-remote", str(remote))
    with pytest.raises(gates.GateError, match="no commits|complete history"):
        gates.create_pr(root, request)
    assert git(root, "ls-remote", str(remote)) == before
    assert not any(call["args"][:2] == ["pr", "create"] for call in gh_calls(state_file))


@pytest.mark.parametrize("status", [0, 1])
@pytest.mark.parametrize("output", ["url", "missing"])
def test_pr_reconciles_created_identity_before_assignment(pr_boundary, status, output):
    root, remote, state_file, request = pr_boundary
    set_gh_state(
        state_file,
        create_status=status,
        create_output="https://github.com/example/research/pull/10" if output == "url" else "",
    )
    if status:
        with pytest.raises(gates.GateError, match="verified head.*creation incomplete") as error:
            gates.create_pr(root, request)
        assert "gh pr create exited 1" in str(error.value)
        assert "Injected create failure" in str(error.value)
    else:
        assert gates.create_pr(root, request).url.endswith("/pull/10")
    state = json.loads(state_file.read_text())
    assert state["create_attempts"] == 1
    assert state.get("assigned", False) == (status == 0)
    calls = [call["args"] for call in gh_calls(state_file)]
    assert sum(args[:2] == ["pr", "create"] for args in calls) == 1
    assert sum(args[:2] == ["pr", "view"] for args in calls) == 1
    lists = [args for args in calls if args[:2] == ["api", "repos/example/research/pulls"]]
    assert len(lists) == (2 if output == "missing" else 1)
    assert all("base=main" in args and "--paginate" in args and "--slurp" in args for args in lists)
    assert git(root, "ls-remote", str(remote), "refs/heads/feature").split()[0] == git(
        root, "rev-parse", "feature"
    )


@pytest.mark.parametrize("recovery", ["zero", "multiple", "malformed", "failed", "wrong-url"])
def test_unknown_pr_creation_is_bounded_and_never_recreated(pr_boundary, recovery):
    root, _, state_file, request = pr_boundary
    set_gh_state(state_file, create_status=1, create_output="no URL available")
    if recovery == "zero":
        set_gh_state(state_file, create_persist=False)
    elif recovery == "failed":
        set_gh_state(state_file, recovery_failure=True)
    elif recovery == "malformed":
        set_gh_state(state_file, recovery_output="not-json")
    else:
        item = {
            "html_url": "https://github.com/example/research/pull/10",
            "head": {"ref": "feature", "repo": {"full_name": "example/research"}},
            "base": {"ref": "main"},
        }
        if recovery == "wrong-url":
            item["html_url"] = "https://github.com/other/research/pull/10"
        set_gh_state(
            state_file,
            recovery_output=json.dumps([[item, item] if recovery == "multiple" else [item]]),
        )
    with pytest.raises(gates.GateError, match="creation outcome unknown") as error:
        gates.create_pr(root, request)
    assert "gh pr create exited 1" in str(error.value)
    assert "no URL available" in str(error.value)
    calls = [call["args"] for call in gh_calls(state_file)]
    assert sum(args[:2] == ["pr", "create"] for args in calls) == 1
    assert sum(args[:2] == ["api", "repos/example/research/pulls"] for args in calls) == 2
    assert not any(args[:2] == ["pr", "edit"] for args in calls)


@pytest.mark.parametrize(
    "identity", ["repository", "head-repository", "head-ref", "base", "head-oid"]
)
def test_created_pr_must_match_full_captured_identity(pr_boundary, identity):
    root, _, state_file, request = pr_boundary
    response = {
        "url": "https://github.com/example/research/pull/10",
        "headRefOid": git(root, "rev-parse", "feature"),
        "headRefName": "feature",
        "headRepository": {"nameWithOwner": "example/research"},
        "baseRefName": "main",
    }
    match identity:
        case "repository":
            response["url"] = "https://github.com/other/research/pull/10"
        case "head-repository":
            response["headRepository"] = {"nameWithOwner": "example/fork"}
        case "head-ref":
            response["headRefName"] = "other"
        case "base":
            response["baseRefName"] = "release"
        case "head-oid":
            response["headRefOid"] = git(root, "rev-parse", "main")
    set_gh_state(state_file, create_status=1, outputs={"pr-view": json.dumps(response)})
    with pytest.raises(gates.GateError, match="pull/10 exists; verification incomplete") as error:
        gates.create_pr(root, request)
    assert "gh pr create exited 1" in str(error.value)
    assert json.loads(state_file.read_text())["create_attempts"] == 1
    assert not any(call["args"][:2] == ["pr", "edit"] for call in gh_calls(state_file))


@pytest.mark.parametrize(
    ("kind", "raw_passes"),
    [("commit", False), ("tree", True), ("subtree", False), ("blob", True), ("index", False)],
)
def test_physical_snapshots_ignore_replacements(repository, tmp_path, kind, raw_passes):
    root = repository
    good = git(root, "rev-parse", "HEAD")
    bad = failing_commit(root)
    raw, replacement = (good, bad) if raw_passes else (bad, good)
    tree = git(root, "rev-parse", f"{raw}^{{tree}}")
    expected = git(root, "show", f"{raw}:tests/test_fixture.py") + "\n"
    git(root, "branch", "physical", raw)
    match kind:
        case "commit":
            objects = (raw, replacement)
        case "tree":
            objects = (tree, git(root, "rev-parse", f"{replacement}^{{tree}}"))
        case "subtree":
            objects = tuple(git(root, "rev-parse", f"{oid}:tests") for oid in (raw, replacement))
        case _:
            objects = tuple(
                git(root, "rev-parse", f"{oid}:tests/test_fixture.py") for oid in (raw, replacement)
            )
    if kind == "index":
        git(root, "read-tree", tree)
        write(root, "tests/test_fixture.py", FAILING if raw_passes else PASSING)
    index = index_path(root).read_bytes()
    git(root, "replace", *objects)
    refs = git(root, "for-each-ref", "refs/replace")
    operation = ("smoke-index",) if kind == "index" else ("verify", "TARGET=physical")
    checked = command(root, "make", *operation, success=raw_passes)
    if raw_passes:
        assert "1 passed" in checked.stdout
    else:
        assert "assert 1 == 2" in checked.stdout + checked.stderr
    captured = gates.capture_target(root, "index" if kind == "index" else "physical")
    assert captured.tree_oid == tree
    exported = tmp_path / "raw-export"
    exported.mkdir()
    gates._export(root, captured, exported)
    assert (exported / "tests/test_fixture.py").read_text() == expected
    assert index_path(root).read_bytes() == index
    assert git(root, "for-each-ref", "refs/replace") == refs


def test_failed_seal_publication_preserves_previous_pointer(repository, monkeypatch):
    root = repository
    snapshot = gates.capture_target(root, "index")
    store = root / ".git/prepared"
    prepared = gates.prepare(root, snapshot)
    before = tree_bytes(store)
    replace = Path.replace

    def fail_publication(path, destination):
        if Path(destination).parent == store and Path(destination).suffix == ".json":
            raise OSError("Injected seal publication failure")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_publication)
    with tempfile.TemporaryDirectory(prefix="repo-admission-") as directory:
        exported = Path(directory)
        gates._export(root, snapshot, exported)
        with pytest.raises(OSError, match="seal publication"):
            gates._snapshot_operation(gates.SnapshotTask(snapshot, exported, store, "prepare"))
    assert {name: (store / name).read_bytes() for name in before} == before
    assert gates.verify(root, snapshot, "smoke").environment == prepared
