from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import tomllib
import warnings
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Literal

INDEXER = "code-index/v1"

type Kind = Literal["class", "function"]
type Providers = dict[str, list[tuple[str, str]]]
type Entity = tuple[str, str]
type Binding = tuple[Entity, str] | None

_KINDS: dict[type[ast.AST], Kind] = {
    ast.ClassDef: "class",
    ast.FunctionDef: "function",
    ast.AsyncFunctionDef: "function",
}
_NESTING: dict[Kind, str] = {"class": ".", "function": ".<locals>."}
_TESTS = ("test_*.py", "*_test.py", "conftest.py")


class CodeIndexError(ValueError):
    """Report a repository, commit, or Git object that cannot be read."""


@dataclass(frozen=True)
class Module:
    path: str
    name: str
    blob: str


@dataclass(frozen=True)
class Symbol:
    path: str
    qualname: str
    kind: Kind
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Import:
    path: str
    line: int
    module: str
    target: str | None


@dataclass(frozen=True)
class Reference:
    test_path: str
    scope: str
    line: int
    path: str
    qualname: str


@dataclass(frozen=True)
class Dependency:
    path: str
    group: str
    requirement: str


@dataclass(frozen=True)
class Failure:
    path: str
    error: str


@dataclass(frozen=True)
class CodeIndex:
    """Describe the Python code of one commit in flat, deterministically sorted tables."""

    commit: str
    tree: str
    indexer: str
    parser: str
    modules: tuple[Module, ...]
    symbols: tuple[Symbol, ...]
    imports: tuple[Import, ...]
    references: tuple[Reference, ...]
    dependencies: tuple[Dependency, ...]
    failures: tuple[Failure, ...]

    def to_json(self) -> bytes:
        """Return the canonical UTF-8 JSON encoding of this index."""
        text = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        return f"{text}\n".encode()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json()).hexdigest()


def _executable(repository: Path) -> Path:
    # `os.path` passes over loops and unreadable paths, which `pathlib` raises on before 3.13.
    if not os.path.isdir(repository):
        raise CodeIndexError(f"{repository} is not a directory")
    # Git searches upward for the checkout, and a link can lead `repository` into another one, so
    # skip `repository` and every checkout around it, both as given and as resolved.
    checkouts = [
        parent
        for path in (repository.absolute(), repository.resolve())
        for parent in (path, *path.parents)
        if parent == path or os.path.exists(parent / ".git")
    ]
    for directory in os.get_exec_path():
        # Judge the directory, since a `git` link in the checkout can name any program outside it.
        folder = Path(os.path.realpath(directory))
        git = folder / "git"
        # A link in the checkout can lead an entry outside, so judge each directory it is written
        # through too, which can all be read once the kernel resolves the entry as written.
        written = Path(directory).absolute()
        # Compare directories, not names, since a case-insensitive volume spells one several ways.
        if (
            os.path.isfile(written / "git")
            and os.access(git, os.X_OK)
            and not any(
                parent.samefile(checkout)
                for checkout in checkouts
                for start in (written, folder)
                for parent in (start, *start.parents)
            )
        ):
            return git
    raise CodeIndexError(f"No git executable outside the checkouts around {repository} is on PATH")


def _git(executable: Path, repository: Path, *args: str, data: bytes = b"") -> bytes:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    # `rev-parse` reads the index for a revision such as `:0:app.py`, which runs `core.fsmonitor`.
    command = [executable, "-C", repository, "-c", "core.fsmonitor=false", "--no-replace-objects"]
    try:
        result = subprocess.run(
            [*command, *args],
            env=environment | {"GIT_NO_LAZY_FETCH": "1", "GIT_GRAFT_FILE": os.devnull},
            input=data,
            capture_output=True,
        )
    except OSError as error:
        raise CodeIndexError(f"{executable} cannot start: {error.strerror}") from error
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise CodeIndexError(f"git {args[0]} failed in {repository}: {detail}")
    return result.stdout


def _read_blobs(git: Callable[..., bytes], oids: list[str]) -> dict[str, bytes]:
    output = git("cat-file", "--batch", data="".join(f"{oid}\n" for oid in oids).encode())
    contents = {}
    offset = 0
    for oid in oids:
        end = output.index(b"\n", offset)
        header = output[offset:end].split()
        if header[1:2] != [b"blob"]:
            raise CodeIndexError(f"Git object {oid} is not a readable blob")
        size = int(header[2])
        contents[oid] = output[end + 1 : end + 1 + size]
        offset = end + 2 + size
    return contents


def _locate(path: str, packages: set[str]) -> tuple[str, str]:
    directory, _, filename = path.rpartition("/")
    parts = [] if filename == "__init__.py" else [filename.removesuffix(".py")]
    while directory and directory in packages:
        directory, _, package = directory.rpartition("/")
        parts.insert(0, package)
    return directory, ".".join(parts)


def _declared_global(scope: ast.AST) -> set[str]:
    names, stack = set(), list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Global):
            names.update(node.names)
        elif type(node) not in _KINDS:
            stack.extend(ast.iter_child_nodes(node))
    return names


def _walk(tree: ast.Module) -> Iterator[tuple[str, str, ast.AST]]:
    stack: list[tuple[str, str, set[str], ast.AST]] = [("", "", set(), tree)]
    while stack:
        scope, prefix, declared, node = stack.pop()
        outer = inner = scope, prefix, declared
        if kind := _KINDS.get(type(node)):
            # As in CPython, a definition its enclosing scope declares `global` is top-level.
            if node.name in declared:
                prefix = ""
            qualname = prefix + node.name
            inner = qualname, qualname + _NESTING[kind], _declared_global(node)
        yield scope, prefix, node
        # Only a definition's body runs in its scope; decorators, defaults, and bases do not.
        stack.extend(
            (*(inner if isinstance(child, ast.stmt) else outer), child)
            for child in reversed(list(ast.iter_child_nodes(node)))
        )


def _find(providers: Providers, name: str, root: str) -> str | None:
    candidates = providers.get(name)
    if not candidates:
        return None
    # Within one import root, Python resolves a package directory before a same-named module.
    return min(
        candidates,
        key=lambda item: (item[0] != root, item[0], not item[1].endswith("/__init__.py"), item[1]),
    )[1]


def _deepest(providers: Providers, name: str, root: str) -> tuple[str, str | None]:
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        prefix = ".".join(parts[:end])
        if target := _find(providers, prefix, root):
            return prefix, target
    return name, None


def _absolute(node: ast.ImportFrom, path: str, located: dict[str, tuple[str, str]]) -> str | None:
    if not node.level:
        return node.module
    # A root-level package's name depends on the checkout directory, so it has no parts.
    parts = [part for part in located[path][1].split(".") if part]
    base = parts if path.rpartition("/")[2] == "__init__.py" else parts[:-1]
    kept = len(base) - node.level + 1
    if kept < 1:
        return None
    return ".".join([*base[:kept], *([node.module] if node.module else [])])


def _imports(
    path: str, nodes: list[ast.AST], located: dict[str, tuple[str, str]], providers: Providers
) -> set[Import]:
    root = located[path][0]
    rows = set()
    for node in nodes:
        match node:
            case ast.Import():
                rows.update(
                    Import(path, node.lineno, *_deepest(providers, alias.name, root))
                    for alias in node.names
                )
            case ast.ImportFrom():
                source = _absolute(node, path, located)
                if source is None:
                    rows.add(
                        Import(path, node.lineno, "." * node.level + (node.module or ""), None)
                    )
                    continue
                for alias in node.names:
                    module, target = _deepest(providers, f"{source}.{alias.name}", root)
                    rows.add(Import(path, node.lineno, module if target else source, target))
    return rows


def _bindings(
    path: str, nodes: list[ast.AST], located: dict[str, tuple[str, str]], providers: Providers
) -> dict[str, Binding]:
    root = located[path][0]
    bindings: dict[str, Binding] = {}
    for node in nodes:
        match node:
            case ast.Import():
                for alias in node.names:
                    module = alias.name if alias.asname else alias.name.partition(".")[0]
                    target = _find(providers, module, root)
                    bindings[alias.asname or module] = ((target, ""), "") if target else None
            case ast.ImportFrom():
                source = _absolute(node, path, located)
                target = source and _find(providers, source, root)
                for alias in node.names:
                    bindings[alias.asname or alias.name] = (
                        ((target, ""), alias.name) if target else None
                    )
            case ast.ClassDef() | ast.FunctionDef() | ast.AsyncFunctionDef():
                bindings[node.name] = ((path, node.name), "")
    return bindings


def _references(
    tests: dict[str, list[tuple[str, str, ast.AST]]],
    located: dict[str, tuple[str, str]],
    providers: Providers,
    namespaces: dict[str, dict[str, Binding]],
    symbols: set[Entity],
) -> set[Reference]:
    # A loop, not recursion, so a long chain of re-exports cannot exhaust the stack.
    def resolve(binding: Binding, loaded: set[str]) -> Entity | None:
        seen = set()
        while binding:
            (path, qualname), name = binding
            if not name:
                return path, qualname
            if qualname:
                member = path, f"{qualname}.{name}"
                return member if member in symbols else None
            root, module = located[path]
            bound = namespaces.get(path, {})
            submodule = _find(providers, f"{module}.{name}", root)
            target = bound.get(name)
            # A name the module binds shadows its same-named submodule. A test's import of that
            # submodule rebinds it, unless the binding came from the submodule and so loaded it
            # first, as in `from .train import train`.
            rebound = f"{module}.{name}" in loaded and not (target and target[0][0] == submodule)
            if name not in bound or (path, name) in seen or rebound:
                return (submodule, "") if submodule else None
            seen.add((path, name))
            binding = target
        return None

    rows = set()
    for test_path, walked in tests.items():
        imported = [node for _, _, node in walked if isinstance(node, ast.Import | ast.ImportFrom)]
        bindings = _bindings(test_path, imported, located, providers)
        loaded = {
            prefix
            for node in imported
            if isinstance(node, ast.Import)
            for alias in node.names
            for prefix in accumulate(alias.name.split("."), "{}.{}".format)
        }
        inner = set()
        for scope, _, node in walked:
            if isinstance(node, ast.Attribute):
                inner.add(id(node.value))
            if (
                id(node) in inner
                or not isinstance(node, ast.Name | ast.Attribute)
                or not isinstance(node.ctx, ast.Load)
            ):
                continue
            line, names = node.lineno, []
            while isinstance(node, ast.Attribute):
                names.append(node.attr)
                node = node.value
            if not isinstance(node, ast.Name):
                continue
            entity = resolve(bindings.get(node.id), loaded)
            for name in reversed(names):
                if not (step := entity and resolve((entity, name), loaded)):
                    break
                entity = step
            if entity and entity[1] and entity[0] != test_path:
                rows.add(Reference(test_path, scope, line, *entity))
    return rows


def _flatten(table: dict, keys: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], object]]:
    for key, value in table.items():
        yield (*keys, key), value
        if isinstance(value, dict):
            yield from _flatten(value, (*keys, key))


def _pyproject(path: str, content: bytes) -> list[Dependency]:
    rows = []
    for keys, value in _flatten(tomllib.loads(content.decode())):
        label = ".".join(keys)
        match keys:
            case ("build-system" | "project" | "dependency-groups",) | (
                "project",
                "optional-dependencies",
            ) if not isinstance(value, dict):
                raise ValueError(f"{label} must be a table")
            case ("build-system", "requires"):
                group = "build-system"
            case ("project", "dependencies"):
                group = "project"
            case ("project", "optional-dependencies", extra):
                group = f"optional:{extra}"
            case ("dependency-groups", name):
                group = f"group:{name}"
            case _:
                continue
        if not isinstance(value, list):
            raise ValueError(f"{label} must be an array")
        for item in value:
            match item:
                case str():
                    rows.append(Dependency(path, group, item))
                case {"include-group": str(included)} if keys[0] == "dependency-groups":
                    rows.append(Dependency(path, group, f"include-group:{included}"))
                case _:
                    raise ValueError(f"{label} has an invalid entry {item!r}")
    return rows


def _requirements(path: str, content: bytes) -> list[Dependency]:
    # As in pip, a trailing backslash continues any line except a comment line.
    return [
        Dependency(path, "requirements", requirement)
        for line in re.sub(r"(?m)^(?!\s*#)(.*)\\\r?\n", r"\1", content.decode()).splitlines()
        if (requirement := line.split(" #", 1)[0].strip()) and not requirement.startswith("#")
    ]


_DECLARATIONS = {"pyproject.toml": _pyproject, "requirements*.txt": _requirements}


def _declaration(path: str) -> Callable[[str, bytes], list[Dependency]] | None:
    name = path.rpartition("/")[2]
    return next(
        (read for pattern, read in _DECLARATIONS.items() if fnmatchcase(name, pattern)), None
    )


def _parse(path: str, content: bytes) -> ast.Module:
    with warnings.catch_warnings():
        # Under -W error a SyntaxWarning becomes a SyntaxError; parsing must not depend on filters.
        warnings.simplefilter("ignore")
        return ast.parse(content, filename=path)


def _failure(error: SyntaxError | ValueError | RecursionError | MemoryError) -> str:
    if isinstance(error, SyntaxError):
        return f"{type(error).__name__}: {error.msg} (line {error.lineno})"
    return f"{type(error).__name__}: {error}"


def _index(commit: str, tree: str, files: dict[str, tuple[str, bytes]]) -> CodeIndex:
    sources = sorted(path for path in files if path.endswith(".py"))
    packages = {
        directory
        for directory, _, filename in (path.rpartition("/") for path in sources)
        if filename == "__init__.py"
    }
    located = {path: _locate(path, packages) for path in sources}
    providers: Providers = {}
    for path, (root, name) in located.items():
        providers.setdefault(name, []).append((root, path))
    modules, symbols, imports, failures = [], [], set(), []
    namespaces: dict[str, dict[str, Binding]] = {}
    tests: dict[str, list[tuple[str, str, ast.AST]]] = {}
    for path in sources:
        blob, content = files[path]
        modules.append(Module(path, located[path][1], blob))
        try:
            syntax = _parse(path, content)
        # CPython reports source past its parser's depth limits as these errors, not SyntaxError.
        except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
            failures.append(Failure(path, _failure(error)))
            continue
        walked = list(_walk(syntax))
        symbols.extend(
            Symbol(
                path,
                prefix + node.name,
                kind,
                node.decorator_list[0].lineno if node.decorator_list else node.lineno,
                node.end_lineno,
            )
            for _, prefix, node in walked
            if (kind := _KINDS.get(type(node)))
        )
        imports |= _imports(path, [node for _, _, node in walked], located, providers)
        namespaces[path] = _bindings(
            path, [node for _, prefix, node in walked if not prefix], located, providers
        )
        if any(fnmatchcase(path.rpartition("/")[2], pattern) for pattern in _TESTS):
            tests[path] = walked
    references = _references(
        tests, located, providers, namespaces, {(row.path, row.qualname) for row in symbols}
    )
    dependencies = []
    for path in sorted(files):
        if read := _declaration(path):
            try:
                dependencies.extend(read(path, files[path][1]))
            except (ValueError, RecursionError) as error:
                failures.append(Failure(path, str(error)))
    return CodeIndex(
        commit,
        tree,
        INDEXER,
        f"{platform.python_implementation()} {platform.python_version()}",
        tuple(modules),
        tuple(
            sorted(symbols, key=lambda row: (row.path, row.start_line, row.end_line, row.qualname))
        ),
        tuple(sorted(imports, key=lambda row: (row.path, row.line, row.module, row.target or ""))),
        tuple(
            sorted(
                references,
                key=lambda row: (row.test_path, row.line, row.path, row.qualname, row.scope),
            )
        ),
        tuple(dependencies),
        tuple(sorted(failures, key=lambda row: row.path)),
    )


def index_commit(repository: Path, commit: str) -> CodeIndex:
    """Index the Python code of a commit from Git objects alone.

    Args:
        repository (Path): Directory of a local Git repository.
        commit (str): Any revision that Git resolves to a commit.

    Returns:
        (CodeIndex) Modules, symbols, imports, test references, dependencies, and per-file
            failures, read without checking out, importing, or executing repository code.
    """
    git = partial(_git, _executable(repository), repository)
    oid = git("rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}").decode().strip()
    tree = git("rev-parse", "--verify", f"{oid}^{{tree}}").decode().strip()
    blobs = {}
    for entry in git("ls-tree", "-rz", "--full-tree", oid).split(b"\0")[:-1]:
        metadata, raw_path = entry.split(b"\t", 1)
        mode, _, blob = metadata.decode().split()
        path = os.fsdecode(raw_path)
        if mode in ("100644", "100755") and (path.endswith(".py") or _declaration(path)):
            blobs[path] = blob
    contents = _read_blobs(git, sorted(set(blobs.values())))
    return _index(oid, tree, {path: (blob, contents[blob]) for path, blob in blobs.items()})
