from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from quant_research.code_index import CodeIndexError, index_commit

ROOT = Path(__file__).resolve().parents[1]
SHAPES = """\
import functools


class Model:
    class Config:
        depth = 2

    def fit(self):
        return self


def outer():
    def inner():
        return 1

    return inner


@functools.cache
def cached():
    return 1


async def fetch():
    return 1


if True:

    def guarded():
        return 1
"""
DEFINITIONS = {
    "Model": (
        "class",
        "class Model:\n    class Config:\n        depth = 2\n\n"
        "    def fit(self):\n        return self",
    ),
    "Model.Config": ("class", "    class Config:\n        depth = 2"),
    "Model.fit": ("function", "    def fit(self):\n        return self"),
    "outer": (
        "function",
        "def outer():\n    def inner():\n        return 1\n\n    return inner",
    ),
    "outer.<locals>.inner": ("function", "    def inner():\n        return 1"),
    "cached": ("function", "@functools.cache\ndef cached():\n    return 1"),
    "fetch": ("function", "async def fetch():\n    return 1"),
    "guarded": ("function", "    def guarded():\n        return 1"),
}
PROJECT = {
    "src/pkg/__init__.py": "",
    "src/pkg/shapes.py": SHAPES,
    "src/pkg/données.py": 'def défaut():\n    return "\\d"\n',
    "src/pkg/broken.py": "def broken(:\n",
    "tests/test_shapes.py": "from pkg.shapes import Model\n\n\ndef test_fit():\n    Model.fit()\n",
}
GRAPH = {
    "__init__.py": "from .a import thing\n",
    "setup.py": "import setuptools\n",
    "src/pkg.py": "",
    "src/pkg/__init__.py": "from .model import Model\n",
    "src/pkg/model.py": "import os\nfrom . import util\nfrom .util import helper, missing\n",
    "src/pkg/util.py": "def helper():\n    return 1\n",
    "src/pkg/sub/__init__.py": "",
    "src/pkg/sub/deep.py": "from .. import model\nfrom ... import beyond\n",
    "src/helpers.py": "",
    "tests/helpers.py": "",
    "tests/test_graph.py": (
        "import pkg.sub.deep\nimport pkg.missing.thing\nfrom pkg import model, Model\n"
        "import numpy.linalg\nimport helpers\nfrom . import sibling\n\n\n"
        "def test_inside():\n    import json\n"
    ),
    "scripts/run.py": "import helpers\n",
}


def git(repository: Path, *arguments: str, data: str | None = None) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    result = subprocess.run(
        [
            "git",
            *("-c", "user.name=Test", "-c", "user.email=test@example.com"),
            *("-c", "commit.gpgsign=false", *arguments),
        ],
        cwd=repository,
        env=environment | {"GIT_CONFIG_NOSYSTEM": "1"},
        input=data,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def commit(repository: Path, files: dict[str, str | bytes]) -> str:
    for name, content in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    git(repository, "add", "--all")
    git(repository, "commit", "--quiet", "--message", "Fixture")
    return git(repository, "rev-parse", "HEAD")


def line(source: str, text: str) -> int:
    return source[: source.index(text)].count("\n") + 1


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    # The index drops `GIT_*` variables, so only a private home hides the host's global Git config.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home/.config"))


@pytest.fixture
def repository(tmp_path):
    path = tmp_path / "repository"
    path.mkdir()
    git(path, "init", "--quiet", "--template=")
    return path


@pytest.mark.smoke
def test_symbols_record_qualnames_kinds_and_definition_lines(repository):
    commit(repository, {"src/pkg/__init__.py": "", "src/pkg/shapes.py": SHAPES})
    index = index_commit(repository, "HEAD")
    assert [(row.path, row.qualname, row.kind) for row in index.symbols] == [
        ("src/pkg/shapes.py", qualname, kind) for qualname, (kind, _) in DEFINITIONS.items()
    ]
    lines = SHAPES.splitlines()
    assert {
        row.qualname: "\n".join(lines[row.start_line - 1 : row.end_line]) for row in index.symbols
    } == {qualname: text for qualname, (_, text) in DEFINITIONS.items()}


def test_definitions_declared_global_take_module_level_qualnames(repository):
    commit(
        repository,
        {
            "app.py": (
                "def outer():\n    global helper, Config\n\n    def helper():\n"
                "        def inner():\n            return 1\n\n"
                "    class Config:\n        def method(self):\n            return 1\n\n"
                "    def local():\n        return 1\n\n\n"
                "class Holder:\n    global made\n\n    def made(self):\n        return 1\n"
            )
        },
    )
    assert [row.qualname for row in index_commit(repository, "HEAD").symbols] == [
        "outer",
        "helper",
        "helper.<locals>.inner",
        "Config",
        "Config.method",
        "outer.<locals>.local",
        "Holder",
        "made",
    ]


def test_output_is_canonical_and_identical_across_hash_seeds_and_warning_filters(repository):
    commit(repository, PROJECT)
    script = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "from pathlib import Path; from quant_research.code_index import index_commit; "
        f"index = index_commit(Path({str(repository)!r}), 'HEAD'); "
        "sys.stdout.buffer.write(index.to_json()); print(index.digest, hash('probe'))"
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    runs = [
        subprocess.run(
            [sys.executable, "-B", "-P", "-s", *flags, "-c", script],
            env=environment | {"PYTHONHASHSEED": seed},
            capture_output=True,
            check=True,
        ).stdout.splitlines()
        for seed, flags in (("0", ()), ("1", ("-W", "error")))
    ]
    (first, first_digest, first_hash), (second, second_digest, second_hash) = (
        (data, *line.split()) for data, line in runs
    )
    index = index_commit(repository, "HEAD")
    assert first_hash != second_hash
    assert first + b"\n" == second + b"\n" == index.to_json()
    assert first_digest.decode() == second_digest.decode() == index.digest
    assert index.digest == hashlib.sha256(index.to_json()).hexdigest()
    canonical = json.dumps(
        json.loads(first),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert canonical.encode() == first
    assert "défaut" in canonical


def test_index_reads_the_commit_not_the_working_tree_or_staging_area(repository):
    first = commit(
        repository,
        {
            "src/app.py": "def run():\n    return 1\n",
            "src/removed.py": "def gone():\n    return 0\n",
        },
    )
    clean = index_commit(repository, first).to_json()
    (repository / "src/app.py").write_text(
        "def run():\n    return 2\n\n\ndef edited():\n    pass\n"
    )
    (repository / "src/staged.py").write_text("def staged():\n    return 3\n")
    git(repository, "add", "src/staged.py")
    (repository / "src/removed.py").unlink()
    dirty = index_commit(repository, "HEAD")
    assert dirty.to_json() == clean
    assert "src/removed.py" in {row.path for row in dirty.modules}
    commit(repository, {"src/app.py": "\n\ndef start():\n    return 2\n"})
    assert [(row.qualname, row.start_line) for row in index_commit(repository, "HEAD").symbols] == [
        ("start", 3),
        ("staged", 1),
    ]
    assert index_commit(repository, first).to_json() == clean


def test_parse_failure_is_reported_for_that_file_and_the_rest_is_indexed(repository):
    commit(
        repository,
        {
            "good.py": "def ok():\n    return 1\n",
            "bad.py": "def ok():\n    return 1\n\n\ndef broken(:\n    pass\n",
            "latin.py": b"# -*- coding: latin-1 -*-\ndef caf\xe9():\n    return 1\n",
            "undeclared.py": b"text = '\xe9'\n",
            "deep.py": "x = " + "-" * 10_000 + "1\n",
            "wide.py": "x = " + "+".join(["1"] * 10_000) + "\n",
        },
    )
    index = index_commit(repository, "HEAD")
    assert [row.path for row in index.modules] == [
        "bad.py",
        "deep.py",
        "good.py",
        "latin.py",
        "undeclared.py",
        "wide.py",
    ]
    assert [(row.path, row.qualname) for row in index.symbols] == [
        ("good.py", "ok"),
        ("latin.py", "café"),
    ]
    assert [row.path for row in index.failures] == ["bad.py", "deep.py", "undeclared.py", "wide.py"]
    assert index.failures[0].error.startswith("SyntaxError: ")
    assert index.failures[0].error.endswith("(line 5)")


def test_indexing_never_runs_repository_code_or_git_configuration(repository, tmp_path):
    marker = tmp_path / "marker"
    trap = f"from pathlib import Path\n\nPath({str(marker)!r}).touch()\nraise SystemExit(1)\n"
    commit(
        repository,
        {
            "pkg/__init__.py": trap,
            "conftest.py": trap,
            "setup.py": trap,
            "tests/test_trap.py": trap,
            "trap.pth": f"import os; os.system('touch {shlex.quote(str(marker))}')\n",
            ".gitattributes": "* filter=trap diff=trap\n",
            "git": f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\n",
        },
    )
    (repository / "git").chmod(0o755)
    script = tmp_path / "trap.sh"
    script.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\n")
    script.chmod(0o755)
    (repository / "bin").mkdir()
    (repository / "bin" / "git").symlink_to(script)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "git").symlink_to(script)
    (repository / "outside").symlink_to(tmp_path / "outside")
    (tmp_path / "tools").symlink_to(tmp_path / "outside")
    (tmp_path / "bin").symlink_to(repository / "bin")
    (tmp_path / "pkg").symlink_to(repository / "pkg")
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    (tmp_path / ".git").symlink_to("/" + "x" * 300)
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "--quiet", "--template=")
    commit(other, {"other.py": ""})
    (repository / "other").symlink_to(other)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for hook in ("post-index-change", "reference-transaction", "pre-auto-gc"):
        (hooks / hook).symlink_to(script)
    for key, value in {
        "core.hooksPath": hooks,
        "core.fsmonitor": script,
        "core.pager": script,
        "pager.cat-file": script,
        "credential.helper": script,
        "gpg.program": script,
        "filter.trap.clean": script,
        "filter.trap.smudge": script,
        "diff.trap.textconv": script,
        "extensions.partialClone": "origin",
        "remote.origin.promisor": "true",
        "remote.origin.url": "ssh://example.invalid/repository",
        "core.sshCommand": script,
    }.items():
        git(repository, "config", key, str(value))
    for command in ("rev-parse", "ls-tree", "cat-file"):
        git(repository, "config", f"alias.{command}", f"!{script}")
    missing = git(repository, "mktree", "--missing", data=f"100644 blob {'1' * 40}\tlost.py\n")
    lazy = git(repository, "commit-tree", missing, "-m", "Missing blob")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(repository)
        unreadable = f"{tmp_path / 'loop'}:/{'x' * 300}:{tmp_path}/missing/../outside"
        inside = (
            f"{repository}:{repository / 'bin'}:{repository / 'outside'}:"
            f"{repository}/outside/../tools:{tmp_path / 'bin'}"
        )
        patch.setenv("PATH", f":{unreadable}:{inside}:{os.environ['PATH']}")
        modules = set(sys.modules)
        index = index_commit(repository, "HEAD")
        # Only a case-insensitive volume has the second spelling of the checkout.
        for supplied in filter(
            Path.exists, (repository / "pkg", tmp_path / "pkg", Path(str(repository).swapcase()))
        ):
            assert index_commit(supplied, "HEAD") == index
        assert [row.path for row in index_commit(repository / "other", "HEAD").modules] == [
            "other.py"
        ]
        for supplied, revision in (
            (repository, ":0:conftest.py"),
            (repository, lazy),
            (repository / "outside", "HEAD"),
        ):
            with pytest.raises(CodeIndexError):
                index_commit(supplied, revision)
        assert set(sys.modules) == modules
        patch.setenv("PATH", f":{inside}")
        with pytest.raises(CodeIndexError, match="No git executable"):
            index_commit(repository, "HEAD")
    assert not marker.exists()
    assert [row.path for row in index.modules] == [
        "conftest.py",
        "pkg/__init__.py",
        "setup.py",
        "tests/test_trap.py",
    ]
    git(repository, "status")
    assert marker.exists()


def test_unreadable_commits_non_repositories_and_broken_git_raise_code_index_error(
    repository, tmp_path, monkeypatch
):
    commit(repository, {"app.py": "value = 1\n"})
    missing_blob = git(repository, "mktree", "--missing", data=f"100644 blob {'1' * 40}\tlost.py\n")
    dangling = git(repository, "commit-tree", missing_blob, "-m", "Missing blob")
    plain = tmp_path / "plain"
    plain.mkdir()
    for path, revision in (
        (repository, "0" * 40),
        (repository, "no-such-branch"),
        (repository, "--output=/dev/null"),
        (repository, dangling),
        (plain, "HEAD"),
        (tmp_path / "absent", "HEAD"),
        (tmp_path / ("x" * 300), "HEAD"),
        (repository / "app.py", "HEAD"),
    ):
        with pytest.raises(CodeIndexError):
            index_commit(path, revision)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "git").write_text("#!/nonexistent/interpreter\n")
    (tmp_path / "broken" / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'broken'}:{os.environ['PATH']}")
    with pytest.raises(CodeIndexError, match="cannot start"):
        index_commit(repository, "HEAD")


def test_modules_take_import_names_from_their_root_and_pin_blob_ids(repository):
    (repository / "link.py").symlink_to("setup.py")
    commit(repository, GRAPH)
    names = {
        "__init__.py": "",
        "scripts/run.py": "run",
        "setup.py": "setup",
        "src/helpers.py": "helpers",
        "src/pkg.py": "pkg",
        "src/pkg/__init__.py": "pkg",
        "src/pkg/model.py": "pkg.model",
        "src/pkg/sub/__init__.py": "pkg.sub",
        "src/pkg/sub/deep.py": "pkg.sub.deep",
        "src/pkg/util.py": "pkg.util",
        "tests/helpers.py": "helpers",
        "tests/test_graph.py": "test_graph",
    }
    assert [(row.path, row.name, row.blob) for row in index_commit(repository, "HEAD").modules] == [
        (path, names[path], git(repository, "rev-parse", f"HEAD:{path}")) for path in sorted(GRAPH)
    ]


def test_imports_resolve_to_repository_modules_or_stay_external(repository):
    commit(repository, GRAPH)
    test = "tests/test_graph.py"
    assert [
        (row.path, row.line, row.module, row.target)
        for row in index_commit(repository, "HEAD").imports
    ] == [
        ("__init__.py", 1, ".a", None),
        ("scripts/run.py", 1, "helpers", "src/helpers.py"),
        ("setup.py", 1, "setuptools", None),
        ("src/pkg/__init__.py", 1, "pkg.model", "src/pkg/model.py"),
        ("src/pkg/model.py", 1, "os", None),
        ("src/pkg/model.py", 2, "pkg.util", "src/pkg/util.py"),
        ("src/pkg/model.py", 3, "pkg.util", "src/pkg/util.py"),
        ("src/pkg/sub/deep.py", 1, "pkg.model", "src/pkg/model.py"),
        ("src/pkg/sub/deep.py", 2, "...", None),
        (test, 1, "pkg.sub.deep", "src/pkg/sub/deep.py"),
        (test, 2, "pkg", "src/pkg/__init__.py"),
        (test, 3, "pkg", "src/pkg/__init__.py"),
        (test, 3, "pkg.model", "src/pkg/model.py"),
        (test, 4, "numpy.linalg", None),
        (test, 5, "helpers", "tests/helpers.py"),
        (test, 6, ".", None),
        (test, line(GRAPH[test], "import json"), "json", None),
    ]


def test_dependencies_keep_each_declaration_in_file_order(repository):
    commit(
        repository,
        {
            "pyproject.toml": (
                '[project]\nname = "fixture"\ndependencies = ["numpy>=2", "pandas"]\n\n'
                '[project.optional-dependencies]\nplot = ["matplotlib"]\n\n'
                '[build-system]\nrequires = ["hatchling>=1.27"]\n\n'
                '[dependency-groups]\ntest = ["pytest>=8"]\n'
                'dev = [{include-group = "test"}, "ruff"]\n\n'
                "[tool.uv]\npackage = false\n"
            ),
            "ci/requirements-dev.txt": (
                "# tools\n\npytest>=8  # runner\n--index-url https://example.com/simple\n  ruff \n"
                "numpy==2.0 \\\n    --hash=sha256:abc \\\r\n    --hash=sha256:def\n"
                "# pinned \\\nscipy==1.14\npandas  # data \\\npolars\n"
            ),
            "requirements.in": "ignored\n",
            "setup.py": "from setuptools import setup\n\nsetup(install_requires=['ignored'])\n",
        },
    )
    assert [
        (row.path, row.group, row.requirement)
        for row in index_commit(repository, "HEAD").dependencies
    ] == [
        ("ci/requirements-dev.txt", "requirements", "pytest>=8"),
        ("ci/requirements-dev.txt", "requirements", "--index-url https://example.com/simple"),
        ("ci/requirements-dev.txt", "requirements", "ruff"),
        (
            "ci/requirements-dev.txt",
            "requirements",
            "numpy==2.0     --hash=sha256:abc     --hash=sha256:def",
        ),
        ("ci/requirements-dev.txt", "requirements", "scipy==1.14"),
        ("ci/requirements-dev.txt", "requirements", "pandas"),
        ("pyproject.toml", "project", "numpy>=2"),
        ("pyproject.toml", "project", "pandas"),
        ("pyproject.toml", "optional:plot", "matplotlib"),
        ("pyproject.toml", "build-system", "hatchling>=1.27"),
        ("pyproject.toml", "group:test", "pytest>=8"),
        ("pyproject.toml", "group:dev", "include-group:test"),
        ("pyproject.toml", "group:dev", "ruff"),
    ]


def test_malformed_pyproject_is_a_failure_and_declares_nothing(repository):
    commit(
        repository,
        {
            "pyproject.toml": '[build-system]\nrequires = ["hatchling"]\n',
            "syntax/pyproject.toml": "[project\n",
            "encoding/pyproject.toml": b'[project]\ndependencies = ["caf\xe9"]\n',
            "array/pyproject.toml": (
                '[project]\ndependencies = ["numpy"]\noptional-dependencies = []\n'
            ),
            "entry/pyproject.toml": '[build-system]\nrequires = ["ok", {include-group = "x"}]\n',
            "nested/pyproject.toml": "x = " + "[" * 2000 + "]" * 2000 + "\n",
        },
    )
    index = index_commit(repository, "HEAD")
    assert [(row.path, row.requirement) for row in index.dependencies] == [
        ("pyproject.toml", "hatchling")
    ]
    assert [row.path for row in index.failures] == [
        "array/pyproject.toml",
        "encoding/pyproject.toml",
        "entry/pyproject.toml",
        "nested/pyproject.toml",
        "syntax/pyproject.toml",
    ]


def test_test_references_resolve_to_the_deepest_repository_symbol(repository):
    tests = """\
import json

import pkg
import pkg.util as tools
import pytest
from pkg import Model, evaluate, path
from pkg.cycle import loop
from pkg.model import train
from test_refs import local_helper

SUBJECT = Model


def local_helper():
    return 1


class TestModel:
    def test_fit(self):
        assert Model.fit(Model()) is not None

    @pytest.mark.parametrize("value", [Model.missing])
    def test_missing(self, value):
        assert pkg.Model.fit and pkg.util.helper()


def test_train():
    assert train() and tools.helper() and json.dumps(1) and local_helper() and tools
    assert evaluate() and path.join("a")
    Model.fit = loop() or train
"""
    commit(
        repository,
        {
            "src/pkg/__init__.py": (
                "from os import path\n\nfrom .evaluate import evaluate\nfrom .model import Model\n"
            ),
            "src/pkg/path.py": "def join():\n    return 1\n",
            "src/pkg/evaluate.py": "def evaluate():\n    return 1\n",
            "src/pkg/model.py": (
                "class Model:\n    def fit(self):\n        return self\n\n\n"
                "def train():\n    return Model()\n"
            ),
            "src/pkg/util.py": "def helper():\n    return 1\n",
            "src/pkg/cycle.py": "from pkg.cycle import loop\n",
            "tests/conftest.py": (
                "from pkg.util import helper\n\n\ndef make():\n    return helper()\n"
            ),
            "tests/test_refs.py": tests,
        },
    )
    model, util, evaluate = "src/pkg/model.py", "src/pkg/util.py", "src/pkg/evaluate.py"
    fit = line(tests, "assert Model.fit")
    missing = line(tests, "assert pkg.Model")
    train = line(tests, "assert train()")
    assert [
        (row.test_path, row.scope, row.line, row.path, row.qualname)
        for row in index_commit(repository, "HEAD").references
    ] == [
        ("tests/conftest.py", "make", 5, util, "helper"),
        ("tests/test_refs.py", "", line(tests, "SUBJECT"), model, "Model"),
        ("tests/test_refs.py", "TestModel.test_fit", fit, model, "Model"),
        ("tests/test_refs.py", "TestModel.test_fit", fit, model, "Model.fit"),
        ("tests/test_refs.py", "TestModel", line(tests, "@pytest"), model, "Model"),
        ("tests/test_refs.py", "TestModel.test_missing", missing, model, "Model.fit"),
        ("tests/test_refs.py", "TestModel.test_missing", missing, util, "helper"),
        ("tests/test_refs.py", "test_train", train, model, "train"),
        ("tests/test_refs.py", "test_train", train, util, "helper"),
        (
            "tests/test_refs.py",
            "test_train",
            line(tests, "assert evaluate()"),
            evaluate,
            "evaluate",
        ),
        ("tests/test_refs.py", "test_train", line(tests, "Model.fit ="), model, "train"),
    ]


def test_long_re_export_chains_resolve_without_exhausting_the_stack(repository):
    depth = sys.getrecursionlimit()
    commit(
        repository,
        {f"m{step}.py": f"from m{step + 1} import thing\n" for step in range(depth)}
        | {
            f"m{depth}.py": "def thing():\n    return 1\n",
            "tests/test_chain.py": (
                "from m0 import thing\n\n\ndef test_thing():\n    assert thing()\n"
            ),
        },
    )
    assert [
        (row.line, row.path, row.qualname) for row in index_commit(repository, "HEAD").references
    ] == [(5, f"m{depth}.py", "thing")]


def test_importing_a_submodule_rebinds_it_unless_the_package_loaded_it_first(repository):
    commit(
        repository,
        {
            "src/pkg/__init__.py": "from os import path\n\nfrom .train import train\n",
            "src/pkg/path.py": "def join():\n    return 1\n",
            "src/pkg/train.py": "def train():\n    return 1\n",
            "tests/test_path.py": (
                "import pkg.path\nimport pkg.train\n\n\n"
                "def test_join():\n    assert pkg.path.join() and pkg.train()\n"
            ),
        },
    )
    assert [
        (row.line, row.path, row.qualname) for row in index_commit(repository, "HEAD").references
    ] == [(6, "src/pkg/path.py", "join"), (6, "src/pkg/train.py", "train")]
