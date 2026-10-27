from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from temporalio.client import Client
from temporalio.worker import Replayer

from quant_research import prospective, research
from quant_research.access import NO_MEMORY
from quant_research.campaigns import PROMPT, ModelRole, register, save_revision
from quant_research.catalog import Catalog
from quant_research.packages import open_package
from quant_research.providers import CallRecord
from quant_research.research_worker import load_worker
from quant_research.sources import SourceDeclaration
from tests import test_campaign_cli as cli_tests
from tests import test_claude_adapter as claude_tests
from tests import test_codex_adapter as codex_tests
from tests import test_research as research_tests


@pytest.fixture
def config(tmp_path):
    for name in ("sources", "datasets", "calls", "rules"):
        (tmp_path / name).mkdir()
    (tmp_path / "catalog.json").write_bytes(Catalog(()).to_bytes())
    (tmp_path / "uv.lock").write_bytes(b"synthetic lock\n")
    value = {
        "schema_version": 1,
        "temporal": {"address": "127.0.0.1:7233", "task_queue": "research"},
        "calls": "calls",
        "archive": "archive",
        "setup": {
            "sources": "sources",
            "datasets": "datasets",
            "catalog": "catalog.json",
            "packages": "packages",
            "lock": "uv.lock",
            "cutoff": "2025-06-01T00:00:00Z",
            "clearances": [],
            "rules": "rules",
        },
        "routes": [
            {"provider": "claude-subscription"},
            {"provider": "codex-subscription", "retains": True},
        ],
    }
    return tmp_path / "worker.json", value


def write_config(config):
    path, value = config
    path.write_text(json.dumps(value))
    return path


def test_worker_uses_config_paths_fixed_policies_and_the_same_calls_store(
    config, tmp_path, monkeypatch
):
    path, value = config
    (tmp_path / "auth.json").write_text("synthetic auth")
    executable = tmp_path / "bin" / "codex"
    executable.parent.mkdir()
    executable.write_text("unused")
    value["routes"][1] |= {"auth": "auth.json", "executable": "bin/codex"}
    monkeypatch.chdir(tmp_path.parent)
    configured = load_worker(write_config(config))
    assert configured.calls == tmp_path / "calls"
    assert configured.setup.sources == tmp_path / "sources"
    assert configured.setup.datasets == tmp_path / "datasets"
    assert configured.setup.rules == tmp_path / "rules"
    assert configured.setup.lock == b"synthetic lock\n"
    assert configured.setup.ledger.events == ()
    assert configured.setup.memory == NO_MEMORY
    assert configured.setup.generated is None
    assert not configured.archive.exists()
    assert not configured.setup.packages.exists()
    claude, codex = configured.setup.policies
    assert claude.labels == codex.labels == frozenset()
    assert claude.retains is codex.retains is True
    assert configured.gateway._routes[claude.provider][0] is claude
    assert configured.gateway._routes[codex.provider][0] is codex
    assert configured.gateway._routes[claude.provider][1].executable == Path("claude")
    adapter = configured.gateway._routes[codex.provider][1]
    assert adapter.codex == str(executable)
    assert adapter.auth == tmp_path / "auth.json"
    captured = []
    monkeypatch.setattr(
        research, "run_worker", lambda *args, **kwargs: captured.append((args, kwargs))
    )
    configured.run()
    [(args, kwargs)] = captured
    assert args[:3] == (configured.address, configured.task_queue, configured.gateway)
    assert args[3].keywords["setup"] is configured.setup
    assert kwargs["archive"].calls == configured.calls
    assert kwargs["archive"].root == configured.archive
    assert kwargs["archive"].importer is None
    assert configured.setup.packages.is_dir()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("schema_version", "1"),
        ("temporal", {"address": "localhost", "task_queue": "q", "namespace": "other"}),
        ("routes", []),
        ("routes", [{"provider": "api"}]),
        ("routes", [{"provider": "codex-subscription", "retains": "true"}]),
        ("routes", [{"provider": "codex-subscription"}]),
        ("routes", [{"provider": "claude-subscription", "retains": False}]),
        ("routes", [{"provider": "claude-subscription", "argv": []}]),
        ("routes", [{"provider": "claude-subscription"}] * 2),
        ("routes", [{"provider": "claude-subscription", "executable": "missing/claude"}]),
        ("calls", "missing"),
        ("archive", 12),
    ],
)
def test_invalid_worker_config_fails_without_publication(config, field, value, cli, tmp_path):
    path, document = config
    document[field] = value
    result = cli(tmp_path, "campaign", "worker", str(write_config(config)), success=False)
    assert result.returncode == 1
    assert result.stderr.startswith("Campaign worker failed: ")
    assert "temporalio" not in result.stderr
    assert not (tmp_path / "archive").exists()
    assert not (tmp_path / "packages").exists()


@pytest.mark.parametrize("field", ["ledger", "memory", "generated", "importer", "api_key"])
def test_worker_refuses_unsupported_bindings(config, field):
    config[1]["setup"][field] = None
    with pytest.raises(ValueError, match="unknown"):
        load_worker(write_config(config))


@pytest.mark.parametrize(
    "field,value",
    [
        ("clearances", [7]),
        ("clearances", "public"),
        ("cutoff", "2025-06-01T00:00:00+00:00"),
        ("sources", "missing"),
        ("datasets", "missing"),
        ("rules", "missing"),
        ("lock", "missing"),
        ("catalog", "missing"),
    ],
)
def test_worker_validates_every_setup_input(config, field, value):
    config[1]["setup"][field] = value
    with pytest.raises((ValueError, OSError)):
        load_worker(write_config(config))


@pytest.mark.parametrize("damage", ["duplicate", "catalog", "calls"])
def test_worker_validates_json_catalog_and_call_log_before_outputs(config, damage, tmp_path):
    path = write_config(config)
    if damage == "duplicate":
        path.write_text(
            path.read_text().replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
            )
        )
    elif damage == "catalog":
        (tmp_path / "catalog.json").write_text("{}")
    else:
        (tmp_path / "calls" / "0.json").write_text("{}")
    with pytest.raises(ValueError):
        load_worker(path)
    assert not (tmp_path / "archive").exists()
    assert not (tmp_path / "packages").exists()


@pytest.mark.parametrize("output", ["archive", "packages"])
@pytest.mark.parametrize("layout", ["equal", "child", "parent", "alias", "symlink"])
def test_worker_requires_calls_disjoint_from_output_roots(config, tmp_path, cli, output, layout):
    value = config[1]
    paths = {
        "equal": "calls",
        "child": "calls/output",
        "parent": ".",
        "alias": "calls/../calls",
        "symlink": "calls-alias/output",
    }
    if layout == "symlink":
        (tmp_path / "calls-alias").symlink_to(tmp_path / "calls", target_is_directory=True)
    target = value if output == "archive" else value["setup"]
    target[output] = paths[layout]
    result = cli(tmp_path, "campaign", "worker", str(write_config(config)), success=False)
    assert result.returncode == 1
    assert result.stderr == f"Campaign worker failed: calls and {output} roots must be disjoint\n"
    assert not (tmp_path / "archive").exists()
    assert not (tmp_path / "packages").exists()
    assert list((tmp_path / "calls").iterdir()) == []


@pytest.mark.parametrize("packages", ["archive", "archive/packages", "packages"])
def test_worker_allows_output_overlap_outside_resolved_calls(config, tmp_path, packages):
    (tmp_path / "calls-alias").symlink_to(tmp_path / "calls", target_is_directory=True)
    config[1]["calls"] = "calls-alias/../calls-alias"
    config[1]["archive"] = "unused/../archive"
    config[1]["setup"]["packages"] = packages
    configured = load_worker(write_config(config))
    assert configured.calls == tmp_path / "calls"
    assert configured.archive == tmp_path / "archive"
    assert configured.setup.packages == tmp_path / packages
    assert not configured.archive.exists()


def test_worker_unreachable_temporal_prints_one_failure_line(config, tmp_path):
    config[1]["temporal"]["address"] = "127.0.0.1:1"
    result = cli_tests.campaign(tmp_path, "worker", str(write_config(config)))
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith(
        "Campaign worker failed: Cannot connect to Temporal at 127.0.0.1:1: "
    )
    assert len(result.stderr.splitlines()) == 1
    assert (tmp_path / "packages").is_dir()
    assert (tmp_path / "archive" / "objects").is_dir()


def test_worker_implementation_runtime_error_is_not_a_connection_error(config, monkeypatch):
    async def connected(address):
        return object()

    class BrokenWorker:
        async def run(self):
            raise RuntimeError("broken worker implementation")

    monkeypatch.setattr(research.Client, "connect", connected)
    monkeypatch.setattr(research, "worker", lambda *args, **kwargs: BrokenWorker())
    with pytest.raises(RuntimeError, match="broken worker implementation"):
        load_worker(write_config(config)).run()


def test_valid_worker_config_loads_temporal_only_at_run(config, tmp_path, cli):
    result = cli(tmp_path, "campaign", "worker", str(write_config(config)), success=False)
    assert result.stderr.endswith("ModuleNotFoundError: No module named 'temporalio'\n")
    assert not (tmp_path / "archive").exists()
    assert not (tmp_path / "packages").exists()


def test_worker_cli_runs_both_routes_restarts_and_publishes_once(temporal, config, tmp_path):
    path, value = config
    value["temporal"] = {"address": temporal, "task_queue": str(uuid4())}
    (tmp_path / "rules" / f"{cli_tests.RULES.digest}.json").write_bytes(cli_tests.RULES.data)
    codex_dir = tmp_path / "fake-codex"
    codex_dir.mkdir()
    codex = codex_tests.fake(
        codex_dir,
        turn=[
            codex_tests.tokens(40, 8),
            codex_tests.said(research_tests.ACCEPT),
            codex_tests.ended(),
        ],
    )
    value["routes"][1] |= {"executable": codex.codex, "auth": str(codex.auth)}
    claude_dir = tmp_path / "fake-claude"
    claude_dir.mkdir()
    claude = claude_dir / "claude"
    claude.write_text(f"#!{sys.executable} -IS\n{claude_tests.FAKE}")
    claude.chmod(0o755)
    value["routes"][0]["executable"] = str(claude)

    def reply(text):
        turns = (claude_tests.LISTING,), claude_tests.GATE, claude_tests.answered(text)
        (claude_dir / "scenario.json").write_text(
            json.dumps(
                {
                    "status": claude_tests.SIGNED_IN,
                    "code": 0,
                    "grandchild": False,
                    "turns": [[json.dumps(item) for item in turn] for turn in turns],
                }
            )
        )

    reply(research_tests.CONCLUDE)
    configured = load_worker(write_config(config))
    base = cli_tests.pilot(configured.setup)
    sources = tuple(
        register(
            configured.setup.sources,
            name=name,
            role=PROMPT,
            data=f"You are the {name}.".encode(),
            declaration=SourceDeclaration(
                media_type="text/plain",
                access_domain="repository",
                location=f"{name}.txt",
                revision="1",
                acquisition="test",
                rights="synthetic",
                labels=frozenset(),
                approved_for=frozenset({"claude-subscription", "codex-subscription"}),
                published_at=None,
                source_available_at=cli_tests.WRITTEN,
                availability_evidence="synthetic",
                received_at=cli_tests.WRITTEN,
            ),
        )
        for name in ("researcher", "critic")
    )
    revision = replace(
        base,
        specification=replace(
            base.specification,
            sources=sources,
            model_roles=(
                ModelRole("researcher", "claude-subscription/claude-opus-5-5", (), "active"),
                ModelRole("critic", "codex-subscription/gpt-6.1-sol", (), "active"),
            ),
        ),
    )
    revision_path = save_revision(tmp_path / "revisions", revision)
    command = [
        sys.executable,
        "-B",
        "-I",
        "-c",
        (
            f"import runpy, sys; sys.path[:] = {sys.path!r}; "
            f"sys.argv = {['quant_research', 'campaign', 'worker', str(path)]!r}; "
            "runpy.run_module('quant_research', run_name='__main__')"
        ),
    ]

    async def body():
        client = await Client.connect(temporal)
        handle = client.get_workflow_handle_for(
            research.ResearchLoop.run, str(revision.campaign_id)
        )
        process = None
        with (tmp_path / "worker.log").open("wb") as log:
            try:
                process = subprocess.Popen(command, cwd=tmp_path.parent, stdout=log, stderr=log)
                result = await asyncio.to_thread(
                    cli_tests.campaign,
                    tmp_path,
                    "start",
                    str(revision_path),
                    "--address",
                    temporal,
                    "--task-queue",
                    configured.task_queue,
                )
                assert result.returncode == 0, result.stderr
                paused = await research_tests.reaches(handle, "paused")
                assert paused.spent["model_calls"] == "1"
                process.terminate()
                await asyncio.to_thread(process.wait, 10)
                reply(research_tests.NO_RESULT)
                process = subprocess.Popen(command, cwd=tmp_path.parent, stdout=log, stderr=log)
                assert (
                    await handle.query(
                        research.ResearchLoop.status, rpc_timeout=timedelta(seconds=30)
                    )
                    == paused
                )
                await research.control(handle, "resume", actor="researcher", reason="Finish")
                report = await research.record(handle)
                assert report.assessment.independent
                assert report.assessment.classification == "no_result"
                records = [
                    entry
                    for entry in configured.gateway._log.load()
                    if isinstance(entry, CallRecord)
                ]
                assert [entry.model.provider for entry in records] == [
                    "claude-subscription"
                ] * 2 + ["codex-subscription"]
                assert all(entry.outcome == "completed" for entry in records)
                [receipt] = (configured.archive / "receipts").iterdir()
                package = open_package(
                    configured.archive / "packages" / "unselected" / receipt.read_text()
                )
                before = prospective.load_log(configured.archive / "prospective").events
                prospective.register_package(configured.archive / "prospective", package)
                assert prospective.load_log(configured.archive / "prospective").events == before
                assert len(before) == 1
                assert report.spent["model_calls"] == "3"
                history = await handle.fetch_history()
                process.terminate()
                await asyncio.to_thread(process.wait, 10)
                process = subprocess.Popen(command, cwd=tmp_path.parent, stdout=log, stderr=log)
                assert await research.record(handle) == report
                assert receipt.read_text() == package.digest
                assert prospective.load_log(configured.archive / "prospective").events == before
                await Replayer(workflows=[research.ResearchLoop]).replay_workflow(history)
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    await asyncio.to_thread(process.wait, 10)

    asyncio.run(asyncio.wait_for(body(), 90))
