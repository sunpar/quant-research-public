from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from quant_research.access import Provider, Restrictions, issue_context
from quant_research.accounting import Usage
from quant_research.codex_adapter import CodexAdapter, policy
from quant_research.model_provenance import UNKNOWN, ModelRecord, RoleAssignment
from quant_research.providers import (
    Answer,
    CallLog,
    Gateway,
    Lease,
    Material,
    Request,
    Served,
    Unanswered,
    dump_result,
    load_result,
)

CODEX = Provider("codex-subscription")
CUTOFF = datetime(2026, 10, 21, 12, tzinfo=UTC)
MODEL = ModelRecord("codex-subscription", "gpt-6.1-sol", UNKNOWN, UNKNOWN)
PROPOSER = RoleAssignment("proposer", MODEL)
CONTEXT = issue_context(clearances=(), cutoff=CUTOFF, generation="synthetic/v1", providers=(CODEX,))
PUBLIC = Restrictions(
    available_at=CUTOFF - timedelta(days=1), labels=frozenset(), approved_for=frozenset({CODEX})
)
PROMPT = "Propose a momentum signal"
SCORE = (("score", float), ("count", int))
REPLY = '{"score": 0.5, "count": 2}'
AGENT = "quant_research/0.162.1 (Mac OS 26.0.0; arm64) node"
SIGN_IN = b'{"tokens": "old"}'
TOKENS = frozenset({"input_tokens", "output_tokens"})
# The isolation set a reviewer signs: a change to the adapter's lists must change these too.
QUARRY_DISABLED = (
    "shell_tool",
    "stable_environment_tools",
    "multi_agent",
    "multi_agent_v2",
    "code_mode",
    "code_mode_only",
    "code_mode_host",
    "hooks",
    "plugins",
    "apps",
    "connectors",
    "image_generation",
    "imagegenext",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "in_app_browser",
    "memories",
    "memory_tool",
    "goals",
    "agent_message_board",
    "skill_search",
    "deferred_executor",
    "deferred_tool_world_state",
    "standalone_web_search",
    "token_budget",
    "current_time_reminder",
    "sleep_tool",
    "send_message_to_user_async",
    "shell_snapshot",
    "workspace_dependencies",
    "codex_git_commit",
)
OVERRIDES = (
    'model_provider="openai"',
    'forced_login_method="chatgpt"',
    'cli_auth_credentials_store="file"',
    "agents.enabled=false",
    "notify=[]",
    'web_search="disabled"',
    "features.skip_host_skill_discovery=true",
    "project_doc_max_bytes=0",
    'developer_instructions=""',
    "skills.include_instructions=false",
    "include_permissions_instructions=false",
    "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_environment_context=false",
    "tools.experimental_request_user_input.enabled=false",
    "tools.update_plan.enabled=false",
)

# A stand-in for codex app-server 0.162.1. Like the node shim, the leader reruns itself as a
# child that owns stdio and waits for it, so a stop must reach the child through the group.
APP_SERVER = r"""
import json, os, select, signal, subprocess, sys, time
from pathlib import Path

base = Path(sys.argv[0]).resolve().parent.parent
scenario = json.loads((base / "scenario.json").read_text())
if "FAKE_CODEX_CHILD" not in os.environ:
    env = {**os.environ, "FAKE_CODEX_CHILD": "1"}
    child = subprocess.run([sys.executable, *sys.orig_argv[1:]], env=env)
    sys.exit(child.returncode)

args = sys.argv[1:]
transcript = (base / "transcript.jsonl").open("a")


def log(**entry):
    transcript.write(json.dumps(entry) + "\n")
    transcript.flush()


def terminated(*_):
    log(signal="SIGTERM")
    os._exit(143)


signal.signal(signal.SIGTERM, signal.SIG_IGN if scenario.get("stubborn") else terminated)
home = Path(os.environ["CODEX_HOME"])
overrides = dict(value.split("=", 1) for flag, value in zip(args, args[1:]) if flag == "-c")
log(start={
    "pid": os.getpid(),
    "argv": args,
    "env": sorted(os.environ),
    "home": str(home),
    "home_entries": {path.name: os.readlink(path) for path in home.iterdir()},
    "cwd": os.getcwd(),
    "cwd_entries": os.listdir(),
    "catalog": json.loads(Path(json.loads(overrides["model_catalog_json"])).read_text()),
})
inputs = []


def emit(entry):
    if entry is None:
        log(exit="scripted")
        sys.exit(0)
    line = entry if isinstance(entry, str) else json.dumps(entry)
    if inputs:
        line = line.replace("$input", json.dumps(inputs[0])[1:-1])
    sys.stdout.buffer.write(line.encode() + b"\n")
    sys.stdout.buffer.flush()


def config():
    nested = {}
    for key, value in overrides.items():
        *parents, leaf = key.split(".")
        node = nested
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = json.loads(value)
    return nested | {
        "tools": {"web_search": None},
        "chatgpt_base_url": "https://chatgpt.com/backend-api/",
        "openai_base_url": None,
        "model_providers": {},
        "mcp_servers": {},
    } | scenario.get("config", {})


def page(items, params):
    half = len(items) // 2
    if params.get("cursor") is None:
        return {"data": items[:half], "nextCursor": "2"}
    return {"data": items[half:], "nextCursor": None}


def features():
    names = [value for flag, value in zip(args, args[1:]) if flag == "--disable"]
    enabled = scenario.get("enabled", [])
    listed = [{"name": name, "enabled": name in enabled} for name in names]
    return listed + [{"name": "unified_exec", "enabled": True}]


def answer(method, params):
    if method == "initialize":
        return {"userAgent": scenario.get("agent", AGENT), "codexHome": str(home)}
    if method == "account/read":
        if scenario.get("refresh"):
            (home / "auth.json").write_bytes(b'{"tokens": "new"}')
        return {"account": scenario.get("account", {"type": "chatgpt", "planType": "pro"})}
    if method == "config/read":
        return {"config": config()}
    if method == "experimentalFeature/list":
        return page(features(), params)
    if method == "thread/start":
        thread = {"id": "thread-1", "environments": scenario.get("environments", [])}
        model = scenario.get("model", params["model"])
        sources = scenario.get("sources", [])
        provider = {"modelProvider": "openai", "instructionSources": sources}
        return {"thread": thread, "model": model, **provider}
    if method == "mcpServerStatus/list":
        return page(scenario.get("servers", []), params)
    if method == "turn/start":
        inputs[:] = [part["text"] for part in params["input"]]
        return {"turn": {"id": "turn-1", "status": "inProgress", "error": None}}


def flood():
    count = 0
    while not select.select([0], [], [], 0)[0]:
        count += 1
        usage = {"total": {"inputTokens": count, "outputTokens": 1}}
        emit({"method": "thread/tokenUsage/updated", "params": {"tokenUsage": usage}})
        emit({"method": "item/agentMessage/delta", "params": {"delta": "x"}})


def lines():
    buffer = b""
    while chunk := os.read(0, 65536):
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield json.loads(line)


for message in lines():
    log(message=message)
    method = message.get("method")
    if "id" not in message:
        continue
    for entry in scenario.get("before", {}).get(method, []):
        emit(entry)
    reply = scenario.get("replies", {}).get(method) or {"result": answer(method, message["params"])}
    emit({"id": message["id"], **reply})
    if method == "account/read":
        emit({"method": "account/updated", "params": {"authMode": "chatgpt"}})
    if method == "turn/start":
        for entry in scenario.get("turn", []):
            emit(entry)
        if scenario.get("flood"):
            flood()
log(exit="eof")
while scenario.get("stubborn"):
    time.sleep(1)
"""


def said(text, phase="final_answer"):
    item = {"type": "agentMessage", "id": "m", "text": text, "phase": phase}
    return {"method": "item/completed", "params": {"item": item, "threadId": "thread-1"}}


def tokens(sent, received):
    total = {"inputTokens": sent, "cachedInputTokens": 0, "outputTokens": received}
    return {"method": "thread/tokenUsage/updated", "params": {"tokenUsage": {"total": total}}}


def ended(status="completed", info=None):
    error = None if info is None else {"message": "turn ended", "codexErrorInfo": info}
    turn = {"id": "turn-1", "status": status, "error": error}
    return {"method": "turn/completed", "params": {"threadId": "thread-1", "turn": turn}}


def unretried(info):
    error = {"message": "turn error", "codexErrorInfo": info}
    return {"method": "error", "params": {"error": error, "willRetry": False}}


def rpc_error(data=None):
    return {"error": {"code": -32603, "message": "turn refused", "data": data}}


def item(kind, method="item/started"):
    return {"method": method, "params": {"item": {"type": kind, "id": "i"}}}


TURN = [tokens(120, 30), said(REPLY), ended()]


@functools.cache
def app_server(directory):
    """Write the fake once per session, since macOS assesses each new executable on first run."""
    path = directory / "codex-app-server"
    path.write_text(f"#!{sys.executable} -IS\nAGENT = {AGENT!r}\n{APP_SERVER}")
    path.chmod(0o755)
    return path


def fake(tmp_path, *, turn_seconds=10.0, **scenario):
    """Link the fake app-server, write its scenario and a sign-in file, and return an adapter."""
    executable = tmp_path / "bin" / "codex"
    executable.parent.mkdir()
    # A hard link keeps argv[0], and with it the scenario, in this test's directory.
    os.link(app_server(tmp_path.parent), executable)
    (tmp_path / "scenario.json").write_text(json.dumps({"turn": TURN} | scenario))
    (tmp_path / "auth.json").write_bytes(SIGN_IN)
    return CodexAdapter(str(executable), tmp_path / "auth.json", turn_seconds)


def send(tmp_path, adapter, *texts, schema=SCORE, cancel=None):
    (tmp_path / "calls").mkdir(exist_ok=True)
    router = Gateway(CallLog(tmp_path / "calls"), [(policy(retains=True), adapter)])
    request = Request(tuple(Material(text, PUBLIC) for text in texts or (PROMPT,)), schema)
    return router.call(
        CONTEXT,
        PROPOSER,
        request,
        lease=Lease(uuid4(), (uuid4(),), 1),
        cancel=cancel or threading.Event(),
    )


def transcript(tmp_path):
    lines = (tmp_path / "transcript.jsonl").read_text().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]


def received(tmp_path):
    return [entry["message"] for entry in transcript(tmp_path) if "message" in entry]


def methods(tmp_path):
    return [message["method"] for message in received(tmp_path)]


def started(tmp_path):
    [start] = [entry["start"] for entry in transcript(tmp_path) if "start" in entry]
    return start


def gone(pid):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        # A PID 1 that never reaps leaves the killed child a zombie that still answers signal 0.
        stat = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        if not stat or stat.startswith("Z"):
            return True
        time.sleep(0.02)
    return False


def cancel_on(tmp_path, method):
    """Return an event that a watcher sets once the fake receives method, and when it did."""
    cancel, at = threading.Event(), []

    def watch():
        deadline = time.monotonic() + 10
        path = tmp_path / "transcript.jsonl"
        while not (path.exists() and method in methods(tmp_path)) and time.monotonic() < deadline:
            time.sleep(0.01)
        at.append(time.monotonic())
        cancel.set()

    threading.Thread(target=watch, daemon=True).start()
    return cancel, at


def sent_usage(sent=0, received=0):
    return Usage(model_calls=1, input_tokens=sent, output_tokens=received, unreported=TOKENS)


@pytest.mark.smoke
def test_a_structured_reply_passes_through_with_its_tokens_as_lower_bounds(tmp_path):
    answer = send(tmp_path, fake(tmp_path), PROMPT, "A second text")
    assert isinstance(answer, Answer) and answer.values == {"score": 0.5, "count": 2}
    assert (answer.record.outcome, answer.material.text) == ("completed", REPLY)
    assert answer.record.served == Served("codex 0.162.1", answer.record.model.model)
    assert answer.record.usage == sent_usage(120, 30)
    assert methods(tmp_path) == [
        "initialize",
        "initialized",
        "account/read",
        "config/read",
        "experimentalFeature/list",
        "experimentalFeature/list",
        "thread/start",
        "mcpServerStatus/list",
        "mcpServerStatus/list",
        "turn/start",
    ]
    turn = received(tmp_path)[-1]["params"]
    assert turn["input"] == [
        {"type": "text", "text": text, "text_elements": []} for text in (PROMPT, "A second text")
    ]
    assert turn["outputSchema"] == {
        "type": "object",
        "properties": {"score": {"type": "number"}, "count": {"type": "integer"}},
        "required": ["score", "count"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("turn", "text"),
    [
        pytest.param(
            [said("thinking", "commentary"), said("final"), said("plain", None), ended()],
            "final",
            id="final-among-others",
        ),
        pytest.param(
            [said("plain", None), said("thinking", "commentary"), ended()],
            "plain",
            id="phaseless-without-final",
        ),
        pytest.param(
            [said("first"), said("second"), ended()],
            "second",
            id="last-final",
        ),
    ],
)
def test_the_final_answer_wins_over_commentary_and_a_phaseless_message(tmp_path, turn, text):
    answer = send(tmp_path, fake(tmp_path, turn=turn), schema=None)
    assert isinstance(answer, Answer) and answer.material.text == text


def test_the_launch_isolates_codex_from_the_real_home(tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY", "CODEX_INTERNAL_X"):
        monkeypatch.setenv(name, "leak")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "real-home"))
    adapter = fake(tmp_path)
    assert isinstance(send(tmp_path, adapter), Answer)
    start = started(tmp_path)
    home = Path(start["home"])
    root = home.parent
    argv = start["argv"]
    assert argv[:3] == ["app-server", "--listen", "stdio://"]
    flags = list(zip(argv[3::2], argv[4::2], strict=True))
    catalog = f"model_catalog_json={json.dumps(str(root / 'catalog.json'))}"
    assert sorted(value for flag, value in flags if flag == "-c") == sorted([*OVERRIDES, catalog])
    assert sorted(value for flag, value in flags if flag == "--disable") == sorted(QUARRY_DISABLED)
    assert {flag for flag, _ in flags} == {"-c", "--disable"}
    leaked = [name for name in start["env"] if name.startswith(("OPENAI_", "CODEX_"))]
    assert leaked == ["CODEX_HOME"] and home != tmp_path / "real-home"
    assert start["home_entries"] == {"auth.json": str(adapter.auth)}
    assert (Path(start["cwd"]), start["cwd_entries"]) == (root / "work", [])
    [entry] = start["catalog"]["models"]
    assert entry["slug"] == "gpt-6.1-sol"
    assert (entry["shell_type"], entry["tool_mode"], entry["experimental_supported_tools"]) == (
        "disabled",
        "direct",
        [],
    )
    thread = next(m for m in received(tmp_path) if m["method"] == "thread/start")["params"]
    assert entry["base_instructions"] == thread["baseInstructions"]
    assert "decimal point" in entry["base_instructions"]
    assert not root.exists()
    assert adapter.auth.read_bytes() == SIGN_IN and not adapter.auth.is_symlink()


def test_a_finished_call_closes_stdin_and_the_server_exits_without_a_signal(tmp_path):
    assert isinstance(send(tmp_path, fake(tmp_path)), Answer)
    ends = [entry for entry in transcript(tmp_path) if "exit" in entry or "signal" in entry]
    assert ends == [{"exit": "eof"}]
    assert gone(started(tmp_path)["pid"])


def test_a_token_refresh_writes_through_to_the_sign_in_file(tmp_path):
    adapter = fake(tmp_path, refresh=True)
    assert isinstance(send(tmp_path, adapter), Answer)
    assert adapter.auth.read_bytes() == b'{"tokens": "new"}' and not adapter.auth.is_symlink()


def test_text_that_misses_the_schema_is_recorded_as_malformed_output_unchanged(tmp_path):
    reply = '{"score": 1, "count": 2}'
    result = send(tmp_path, fake(tmp_path, turn=[tokens(9, 4), said(reply), ended()]))
    assert isinstance(result, Unanswered)
    [record] = result.records
    assert (record.outcome, CallLog(tmp_path / "calls").reply(record), record.usage) == (
        "malformed_output",
        reply,
        sent_usage(9, 4),
    )


@pytest.mark.parametrize(
    ("scenario", "cause"),
    [
        pytest.param({"turn": [ended("failed", "other")]}, "provider_error", id="failed-other"),
        pytest.param(
            {"turn": [ended("failed", {"httpConnectionFailed": {"httpStatusCode": 502}})]},
            "http_connection_failed",
            id="failed-variant",
        ),
        pytest.param({"turn": [said("note", "commentary"), ended()]}, "no_answer", id="no-answer"),
        pytest.param({"turn": [None]}, "exit", id="exit-after-turn-start"),
        pytest.param({"turn": ["not json"]}, "protocol", id="raw-line"),
        pytest.param(
            {"replies": {"turn/start": rpc_error()}}, "provider_error", id="turn-start-error"
        ),
        pytest.param(
            {"turn": [said(REPLY), ended("interrupted")]}, "interrupted", id="interrupted"
        ),
        pytest.param(
            {"turn": [ended("failed", {"usageLimitExceeded": {}})]},
            "usage_limit",
            id="usage-limit-object-remains-failed",
        ),
        pytest.param(
            {"turn": [ended("failed", {"rateLimitExceeded": {}})]},
            "rate_limit",
            id="rate-limit-object-remains-failed",
        ),
    ],
)
def test_a_failed_run_is_a_provider_error(tmp_path, caplog, scenario, cause):
    result = send(tmp_path, fake(tmp_path, **scenario))
    assert isinstance(result, Unanswered)
    [record] = result.records
    assert (record.outcome, record.usage) == ("provider_error", sent_usage())
    assert record.reason == cause
    assert f"cause {cause}:" in caplog.text


SECRET = "provider-secret-sentinel"


@pytest.mark.parametrize(
    ("scenario", "reason", "diagnostic"),
    [
        ({"turn": [ended("failed", SECRET)]}, "provider_error", SECRET),
        (
            {"turn": [ended("failed", {SECRET: {"message": "private payload"}})]},
            "provider_error",
            SECRET,
        ),
        ({"turn": [ended(SECRET)]}, "provider_error", SECRET),
        ({"replies": {"turn/start": rpc_error(SECRET)}}, "provider_error", SECRET),
        (
            {"turn": [ended("failed", {"httpConnectionFailed": {"message": SECRET}})]},
            "http_connection_failed",
            "httpConnectionFailed",
        ),
        ({"turn": [ended("failed", [SECRET])]}, "provider_error", "unknown codexErrorInfo"),
        ({"servers": [{"name": SECRET, "tools": {}}]}, "inventory", SECRET),
        ({"turn": [item(SECRET)]}, "tools", SECRET),
    ],
)
def test_failure_reasons_survive_reopen_without_provider_text(
    tmp_path, caplog, scenario, reason, diagnostic
):
    result = send(tmp_path, fake(tmp_path, **scenario))
    [record] = result.records
    assert record.reason == reason and record.reasons == ()
    assert diagnostic in caplog.text
    assert "private payload" not in caplog.text
    if diagnostic != SECRET:
        assert SECRET not in caplog.text
    assert CallLog(tmp_path / "calls").load()[-1] == record
    wire = dump_result(result)
    assert load_result(wire, Request((Material(PROMPT, PUBLIC),), SCORE)) == result
    stored = b"".join(path.read_bytes() for path in (tmp_path / "calls").iterdir())
    stored += b"".join(path.read_bytes() for path in (tmp_path / "calls.transcripts").iterdir())
    assert SECRET not in wire and SECRET.encode() not in stored


def test_usage_reported_before_a_turn_start_error_is_kept(tmp_path, caplog):
    scenario = {"before": {"turn/start": [tokens(50, 3)]}, "replies": {"turn/start": rpc_error()}}
    result = send(tmp_path, fake(tmp_path, **scenario))
    [record] = result.records
    assert (record.outcome, record.usage) == ("provider_error", sent_usage(50, 3))
    assert record.reason == "provider_error"
    assert "cause provider_error: turn/start: turn refused" in caplog.text


def test_a_rerouted_model_fails_the_call_even_with_an_answer(tmp_path, caplog):
    rerouted = {
        "method": "model/rerouted",
        "params": {
            "fromModel": "gpt-6.1-sol",
            "toModel": "gpt-5",
            "reason": "highRiskCyberActivity",
        },
    }
    result = send(tmp_path, fake(tmp_path, turn=[tokens(7, 0), rerouted, said(REPLY), ended()]))
    [record] = result.records
    assert (record.outcome, record.usage) == ("provider_error", sent_usage(7, 0))
    assert record.reason == "rerouted"
    assert "cause rerouted:" in caplog.text and '"toModel": "gpt-5"' in caplog.text
    assert gone(started(tmp_path)["pid"])


@pytest.mark.parametrize(
    ("scenario", "limit"),
    [
        pytest.param({"turn": [tokens(11, 0)]}, 2.0, id="mid-turn"),
        pytest.param({"turn": [tokens(11, 0)], "stubborn": True}, 4.0, id="mid-turn-stubborn"),
        pytest.param(None, 1.0, id="preset"),
    ],
)
def test_cancellation_stops_the_process_group(tmp_path, caplog, scenario, limit):
    caplog.set_level(logging.INFO)
    adapter = fake(tmp_path, **(scenario or {}))
    if scenario is None:
        cancel, at = threading.Event(), [time.monotonic()]
        cancel.set()
    else:
        cancel, at = cancel_on(tmp_path, "turn/start")
    result = send(tmp_path, adapter, cancel=cancel)
    assert time.monotonic() - at[0] < limit
    [record] = result.records
    assert record.outcome == "cancelled" and "cause cancelled:" in caplog.text
    assert record.reason == "cancelled"
    if scenario is None:
        assert record.usage == Usage(model_calls=1)
        assert not (tmp_path / "transcript.jsonl").exists()
        return
    assert record.usage == sent_usage(11, 0)
    assert gone(started(tmp_path)["pid"])
    assert not Path(started(tmp_path)["home"]).parent.exists()
    signals = [entry for entry in transcript(tmp_path) if "signal" in entry]
    assert signals == []


def test_a_turn_past_its_deadline_fails_and_stops_the_process(tmp_path, caplog):
    began = time.monotonic()
    result = send(tmp_path, fake(tmp_path, turn=[tokens(5, 1)], turn_seconds=0.3))
    assert time.monotonic() - began < 2
    [record] = result.records
    assert (record.outcome, record.usage) == ("provider_error", sent_usage(5, 1))
    assert record.reason == "deadline"
    assert "cause deadline:" in caplog.text
    assert gone(started(tmp_path)["pid"])


@pytest.mark.parametrize("end", ["cancel", "deadline"])
def test_a_turn_that_never_stops_reporting_still_ends_on_cancel_or_deadline(tmp_path, caplog, end):
    caplog.set_level(logging.INFO)
    adapter = fake(tmp_path, turn=[], flood=True, turn_seconds=10.0 if end == "cancel" else 0.3)
    cancel, at = cancel_on(tmp_path, "turn/start") if end == "cancel" else (None, [])
    began = time.monotonic()
    result = send(tmp_path, adapter, cancel=cancel)
    assert time.monotonic() - (at[0] if at else began) < 1.5
    [record] = result.records
    assert record.outcome == ("cancelled" if end == "cancel" else "provider_error")
    assert record.reason == ("cancelled" if end == "cancel" else "deadline")
    assert f"cause {record.reason}:" in caplog.text
    assert record.usage.input_tokens > 0 and record.usage.unreported == TOKENS
    assert gone(started(tmp_path)["pid"])


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        pytest.param({"turn": [ended("failed", "usageLimitExceeded")]}, "usage_limit", id="turn"),
        pytest.param(
            {"turn": [unretried("usageLimitExceeded"), ended("failed")]},
            "usage_limit",
            id="unretried-error-then-failed",
        ),
        pytest.param(
            {"replies": {"turn/start": rpc_error("usageLimitExceeded")}},
            "usage_limit",
            id="rpc-data",
        ),
        pytest.param(
            {"turn": [ended("failed", "rateLimitExceeded")]}, "rate_limit", id="rate-limit"
        ),
    ],
)
def test_a_usage_cap_is_rate_limited_wherever_it_arrives(tmp_path, scenario, reason):
    result = send(tmp_path, fake(tmp_path, **scenario))
    [record] = result.records
    assert (record.outcome, record.usage) == ("rate_limited", sent_usage())
    assert record.reason == reason


@pytest.mark.parametrize(
    ("scenario", "threaded"),
    [
        pytest.param({"account": None}, False, id="signed-out"),
        pytest.param({"account": {"type": "apiKey"}}, False, id="api-key"),
        pytest.param({"turn": [ended("failed", "unauthorized")]}, True, id="turn-unauthorized"),
    ],
)
def test_an_expired_sign_in_fails(tmp_path, caplog, scenario, threaded):
    result = send(tmp_path, fake(tmp_path, **scenario))
    [record] = result.records
    assert record.outcome == "provider_error" and "cause signed_out:" in caplog.text
    assert record.reason == "signed_out"
    assert ("thread/start" in methods(tmp_path)) == threaded


@pytest.mark.parametrize(
    ("scenario", "cause"),
    [
        pytest.param({"servers": [{"name": "codex_apps", "tools": {}}]}, "inventory", id="mcp"),
        pytest.param({"enabled": ["apps"]}, "features", id="feature-on"),
        pytest.param({"config": {"web_search": "live"}}, "config", id="web-search"),
        pytest.param({"config": {"openai_base_url": "http://x.test"}}, "config", id="base-url"),
        pytest.param(
            {"config": {"chatgpt_base_url": "https://x.test/backend-api"}},
            "config",
            id="chatgpt-url",
        ),
        pytest.param({"config": {"cli_auth_credentials_store": "keyring"}}, "config", id="keyring"),
        pytest.param({"config": {"project_doc_max_bytes": False}}, "config", id="false-for-zero"),
        pytest.param({"environments": [{"id": "local"}]}, "thread", id="environment"),
        pytest.param({"sources": ["/home/AGENTS.md"]}, "thread", id="instruction-source"),
        pytest.param({"model": "gpt-5"}, "thread", id="another-model"),
        pytest.param({"agent": AGENT.replace("0.162.1", "0.163.0")}, "version", id="version"),
    ],
)
def test_a_nonempty_inventory_refuses_before_any_material_is_sent(
    tmp_path, caplog, scenario, cause
):
    result = send(tmp_path, fake(tmp_path, **scenario))
    [record] = result.records
    assert (record.outcome, record.usage) == ("provider_error", Usage(model_calls=1))
    assert record.reason == cause
    assert f"cause {cause}:" in caplog.text
    assert "turn/start" not in methods(tmp_path)
    assert PROMPT not in json.dumps(received(tmp_path))


@pytest.mark.parametrize(
    ("event", "cause"),
    [
        pytest.param(item("commandExecution"), "tools", id="command-item"),
        pytest.param(item("webSearch", "item/completed"), "tools", id="web-search-item"),
        pytest.param(
            {"id": 99, "method": "item/tool/call", "params": {}}, "server_request", id="tool-call"
        ),
    ],
)
def test_a_tool_in_the_turn_ends_the_call(tmp_path, caplog, event, cause):
    result = send(tmp_path, fake(tmp_path, turn=[item("reasoning"), event, said(REPLY), ended()]))
    [record] = result.records
    assert record.outcome == "provider_error" and f"cause {cause}:" in caplog.text
    assert gone(started(tmp_path)["pid"])


def test_a_turn_without_a_usage_update_counts_zero_tokens_as_unreported(tmp_path):
    answer = send(tmp_path, fake(tmp_path, turn=[said(REPLY), ended()]))
    assert isinstance(answer, Answer) and answer.record.usage == sent_usage()


def test_a_missing_executable_fails_without_raising(tmp_path, caplog):
    adapter = CodexAdapter(str(tmp_path / "none"), tmp_path / "auth.json")
    [record] = send(tmp_path, adapter).records
    assert (record.outcome, record.usage) == ("provider_error", Usage(model_calls=1))
    assert "cause spawn:" in caplog.text


def test_relative_executable_and_sign_in_paths_resolve_from_the_callers_directory(
    tmp_path, monkeypatch
):
    fake(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert isinstance(send(tmp_path, CodexAdapter("bin/codex", Path("auth.json"))), Answer)
    link = Path(started(tmp_path)["home_entries"]["auth.json"])
    assert link.is_absolute() and link.samefile(tmp_path / "auth.json")


def test_concurrent_calls_on_one_adapter_keep_separate_homes(tmp_path):
    adapter = fake(tmp_path, turn=[said("$input"), ended()])
    texts = ("First prompt", "Second prompt")
    answers = {}

    def ask(text):
        answers[text] = send(tmp_path, adapter, text, schema=None)

    threads = [threading.Thread(target=ask, args=(text,)) for text in texts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert {text: answer.material.text for text, answer in answers.items()} == {t: t for t in texts}
    homes = {entry["start"]["home"] for entry in transcript(tmp_path) if "start" in entry}
    assert len(homes) == 2
