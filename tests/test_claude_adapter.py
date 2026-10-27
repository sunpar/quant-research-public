from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from quant_research.access import INTERNAL_CODE, Provider, Restrictions, issue_context
from quant_research.accounting import Usage
from quant_research.claude_adapter import POLICY, PROVIDER, VERSION, ClaudeAdapter
from quant_research.model_provenance import UNKNOWN, ModelRecord, RoleAssignment
from quant_research.providers import (
    Answer,
    CallLog,
    Gateway,
    Lease,
    Material,
    ProviderPolicy,
    Request,
    ScriptedProvider,
    Served,
    dump_result,
    load_result,
)

CUTOFF = datetime(2026, 10, 21, 12, tzinfo=UTC)
FALLBACK = Provider("fake-b")
MODEL = ModelRecord("claude-subscription", "claude-opus-5-5", UNKNOWN, UNKNOWN)
FALLBACK_MODEL = ModelRecord("fake-b", "model-b", UNKNOWN, UNKNOWN)
CONTEXT = issue_context(
    clearances=(INTERNAL_CODE,),
    cutoff=CUTOFF,
    generation="synthetic/v1",
    providers=(PROVIDER, FALLBACK),
)
PUBLIC = Restrictions(
    available_at=CUTOFF - timedelta(days=1),
    labels=frozenset(),
    approved_for=frozenset({PROVIDER, FALLBACK}),
)
PROMPT = "Propose a momentum signal"
PIPE_FILLER = "Use daily bars. " * 50_000
SCORE = (("score", float), ("count", int))
TOKENS = frozenset({"input_tokens", "output_tokens"})
NOT_SENT = Usage(model_calls=1)
SENT = Usage(model_calls=1, unreported=TOKENS)
REPORTED = Usage(model_calls=1, input_tokens=543, output_tokens=7, unreported=TOKENS)
SIGNED_IN = {
    "loggedIn": True,
    "authMethod": "claude.ai",
    "apiProvider": "firstParty",
    "subscriptionType": "max",
}
SESSION = "c159e190-c7a7-4021-b046-e15da59de798"
HOOKS = {
    "type": "control_request",
    "request_id": "hooks",
    "request": {"subtype": "get_hooks_listing"},
}
COST = {"type": "user", "message": {"role": "user", "content": "/cost"}}
UNAVAILABLE = "`/cost` isn't available in this environment."
EXPIRED = "Failed to authenticate: OAuth session expired and could not be refreshed"
HANG, EXIT = "<hang>", "<exit>"
ZERO = {
    "input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 0,
}
USAGE = {
    "input_tokens": 3,
    "cache_creation_input_tokens": 40,
    "cache_read_input_tokens": 500,
    "output_tokens": 7,
    "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
    "service_tier": "standard",
}
BUILTINS = [
    {"name": name, "path": "builtin", "source": f"{name}@builtin"}
    for name in ("cc-plugin-agents-md", "cc-plugin-plugin-authoring")
]

FAKE = r"""
import json, os, subprocess, sys, time
from pathlib import Path

here = Path(sys.argv[0]).parent
scenario = json.loads((here / "scenario.json").read_text())


def record(**entry):
    with (here / "log.jsonl").open("a") as log:
        log.write(json.dumps(entry) + "\n")


# The interpreter adds these two to the environment it was given.
env = {k: v for k, v in os.environ.items() if k not in ("LC_CTYPE", "__CF_USER_TEXT_ENCODING")}
record(argv=sys.argv[1:], directory=os.getcwd(), listing=os.listdir(), env=env)
if sys.argv[1:] == ["auth", "status"]:
    print(json.dumps(scenario["status"], indent=2))
    sys.exit(scenario["code"])
if scenario["grandchild"]:
    sleeper = subprocess.Popen(
        ["/bin/sleep", "30"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    record(grandchild=sleeper.pid)
turns = iter(scenario["turns"])
while line := sys.stdin.readline():
    record(stdin=line)
    for out in next(turns, []):
        if out == "<hang>":
            time.sleep(30)
        if out == "<exit>":
            sys.exit(1)
        print(out, flush=True)
"""


@pytest.fixture(scope="session")
def script(tmp_path_factory):
    # macOS checks each new executable on its first run, which costs more than the run itself,
    # so every test links to this one script.
    path = tmp_path_factory.mktemp("claude") / "claude"
    path.write_text(f"#!{sys.executable} -IS\n{FAKE}")
    path.chmod(0o755)
    return path


@pytest.fixture
def claude(tmp_path, script):
    def write(*turns, hooks=(LISTING,), status=SIGNED_IN, code=0, grandchild=False):
        directory = tmp_path / "bin"
        directory.mkdir()
        lines = [
            [out if isinstance(out, str) else json.dumps(out) for out in turn]
            for turn in (hooks, *turns)
        ]
        scenario = {"status": status, "code": code, "grandchild": grandchild, "turns": lines}
        (directory / "scenario.json").write_text(json.dumps(scenario))
        (directory / "claude").symlink_to(script)
        return directory / "claude"

    return write


def seen(fake):
    log = fake.parent / "log.jsonl"
    entries = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return SimpleNamespace(
        runs=[entry for entry in entries if "argv" in entry],
        stdin=[json.loads(entry["stdin"]) for entry in entries if "stdin" in entry],
        grandchild=next((entry["grandchild"] for entry in entries if "grandchild" in entry), None),
    )


def listing(*hooks):
    response = {
        "subtype": "success",
        "request_id": "hooks",
        "response": {"events": [], "hooks": hooks},
    }
    return {"type": "control_response", "response": response}


LISTING = listing()


def init(**changes):
    return {
        "type": "system",
        "subtype": "init",
        "cwd": "/private/tmp/session",
        "session_id": SESSION,
        "tools": [],
        "mcp_servers": [],
        "model": MODEL.model,
        "permissionMode": "default",
        "slash_commands": [],
        "apiKeySource": "none",
        "claude_code_version": "2.1.301",
        "output_style": "default",
        "agents": ["general-purpose"],
        "skills": [],
        "plugins": BUILTINS,
    } | changes


def assistant(text, model=MODEL.model, **fields):
    message = {"model": model, "role": "assistant", "content": [{"type": "text", "text": text}]}
    return {"type": "assistant", "message": message, "session_id": SESSION} | fields


def result(text, **fields):
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "result": text,
        "session_id": SESSION,
        "usage": USAGE,
    } | fields


def answered_locally(text, **changes):
    return [
        init(**changes),
        assistant(text, "<synthetic>"),
        result(text, num_turns=0, usage=ZERO),
    ]


def gate(**changes):
    return [
        {"type": "system", "subtype": "ui_invalidate"},
        *answered_locally(UNAVAILABLE, **changes),
    ]


GATE = gate()


def answered(text, **fields):
    limits = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}}
    return [init(), limits, assistant(text), result(text, **fields)]


def refused(text, error):
    return [
        init(),
        assistant(text, "<synthetic>", error=error, is_api_error_message=True),
        result(text, is_error=True, terminal_reason="api_error", usage=ZERO),
    ]


def prompt(*texts):
    content = [{"type": "text", "text": text} for text in texts]
    return {"type": "user", "message": {"role": "user", "content": content}}


def material(text):
    return Material(text, PUBLIC)


def send(tmp_path, fake, *materials, timeout=10.0, schema=None, cancel=None, fallback=None):
    (tmp_path / "calls").mkdir()
    routes = [(POLICY, ClaudeAdapter(fake, timeout=timeout))]
    if fallback is not None:
        routes.append(
            (ProviderPolicy(provider=FALLBACK, labels=frozenset(), retains=False), fallback)
        )
    router = Gateway(CallLog(tmp_path / "calls"), routes)
    assignment = RoleAssignment("proposer", MODEL, () if fallback is None else (FALLBACK_MODEL,))
    answer = router.call(
        CONTEXT,
        assignment,
        Request(materials, schema),
        lease=Lease(uuid4(), (uuid4(), uuid4()), 1),
        cancel=cancel or threading.Event(),
    )
    return answer, router.records


def alive(pid):
    # A container whose init never reaps orphans keeps the killed grandchild as a zombie.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return state.stdout.strip()[:1] not in ("", "Z")


def until(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_a_reply_after_the_gate_turn_completes_with_its_tokens_as_lower_bounds(tmp_path, claude):
    fake = claude(GATE, answered("A momentum signal"))
    answer, [record] = send(tmp_path, fake, material(PROMPT), material(PIPE_FILLER))
    assert isinstance(answer, Answer)
    assert (record.outcome, CallLog(tmp_path / "calls").reply(record), record.usage) == (
        "completed",
        "A momentum signal",
        REPORTED,
    )
    assert record.served == Served(f"claude {VERSION}", record.model.model)
    assert seen(fake).stdin == [HOOKS, COST, prompt(PROMPT, PIPE_FILLER)]


@pytest.mark.parametrize(
    ("text", "outcome", "values"),
    [
        pytest.param(
            '{"score": 0.75, "count": 20}', "completed", {"score": 0.75, "count": 20}, id="exact"
        ),
        pytest.param(
            'Here it is: {"score": 0.75, "count": 20}', "malformed_output", None, id="prose"
        ),
    ],
)
def test_a_schema_reply_is_asked_for_in_the_system_prompt_and_judged_by_the_gateway(
    tmp_path, claude, text, outcome, values
):
    fake = claude(GATE, answered(text))
    answer, [record] = send(tmp_path, fake, material(PROMPT), schema=SCORE)
    assert (record.outcome, CallLog(tmp_path / "calls").reply(record), record.usage) == (
        outcome,
        text,
        REPORTED,
    )
    assert (answer.values if isinstance(answer, Answer) else None) == values
    argv = seen(fake).runs[1]["argv"]
    instruction = argv[argv.index("--system-prompt") + 1]
    assert "--json-schema" not in argv
    assert 0 <= instruction.index('"score"') < instruction.index('"count"')


@pytest.mark.parametrize(
    ("turn", "usage"),
    [
        pytest.param(
            answered("API Error: 500", is_error=True, terminal_reason="api_error"),
            REPORTED,
            id="api-error",
        ),
        pytest.param([init(), EXIT], SENT, id="exit-before-result"),
        pytest.param(refused(EXPIRED, "authentication_failed"), SENT, id="signed-out-mid-turn"),
        pytest.param(
            answered_locally("`/model` isn't available in this environment."),
            SENT,
            id="answered-locally",
        ),
        pytest.param([init(), "not json", *answered("unused")[1:]], SENT, id="not-json"),
        pytest.param(
            [init(), '{"type": "assistant", "type": "result"}', *answered("unused")[1:]],
            SENT,
            id="duplicate-key",
        ),
        pytest.param([init(), "[]", *answered("unused")[1:]], SENT, id="not-an-object"),
        pytest.param(
            [*answered("unused")[:-1], {"type": "result", "is_error": False, "usage": USAGE}],
            REPORTED,
            id="result-missing-fields",
        ),
        pytest.param([init(tools=["Bash"]), *answered("unused")[1:]], SENT, id="inventory-changed"),
        *(
            pytest.param(
                [init(), {"type": "system", "subtype": subtype}, *answered("unused")[1:]],
                SENT,
                id=subtype,
            )
            for subtype in ("model_fallback", "model_refusal_fallback", "model_consent_fallback")
        ),
    ],
)
def test_a_failed_real_turn_is_a_provider_error(tmp_path, claude, turn, usage):
    fake = claude(GATE, turn)
    _, [record] = send(tmp_path, fake, material(PROMPT))
    assert (record.outcome, record.usage) == ("provider_error", usage)
    assert seen(fake).stdin == [HOOKS, COST, prompt(PROMPT)]


SECRET = "provider-secret-sentinel"


@pytest.mark.parametrize(
    ("turn", "reason"),
    [
        (refused(SECRET, SECRET), "provider_error"),
        (answered(SECRET, is_error=True, terminal_reason=SECRET), "provider_error"),
        (answered(SECRET, is_error=True, subtype=SECRET), "provider_error"),
        (refused(SECRET, "authentication_failed"), "signed_out"),
        (answered(SECRET, is_error=True, terminal_reason="api_error"), "api_error"),
        ([init(tools=[SECRET])], "tools"),
        ([init(), SECRET], "protocol"),
    ],
)
def test_failure_reasons_survive_reopen_without_provider_text(tmp_path, claude, turn, reason):
    result, [record] = send(tmp_path, claude(GATE, turn), material(PROMPT))
    assert record.reason == reason and record.reasons == ()
    assert CallLog(tmp_path / "calls").load()[-1] == record
    wire = dump_result(result)
    assert load_result(wire, Request((material(PROMPT),))) == result
    stored = b"".join(path.read_bytes() for path in (tmp_path / "calls").iterdir())
    stored += b"".join(path.read_bytes() for path in (tmp_path / "calls.transcripts").iterdir())
    assert SECRET not in wire and SECRET.encode() not in stored


def test_a_usage_cap_is_rate_limited_and_the_next_route_runs(tmp_path, claude):
    fake = claude(GATE, refused("You've hit your limit", "rate_limit"))
    answer, records = send(
        tmp_path, fake, material(PROMPT), fallback=ScriptedProvider(["Looks fine"])
    )
    assert isinstance(answer, Answer) and answer.material.text == "Looks fine"
    assert [(r.model, r.outcome) for r in records] == [
        (MODEL, "rate_limited"),
        (FALLBACK_MODEL, "completed"),
    ]
    assert records[0].usage == SENT
    assert records[0].reason == "rate_limit" and records[1].reason is None


@pytest.mark.parametrize(
    ("status", "code"),
    [
        pytest.param(
            {"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"}, 1, id="expired"
        ),
        pytest.param(SIGNED_IN | {"apiProvider": "bedrock"}, 0, id="third-party-api"),
    ],
)
def test_without_a_subscription_sign_in_no_session_starts(tmp_path, claude, status, code):
    fake = claude(GATE, answered("unused"), status=status, code=code)
    _, [record] = send(tmp_path, fake, material(PROMPT))
    assert (record.outcome, record.usage) == ("provider_error", NOT_SENT)
    assert record.reason == "signed_out"
    assert [run["argv"] for run in seen(fake).runs] == [["auth", "status"]]


@pytest.mark.parametrize(
    "turn",
    [
        pytest.param(
            gate(mcp_servers=[{"name": "probe", "status": "failed", "source": "user"}]),
            id="mcp-server",
        ),
        pytest.param(gate(tools=["Bash"]), id="tool"),
        pytest.param(gate(skills=["deep-research"]), id="skill"),
        pytest.param(
            gate(plugins=[*BUILTINS, {"name": "extra", "path": "/plugins", "source": "extra@u"}]),
            id="plugin",
        ),
        pytest.param(
            gate(
                plugins=[
                    {
                        "name": "cc-plugin-agents-md",
                        "path": "/Users/me/.claude/plugins/cache/user-market/cc-plugin-agents-md",
                        "source": "cc-plugin-agents-md@user-market",
                    },
                    BUILTINS[1],
                ]
            ),
            id="plugin-origin",
        ),
        pytest.param(gate(claude_code_version="2.1.302"), id="version"),
        pytest.param(gate(model="claude-haiku-5-5"), id="model"),
        pytest.param(gate(apiKeySource="ANTHROPIC_API_KEY"), id="api-key-source"),
        pytest.param(gate(output_style="Learning"), id="output-style"),
        pytest.param([*GATE[:-1], result(UNAVAILABLE, num_turns=1, usage=ZERO)], id="gate-turn"),
        pytest.param([GATE[0], *GATE[2:]], id="result-before-init"),
    ],
)
def test_an_unexpected_session_is_refused_before_the_request_is_written(tmp_path, claude, turn):
    fake = claude(turn, answered("unused"))
    _, [record] = send(tmp_path, fake, material(PROMPT))
    assert (record.outcome, record.usage) == ("provider_error", NOT_SENT)
    assert seen(fake).stdin == [HOOKS, COST]


@pytest.mark.parametrize(
    "hooks",
    [
        pytest.param(
            [listing({"event": "UserPromptSubmit", "matcher": "", "source": "userSettings"})],
            id="hook",
        ),
        pytest.param(
            [{"type": "control_response", "response": {"subtype": "error", "request_id": "hooks"}}],
            id="unsuccessful",
        ),
        pytest.param([EXIT], id="exit"),
    ],
)
def test_a_session_with_hooks_is_refused_before_the_gate_turn(tmp_path, claude, hooks):
    fake = claude(GATE, answered("unused"), hooks=hooks)
    _, [record] = send(tmp_path, fake, material(PROMPT))
    assert (record.outcome, record.usage) == ("provider_error", NOT_SENT)
    assert seen(fake).stdin == [HOOKS]


def test_cancelling_during_the_real_turn_kills_the_process_group(tmp_path, claude):
    fake = claude(GATE, [HANG], grandchild=True)
    cancel = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        call = pool.submit(send, tmp_path, fake, material(PROMPT), timeout=30, cancel=cancel)
        assert until(lambda: len(seen(fake).stdin) == 3)
        grandchild = seen(fake).grandchild
        assert alive(grandchild)
        cancel.set()
        cancelled = time.monotonic()
        _, [record] = call.result(timeout=10)
    assert time.monotonic() - cancelled < 1
    assert (record.outcome, record.usage) == ("cancelled", SENT)
    assert record.reason == "cancelled"
    assert until(lambda: not alive(grandchild))


def test_a_call_cancelled_before_it_starts_runs_no_process(tmp_path, claude):
    fake = claude(GATE, answered("unused"))
    cancel = threading.Event()
    cancel.set()
    _, [record] = send(tmp_path, fake, material(PROMPT), cancel=cancel)
    assert (record.outcome, record.usage) == ("cancelled", NOT_SENT)
    assert record.reason == "cancelled"
    assert seen(fake).runs == []


def test_the_deadline_fails_the_call_and_kills_the_process_group(tmp_path, claude):
    fake = claude(GATE, [HANG], grandchild=True)
    started = time.monotonic()
    _, [record] = send(tmp_path, fake, material(PROMPT), timeout=1.0)
    assert 1.0 <= time.monotonic() - started < 2.0
    assert (record.outcome, record.usage) == ("provider_error", SENT)
    assert record.reason == "deadline"
    assert seen(fake).stdin == [HOOKS, COST, prompt(PROMPT)]
    assert until(lambda: not alive(seen(fake).grandchild))


def test_the_cli_sees_only_allowlisted_variables_an_empty_directory_and_fixed_flags(
    tmp_path, claude, monkeypatch
):
    for name in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_USE_BEDROCK",
        "AWS_PROFILE",
        "CLAUDE_CODE_SIMPLE",
    ):
        monkeypatch.setenv(name, "1")
    fake = claude(GATE, answered("Looks fine"))
    _, [record] = send(tmp_path, fake, material(PROMPT))
    assert record.outcome == "completed"
    inherited = ("HOME", "PATH", "USER", "LOGNAME", "SHELL", "TMPDIR")
    environment = {name: os.environ[name] for name in inherited if name in os.environ} | {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_MAX_RETRIES": "2",
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
        "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
        "CLAUDE_CODE_DISABLE_FILE_MENTIONS": "1",
    }
    auth, session = seen(fake).runs
    assert [run["env"] for run in (auth, session)] == [environment, environment]
    assert [run["listing"] for run in (auth, session)] == [[], []]
    assert auth["directory"] == session["directory"] and not Path(auth["directory"]).exists()
    assert auth["argv"] == ["auth", "status"]
    assert session["argv"][:-1] == [
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--tools",
        "",
        "--safe-mode",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--model",
        "claude-opus-5-5",
        "--system-prompt",
    ]
    assert PROMPT not in session["argv"][-1]


def test_the_route_takes_public_material_only_and_is_recorded_as_retaining_it(tmp_path, claude):
    fake = claude(GATE, answered("unused"))
    internal = Material(PROMPT, replace(PUBLIC, labels=frozenset({INTERNAL_CODE})))
    _, [record] = send(tmp_path, fake, internal)
    assert (record.outcome, record.reasons) == ("refused", ("outside_provider_policy",))
    assert seen(fake).runs == []
    assert (POLICY.provider, POLICY.retains) == (PROVIDER, True)
