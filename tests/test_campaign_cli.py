import asyncio
import json
import subprocess
import sys
import threading
from dataclasses import fields, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode

from quant_research import reference, research
from quant_research.__main__ import LedgerReceipts, ledger_receipts, live_receipts
from quant_research.access import Provider, Restrictions, issue_context, licensed
from quant_research.accounting import Budget, Ledger, Usage, Work
from quant_research.campaigns import (
    PROMPT,
    CampaignRevision,
    CampaignSpecification,
    EvaluationRules,
    ModelRole,
    ResourceLimits,
    Setup,
    file_bytes,
    register,
    resolve,
    revise,
    save_revision,
    start_campaign,
)
from quant_research.catalog import Catalog
from quant_research.findings import Ledger as Findings
from quant_research.model_provenance import UNKNOWN, ModelRecord, RoleAssignment
from quant_research.providers import (
    CallLog,
    CallRecord,
    Closed,
    Dispatched,
    Fault,
    Gateway,
    Lease,
    Material,
    ProviderPolicy,
    Receipt,
    Request,
    ScriptedProvider,
)
from quant_research.records import canonical_bytes
from quant_research.sources import SourceDeclaration

TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
CUTOFF = datetime(2025, 6, 1, tzinfo=UTC)
WRITTEN = datetime(2025, 1, 1, tzinfo=UTC)
CODEX = Provider("codex-subscription")
POLICY = ProviderPolicy(provider=CODEX, labels=frozenset(), retains=True)
CAMPAIGN = "09581450-9da9-49e3-8055-65c8873da192"
# Inside the pilot's floor: before its cutoff and approved for its one provider.
DECLARED = ("--available-at", "2025-01-01T00:00:00Z", "--approved-for", CODEX)
LOADED = ("temporalio", "surrealdb", "quant_research.research", "quant_research.store")
A, B = Provider("fake-a"), Provider("fake-b")
VENDOR = licensed("vendor")
WIDE = ProviderPolicy(provider=A, labels=frozenset({VENDOR}), retains=False)
NARROW = ProviderPolicy(provider=B, labels=frozenset(), retains=True)
MODEL_A = ModelRecord("fake-a", "model-a", "2026-05-01", date(2026, 1, 31))
CHAIN = RoleAssignment("researcher", MODEL_A, (ModelRecord("fake-b", "model-b", UNKNOWN, UNKNOWN),))
CONTEXT = issue_context(clearances=(VENDOR,), cutoff=CUTOFF, generation="g", providers=(A, B))
ANSWER = '{"action": "next"}'
NAMED = ("cost_unreported", "differs", "uncharged", "unrecorded", "unsealed")
STOP = json.dumps({"action": "stop", "arguments": "{}", "rationale": "Nothing more to learn"})


def window(start: str, end: str, cutoff: str) -> dict:
    """Return one window of a rules file, in reference.json's schema."""
    evaluation = {"alpha": "0.05", "minimum_effect": "0.001", "subperiods": 2}
    return {
        "baseline": "buy_and_hold_market",
        "cutoff": cutoff,
        "evaluation": {**evaluation, "version": "pilot-evaluation/v1"},
        "schema_version": 1,
        "window": {"end": end, "start": start},
    }


# Both windows end before CUTOFF, so the campaign resolves under it.
RULES = reference.load_rules(
    canonical_bytes(
        {
            "held_back": window("2020-01-01", "2022-06-01", "2022-07-01T00:00:00Z"),
            "schema_version": 1,
            "search": window("2017-07-01", "2019-12-01", "2020-01-01T00:00:00Z"),
        }
    )
)


@pytest.fixture
def setup(tmp_path) -> Setup:
    sources = tmp_path / "sources"
    sources.mkdir()
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / f"{RULES.digest}.json").write_bytes(RULES.data)
    return Setup(
        sources=sources,
        datasets=tmp_path / "datasets",
        catalog=Catalog(()),
        ledger=Findings(),
        clearances=frozenset(),
        cutoff=CUTOFF,
        policies=(POLICY,),
        rules=rules,
        packages=tmp_path / "packages",
        lock=b"synthetic lock\n",
    )


def pilot(setup: Setup) -> CampaignRevision:
    """Start a campaign whose one source is its prompt and that holds every conclusion."""
    prompt = register(
        setup.sources,
        name=research.ROLE,
        role=PROMPT,
        data=b"You are the researcher. Propose one step at a time.\n",
        declaration=SourceDeclaration(
            media_type="text/markdown",
            access_domain="repository",
            location="prompts/researcher.md",
            revision="1",
            acquisition="hand-authored test fixture",
            rights="synthetic, unrestricted",
            labels=frozenset(),
            approved_for=frozenset({CODEX}),
            published_at=None,
            source_available_at=WRITTEN,
            availability_evidence="authored for these tests",
            received_at=WRITTEN,
        ),
    )
    specification = CampaignSpecification(
        "Does the real-time Sahm rule time the market?",
        (prompt,),
        (),
        EvaluationRules("pilot-evaluation/v1", RULES.digest),
        ResourceLimits(1, 2, 100, max_model_calls=6),
        ("checkpoint before conclude",),
        (ModelRole(research.ROLE, "codex-subscription/gpt-6.1-sol", (), "active"),),
    )
    return start_campaign(specification, author="researcher", at=TIME, reason="Pilot")


def steered(parent: CampaignRevision) -> CampaignRevision:
    question = "Does the rule survive a one-month delay?"
    return revise(
        parent,
        replace(parent.specification, question=question),
        author="sundeep",
        at=TIME + timedelta(days=1),
        reason="Ask about the delayed rule",
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("start",),
        ("start", "revision.json"),
        ("status", "not-a-uuid", "--call-log", "calls"),
        ("status", CAMPAIGN),
        ("pause", CAMPAIGN, "--reason", "Hold"),
        ("resume", CAMPAIGN, "--actor", "sundeep"),
        ("start", "revision.json", "--task-queue", "research", "--label", "confirmation"),
        ("status", CAMPAIGN, "--call-log", "calls", *DECLARED),
        ("steer", "revision.json", "--actor", "sundeep"),
    ],
    ids=[
        "no step",
        "no revision",
        "no queue",
        "bad campaign",
        "no call log",
        "no actor",
        "no reason",
        "start declares",
        "status declares",
        "steer names an actor",
    ],
)
def test_campaign_commands_refuse_missing_and_foreign_arguments(tmp_path, cli, arguments):
    result = cli(tmp_path, "campaign", *arguments, success=False)
    assert result.returncode == 2
    assert result.stderr.startswith("usage: quant_research ")


@pytest.mark.parametrize("damage", ["missing", "directory", "undecodable", "indented", "nested"])
@pytest.mark.parametrize("step", ["start", "steer"])
def test_a_revision_that_cannot_be_read_or_parsed_fails_before_temporal(
    tmp_path, cli, setup, step, damage
):
    path = tmp_path / "revision.json"
    match damage:
        case "directory":
            path.mkdir()
        case "undecodable":
            path.write_bytes(b"\xff")
        case "indented":
            path.write_text(json.dumps(json.loads(file_bytes(pilot(setup))), indent=2))
        case "nested":
            path.write_bytes(b"[" * 100_000 + b"]" * 100_000)
    queue = ("--task-queue", "research") if step == "start" else ()
    result = cli(tmp_path, "campaign", step, str(path), *queue, success=False)
    assert result.returncode == 1
    assert result.stderr.startswith(f"Campaign {step} failed: ")
    assert result.stderr.count("\n") == 1


@pytest.mark.parametrize(
    "step, declaration, reason",
    [
        ("pause", ("--available-at", "2025-06-01"), "Invalid canonical --available-at time"),
        (
            "resume",
            ("--available-at", "2025-06-01T00:00:00+00:00"),
            "Invalid canonical --available-at time",
        ),
        ("cancel", ("--available-at", "soon"), "Invalid --available-at time"),
        (
            "steer",
            ("--label", "confirmation"),
            "--label and --approved-for declare text only beside --available-at",
        ),
        (
            "pause",
            ("--approved-for", CODEX),
            "--label and --approved-for declare text only beside --available-at",
        ),
    ],
    ids=["date", "offset", "word", "label alone", "provider alone"],
)
def test_a_declaration_needs_canonical_utc_time_before_its_labels_count(
    tmp_path, cli, setup, step, declaration, reason
):
    parent = pilot(setup)
    _, child = (save_revision(tmp_path / "campaigns", r) for r in (parent, steered(parent)))
    target = (
        (str(child),)
        if step == "steer"
        else (CAMPAIGN, "--actor", "sundeep", "--reason", "Use 2000 to 2019")
    )
    result = cli(tmp_path, "campaign", step, *target, *declaration, success=False)
    assert result.returncode == 1
    assert result.stderr == f"Campaign {step} failed: {reason}\n"


@pytest.mark.parametrize("log", ["missing", "file", "gap", "damaged"])
def test_status_refuses_a_call_log_it_cannot_replay(tmp_path, cli, log):
    calls = tmp_path / "calls"
    match log:
        case "file":
            calls.write_text("{}")
        case "gap":
            calls.mkdir()
            (calls / "1.json").write_text("{}")
        case "damaged":
            calls.mkdir()
            (calls / "0.json").write_text('{"a": 1}')
    result = cli(tmp_path, "campaign", "status", CAMPAIGN, "--call-log", str(calls), success=False)
    assert result.returncode == 1
    assert result.stderr.startswith(f"Campaign status failed: Cannot replay call log {calls}: ")


@pytest.mark.smoke
@pytest.mark.parametrize("step", ["start", "status", "pause", "resume", "cancel", "steer"])
def test_every_valid_command_loads_temporal_only_after_its_checks(tmp_path, cli, setup, step):
    parent = pilot(setup)
    first, child = (save_revision(tmp_path / "campaigns", r) for r in (parent, steered(parent)))
    (tmp_path / "calls").mkdir()
    declared = (*DECLARED, "--label", "confirmation")
    arguments = {
        "start": (str(first), "--task-queue", "research"),
        "status": (CAMPAIGN, "--call-log", str(tmp_path / "calls")),
        "steer": (str(child), *declared),
    }.get(step, (CAMPAIGN, "--actor", "sundeep", "--reason", "Hold", *declared))
    result = cli(tmp_path, "campaign", step, *arguments, success=False)
    assert result.stderr.endswith("ModuleNotFoundError: No module named 'temporalio'\n")
    assert f"Campaign {step} failed" not in result.stderr


def test_parsing_and_checking_a_campaign_command_loads_neither_temporal_nor_the_store(tmp_path):
    script = (
        f"import importlib.util, json, runpy, sys\nsys.path[:] = {sys.path!r}\n"
        f"sys.argv = ['quant_research', 'campaign', 'start', 'none.json', '--task-queue', 'q']\n"
        "try:\n    runpy.run_module('quant_research', run_name='__main__')\n"
        "except SystemExit:\n    pass\n"
        f"names = {LOADED!r}\n"
        "found = [name for name in names if importlib.util.find_spec(name)]\n"
        "print(json.dumps({'found': found, 'loaded': [n for n in names if n in sys.modules]}))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-c", script], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.stderr.startswith("Campaign start failed: [Errno 2]"), result.stderr
    assert json.loads(result.stdout) == {"found": list(LOADED), "loaded": []}


def ask(gateway: Gateway, attempt: UUID, *labels: str) -> tuple[Receipt, ...]:
    restrictions = Restrictions(
        available_at=WRITTEN, labels=frozenset(labels), approved_for=frozenset({A, B})
    )
    request = Request((Material("Next step?", restrictions),), (("action", str),))
    lease = Lease(attempt, (uuid4(), uuid4()), 1)
    gateway.call(CONTEXT, CHAIN, request, lease=lease, cancel=threading.Event())
    return gateway.close(lease)


def claim(log: CallLog, attempt: UUID) -> Lease:
    """Store the claim of an attempt's first call, as a worker that stops before the reply."""
    lease = Lease(attempt, (uuid4(), uuid4()), 1)
    log.append(
        Dispatched(
            lease.dispatches[0],
            attempt,
            1,
            "researcher",
            MODEL_A,
            WIDE.version,
            "0" * 64,
            True,
            "scripted",
            uuid4(),
        )
    )
    return lease


@pytest.fixture
def asked(tmp_path) -> tuple[CallLog, dict[UUID, tuple[Receipt, ...]]]:
    """Log a fallback, a refused fallback, a sealed claim, and another campaign's call."""
    (tmp_path / "calls").mkdir()
    log = CallLog(tmp_path / "calls")
    gateway = Gateway(
        log,
        [
            (WIDE, ScriptedProvider([Fault.FAIL, Fault.FAIL, ANSWER])),
            (NARROW, ScriptedProvider([ANSWER])),
        ],
    )
    fallback, refused, sealed = uuid4(), uuid4(), uuid4()
    receipts = {fallback: ask(gateway, fallback), refused: ask(gateway, refused, VENDOR)}
    receipts[sealed] = gateway.close(claim(log, sealed))
    ask(gateway, uuid4())
    return log, receipts


def settled(receipts: dict[UUID, tuple[Receipt, ...]], charge=lambda receipt: receipt.usage):
    """Charge each attempt's wall time and receipts as ResearchLoop._settle does, beside a hold."""
    campaign, trial = uuid4(), uuid4()
    ledger = Ledger({campaign: Budget(model_calls=100)})
    hold, working = Work(campaign, trial, uuid4(), uuid4()), uuid4()
    ledger.reserve(hold, Usage(model_calls=2))
    for attempt, sent in receipts.items():
        ledger.reserve(Work(campaign, trial, working, attempt), Usage(model_calls=2))
        ledger.charge(uuid4(), attempt, "researcher", Usage(wall_seconds=Decimal("1.5")))
        for receipt in sent:
            if (usage := charge(receipt)) is not None:
                ledger.charge(receipt.call_id, attempt, receipt.role, usage)
        ledger.release(attempt, "successful")
    ledger.release(hold.attempt_id, "inconclusive")
    return ledger


@pytest.mark.smoke
def test_ledger_receipts_match_every_sent_call_with_its_charge_and_skip_other_campaigns(asked):
    log, receipts = asked
    assert ledger_receipts(settled(receipts), log.load()) == LedgerReceipts(
        reserved=4,
        sent=4,
        charged=4,
        matched=4,
        refused=1,
        unknown=1,
        cost_unreported=(),
        differs=(),
        uncharged=(),
        unrecorded=(),
        unsealed=(),
        agreed=True,
    )


def inexact(usage: Usage) -> Usage:
    return replace(usage, cost_usd=Decimal(0), unreported=usage.unreported | {"cost_usd"})


@pytest.mark.parametrize(
    "change, named",
    [
        (inexact, "cost_unreported"),
        (lambda usage: replace(usage, input_tokens=usage.input_tokens + 1), "differs"),
        (lambda usage: None, "uncharged"),
    ],
    ids=["inexact cost", "other usage", "no charge"],
)
def test_a_charge_unlike_its_sent_call_is_named_by_the_call_id(asked, change, named):
    log, receipts = asked
    target = next(iter(receipts.values()))[-1]
    ledger = settled(receipts, lambda r: change(r.usage) if r == target else r.usage)
    found = ledger_receipts(ledger, log.load())
    lists = {name: getattr(found, name) for name in NAMED}
    assert lists == dict.fromkeys(NAMED, ()) | {named: (str(target.call_id),)}
    assert (found.matched, found.agreed) == (3, named == "cost_unreported")


def test_a_charge_or_a_claim_without_a_record_is_named_and_disagrees(asked):
    log, receipts = asked
    ledger, invented = settled(receipts), uuid4()
    ledger.charge(invented, next(iter(receipts)), "researcher", Usage(model_calls=1))
    found = ledger_receipts(ledger, log.load())
    assert (found.unrecorded, found.matched, found.agreed) == ((str(invented),), 4, False)
    stray = uuid4()
    lease = claim(log, stray)
    log.append(Closed(stray))
    found = ledger_receipts(settled(receipts | {stray: ()}), log.load())
    assert (found.unsealed, found.reserved, found.agreed) == ((str(lease.dispatches[0]),), 5, False)


def test_live_receipts_match_calls_by_id_without_inventing_unused_hold_calls(asked):
    log, receipts = asked
    ledger = settled(receipts)
    found = live_receipts(ledger, log.load(), settled=frozenset(receipts))
    assert (found.reserved, found.settled, found.unclosed, found.agreed) == (4, 3, (), True)
    assert sorted(check.state for check in found.calls) == ["matched"] * 4 + ["refused"]
    assert {check.call_id for check in found.calls} == {
        str(entry.call_id)
        for entry in log.load()
        if isinstance(entry, CallRecord) and entry.attempt_id in receipts
    }
    assert any(check.outcome == "unknown" and check.state == "matched" for check in found.calls)


@pytest.mark.parametrize("closed", [False, True])
def test_live_claim_and_reply_are_pending_until_workflow_settlement(asked, closed):
    log, receipts = asked
    attempt = uuid4()
    campaign = uuid4()
    ledger = Ledger({campaign: Budget()})
    ledger.reserve(Work(campaign, uuid4(), uuid4(), attempt), Usage())
    lease = claim(log, attempt)
    if closed:
        log.append(Closed(attempt))
    found = live_receipts(ledger, log.load(), settled=frozenset())
    assert [(check.call_id, check.state) for check in found.calls] == [
        (str(lease.dispatches[0]), "pending")
    ]
    assert found.agreed
    ledger.charge(uuid4(), attempt, "researcher", Usage(wall_seconds=Decimal(1)))
    ledger.release(attempt, "failed")
    final = live_receipts(ledger, log.load(), settled=frozenset({attempt}))
    assert final.calls[0].state == "unsealed"
    assert not final.agreed
    assert final.unclosed == (() if closed else (str(attempt),))
    target = next(iter(receipts))
    uncharged = Ledger({campaign: Budget()})
    uncharged.reserve(Work(campaign, uuid4(), uuid4(), target), Usage())
    pending = live_receipts(uncharged, log.load(), settled=frozenset())
    assert {check.state for check in pending.calls} == {"pending"}
    assert pending.agreed
    final = live_receipts(uncharged, log.load(), settled=frozenset({target}))
    assert {check.state for check in final.calls} == {"uncharged"}
    assert not final.agreed


@pytest.mark.parametrize("outcome", ["refused", "cancelled_before_claim"])
def test_live_settled_ask_requires_a_close_even_without_model_charges(tmp_path, outcome):
    calls = tmp_path / "calls"
    calls.mkdir()
    log = CallLog(calls)
    gateway = Gateway(
        log,
        [
            (replace(WIDE, labels=frozenset()), ScriptedProvider([])),
            (NARROW, ScriptedProvider([])),
        ],
    )
    attempt = uuid4()
    if outcome == "refused":
        assert ask(gateway, attempt, VENDOR) == ()
    else:
        assert gateway.close(Lease(attempt, (uuid4(),), 0)) == ()
    ledger = settled({attempt: ()})
    genuine = live_receipts(ledger, log.load(), settled=frozenset({attempt}))
    assert genuine.agreed and genuine.unclosed == () and genuine.settled == 1
    for entries in ((), (Closed(uuid4()),)):
        missing = live_receipts(ledger, entries, settled=frozenset({attempt}))
        assert not missing.agreed
        assert missing.unclosed == (str(attempt),)


def test_live_released_holds_are_finished_without_becoming_model_asks(asked):
    log, receipts = asked
    campaign, attempt, unused = uuid4(), next(iter(receipts)), uuid4()
    ledger = Ledger({campaign: Budget()})
    for held in (attempt, unused):
        ledger.reserve(Work(campaign, uuid4(), uuid4(), held), Usage())
        ledger.charge(uuid4(), held, "evaluation", Usage(compute_seconds=Decimal(1)))
        ledger.release(held, "inconclusive")
    empty = live_receipts(ledger, (), settled=frozenset())
    assert (empty.reserved, empty.settled, empty.calls, empty.unclosed, empty.agreed) == (
        2,
        0,
        (),
        (),
        True,
    )
    finished = live_receipts(ledger, log.load(), settled=frozenset())
    assert {check.state for check in finished.calls} == {"uncharged"}
    assert not finished.agreed
    lease = claim(log, unused)
    finished = live_receipts(ledger, log.load(), settled=frozenset())
    assert (
        next(check.state for check in finished.calls if check.call_id == str(lease.dispatches[0]))
        == "unsealed"
    )


@pytest.mark.parametrize("damage", ["role", "usage", "attempt", "call_id", "missing", "closed"])
def test_live_settled_evidence_remains_strict_even_with_matching_counts(asked, damage):
    log, receipts = asked
    ledger = settled(receipts)
    target = next(iter(receipts.values()))[-1]
    entries = list(log.load())
    index = next(
        i
        for i, entry in enumerate(entries)
        if isinstance(entry, CallRecord) and entry.call_id == target.call_id
    )
    record = entries[index]
    match damage:
        case "role":
            entries[index] = replace(record, role="critic")
        case "usage":
            entries[index] = replace(record, usage=replace(record.usage, input_tokens=5))
        case "attempt":
            entries[index] = replace(record, attempt_id=uuid4())
        case "call_id":
            entries[index] = replace(record, call_id=uuid4())
        case "missing":
            entries.pop(index)
        case "closed":
            entries = [
                entry
                for entry in entries
                if not isinstance(entry, Closed) or entry.attempt_id != record.attempt_id
            ]
    found = live_receipts(ledger, tuple(entries), settled=frozenset(receipts))
    assert not found.agreed
    states = {check.call_id: check.state for check in found.calls}
    if damage in {"role", "usage", "attempt"}:
        assert states[str(target.call_id)] == "differs"
        assert not live_receipts(ledger, tuple(entries), settled=frozenset()).agreed
    elif damage == "closed":
        assert found.unclosed == (str(record.attempt_id),)
    else:
        assert states[str(target.call_id)] == "unrecorded"


def test_live_receipts_allow_the_exact_cost_fallback_and_reject_a_charge_against_refusal(asked):
    log, receipts = asked
    ledger = settled(receipts, lambda receipt: inexact(receipt.usage))
    found = live_receipts(ledger, log.load(), settled=frozenset(receipts))
    assert found.agreed
    assert {check.state for check in found.calls} == {"cost_unreported", "refused"}
    refused = next(
        entry
        for entry in log.load()
        if isinstance(entry, CallRecord) and entry.outcome == "refused"
    )
    ledger.charge(refused.call_id, refused.attempt_id, refused.role, Usage(model_calls=1))
    found = live_receipts(ledger, log.load(), settled=frozenset(receipts))
    assert not found.agreed
    assert (
        next(check.state for check in found.calls if check.call_id == str(refused.call_id))
        == "differs"
    )


def campaign(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a campaign command in an interpreter that has Temporal, as a researcher's shell does."""
    bootstrap = (
        f"import runpy, sys; sys.path[:] = {sys.path!r}; "
        f"sys.argv = {['quant_research', 'campaign', *arguments]!r}; "
        "runpy.run_module('quant_research', run_name='__main__')"
    )
    return subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", bootstrap],
        cwd=directory,
        capture_output=True,
        text=True,
    )


def test_an_unreachable_temporal_fails_with_its_address(tmp_path):
    hold = ("--actor", "sundeep", "--reason", "Hold", "--address", "127.0.0.1:1")
    result = campaign(tmp_path, "pause", CAMPAIGN, *hold)
    assert result.returncode == 1
    assert result.stderr.startswith(
        "Campaign pause failed: Cannot connect to Temporal at 127.0.0.1:1: "
    )


def test_the_cli_drives_a_loop_through_every_command_and_reconciles_its_receipts(
    temporal, tmp_path, setup
):
    parent, other = pilot(setup), pilot(setup)
    child = steered(parent)
    first, second = (save_revision(tmp_path / "campaigns", r) for r in (parent, child))
    campaign_id, queue = str(parent.campaign_id), str(uuid4())
    calls, empty = tmp_path / "calls", tmp_path / "empty"
    calls.mkdir()
    empty.mkdir()
    provider = ScriptedProvider([STOP, STOP, STOP])
    gateway = Gateway(CallLog(calls), [(POLICY, provider)])
    shape = {item.name for item in fields(research.Report)} - {"steering"} | {"receipts"}

    async def run(*arguments: str, code: int = 0) -> subprocess.CompletedProcess[str]:
        result = await asyncio.to_thread(campaign, tmp_path, *arguments, "--address", temporal)
        assert result.returncode == code, result.stdout + result.stderr
        return result

    async def status(log: Path = calls) -> dict:
        shown = json.loads((await run("status", campaign_id, "--call-log", str(log))).stdout)
        assert set(shown) == shape
        return shown

    async def held(handle, step: int) -> None:
        expected = f"runner: Checkpoint before conclude holds step {step}"
        while (await research.status(handle)).paused_by != expected:
            await asyncio.sleep(0.05)

    async def body() -> None:
        client = await Client.connect(temporal)
        handle = client.get_workflow_handle_for(research.ResearchLoop.run, campaign_id)
        async with research.worker(
            client,
            queue,
            gateway,
            partial(resolve, setup=setup),
            archive=research.CampaignArchive(tmp_path / "archive", calls),
        ):
            unknown = await run("status", campaign_id, "--call-log", str(calls), code=1)
            assert (
                unknown.stderr
                == f"Campaign status failed: workflow not found for ID: {campaign_id}\n"
            )
            with pytest.raises(RPCError) as missing:
                await handle.describe()
            assert missing.value.status == RPCStatusCode.NOT_FOUND
            started = json.loads((await run("start", str(first), "--task-queue", queue)).stdout)
            run_id = (await handle.describe()).run_id
            assert started == {
                "campaign_id": campaign_id,
                "revision": parent.revision_id,
                "run_id": run_id,
            }
            await held(handle, 1)
            report = await status()
            assert {key: report[key] for key in ("workflow_id", "run_id", "state", "revision")} == {
                "workflow_id": campaign_id,
                "run_id": run_id,
                "state": "paused",
                "revision": parent.revision_id,
            }
            assert report["paused_by"] == "runner: Checkpoint before conclude holds step 1"
            assert (report["steps"], report["ending"]) == ([], None)
            assert report["ledger"][0]["usage"]["model_calls"] == 1
            assert (report["limits"]["max_model_calls"], report["spent"]["model_calls"]) == (6, "1")
            receipts = report["receipts"]
            assert (receipts["reserved"], receipts["settled"], receipts["agreed"]) == (4, 1, True)
            assert [call["state"] for call in receipts["calls"]] == ["matched"]
            assert receipts["unclosed"] == []

            joined = await run("start", str(first), "--task-queue", queue)
            assert json.loads(joined.stdout) == started
            another = await run("start", str(second), "--task-queue", queue, code=1)
            assert another.stderr == (
                f"Campaign start failed: Campaign {campaign_id} runs another revision, other "
                "rules, or another runner\n"
            )
            late = ("--available-at", "2025-06-02T00:00:00Z", "--approved-for", CODEX)
            refused = await run(
                "resume", campaign_id, "--actor", "sundeep", "--reason", "Refused", *late, code=1
            )
            assert refused.stderr == (
                "Campaign resume failed: Declared text available at 2025-06-02T00:00:00Z is after "
                "the run's cutoff 2025-06-01T00:00:00Z\n"
            )

            paused = await run(
                "pause", campaign_id, "--actor", "sundeep", "--reason", "UNDECLARED pause"
            )
            assert json.loads(paused.stdout) == {"campaign_id": campaign_id, "state": "paused"}
            assert (await status())["paused_by"] == "sundeep: UNDECLARED pause"
            steer = await run("steer", str(second), *DECLARED)
            assert json.loads(steer.stdout) == {
                "campaign_id": campaign_id,
                "revision": child.revision_id,
                "state": "paused",
            }
            assert (await status())["revision"] == child.revision_id
            resumed = await run(
                "resume", campaign_id, "--actor", "sundeep", "--reason", "UNDECLARED resume"
            )
            assert json.loads(resumed.stdout) == {"campaign_id": campaign_id, "state": "running"}
            await held(handle, 2)
            report = await status()
            assert [(step["action"], step["result"]) for step in report["steps"]] == [
                ("stop", "set_aside")
            ]
            assert report["paused_by"] == "runner: Checkpoint before conclude holds step 2"
            receipts = report["receipts"]
            assert [call["state"] for call in receipts["calls"]] == ["matched", "matched"]
            assert receipts["agreed"]
            cancelled = await run(
                "cancel", campaign_id, "--actor", "sundeep", "--reason", "UNDECLARED cancel"
            )
            assert json.loads(cancelled.stdout) == {
                "campaign_id": campaign_id,
                "state": "cancelled",
            }
            await handle.result()

            report = await status()
            assert (report["state"], report["paused_by"], report["revision"]) == (
                "cancelled",
                None,
                child.revision_id,
            )
            assert report["ending"] == {
                "outcome": "cancelled",
                "reason": "researcher",
                "steps": 2,
                "final": None,
            }
            assert [(step["result"], step["revision"]) for step in report["steps"]] == [
                ("set_aside", parent.revision_id),
                ("cancelled", child.revision_id),
            ]
            [line] = report["ledger"]
            assert (line["campaign_id"], line["role"], line["outcome"]) == (
                campaign_id,
                "researcher",
                "successful",
            )
            assert line["usage"]["model_calls"] == len(provider.calls) == 2
            assert report["receipts"] == {
                "reserved": 5,
                "sent": 2,
                "charged": 2,
                "matched": 2,
                "refused": 0,
                "unknown": 0,
                "cost_unreported": [],
                "differs": [],
                "uncharged": [],
                "unrecorded": [],
                "unsealed": [],
                "agreed": True,
            }
            disagreed = await run("status", campaign_id, "--call-log", str(empty), code=1)
            assert disagreed.stderr == "Receipt reconciliation found disagreements\n"
            receipts = json.loads(disagreed.stdout)["receipts"]
            sent = sorted(
                str(e.call_id) for e in CallLog(calls).load() if isinstance(e, CallRecord)
            )
            assert (receipts["matched"], receipts["unrecorded"], receipts["agreed"]) == (
                0,
                sent,
                False,
            )
            closed = await run(
                "resume", campaign_id, "--actor", "sundeep", "--reason", "Go", code=1
            )
            assert closed.stderr == "Campaign resume failed: workflow execution already completed\n"
            with pytest.raises(RPCError) as completed:
                await research.control(handle, "resume", actor="sundeep", reason="Go")
            assert completed.value.status == RPCStatusCode.NOT_FOUND

            assert "Researcher steer: Ask about the delayed rule" in provider.calls[1].texts
            shown = [text for call in provider.calls for text in call.texts]
            assert not [text for text in shown if "UNDECLARED" in text or "Refused" in text]

            terminated = await research.start(client, other, queue)
            await held(terminated, 1)
            await terminated.terminate("The operator stopped the loop")
            other_id = str(other.campaign_id)
            failed = await run("status", other_id, "--call-log", str(calls), code=1)
            assert failed.stderr == "Campaign status failed: terminated by user via frontend\n"
            assert failed.stdout == ""

    asyncio.run(asyncio.wait_for(body(), 120))


def test_oversized_steer_is_refused_before_parsing_or_loading_temporal(tmp_path, cli):
    revision = tmp_path / "too-large.json"
    revision.write_bytes(b"x" * 65537)
    result = cli(tmp_path, "campaign", "steer", str(revision), success=False)
    assert result.returncode == 1
    assert result.stderr == (
        "Campaign steer failed: A steer revision holds at most 65536 UTF-8 bytes\n"
    )
