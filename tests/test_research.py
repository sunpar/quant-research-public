from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from functools import partial
from typing import NamedTuple
from uuid import uuid4

import pytest
from temporalio import workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client, WorkflowHandle, WorkflowUpdateFailedError
from temporalio.converter import DataConverter
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

from quant_research import (
    acceptance,
    accounting,
    datasets,
    evaluation,
    execution,
    prospective,
    reference,
    research,
    signals,
)
from quant_research.access import Provider, Restrictions, derive, licensed
from quant_research.accounting import Charged, Released, Reserved, load_ledger
from quant_research.activities import Activities
from quant_research.agent_tools import MAX_RESULT_BYTES, TOOLS, Located, Toolset
from quant_research.campaigns import (
    PROMPT,
    CampaignRevision,
    CampaignSpecification,
    DataPermission,
    EvaluationRules,
    ModelRole,
    ResourceLimits,
    Setup,
    Source,
    contract_sources,
    file_bytes,
    parse_file,
    register,
    resolve,
    revise,
    start_campaign,
)
from quant_research.catalog import Approved, Catalog, Documented, Evidence, Validated, Vintage
from quant_research.findings import Ledger
from quant_research.locators import MARKDOWN_LOCATOR, publish_locations
from quant_research.model_provenance import MODELS, RoleAssignment
from quant_research.packages import LineageLink
from quant_research.providers import (
    CallLog,
    CallRecord,
    Fault,
    Gateway,
    Material,
    ProviderPolicy,
    Request,
    ScriptedProvider,
    dump_material,
    load_material,
)
from quant_research.records import canonical_bytes, format_time
from quant_research.research import (
    BOUND,
    CLOSING,
    OMITTED,
    ROLE,
    ROLE_BYTES,
    SCHEMA,
    STEP_CAP,
    WITHHELD,
    Assessment,
    Before,
    Conclude,
    Ending,
    Every,
    Experiment,
    HeldBack,
    Report,
    ResearchError,
    Role,
    Stop,
    Transcript,
    UseTool,
    check_declaration,
    checkpoints,
    parse_move,
)
from quant_research.signals import Lag, Series, Subtract, Threshold
from quant_research.sources import SourceDeclaration, open_generation
from quant_research.steering import Control

TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
CUTOFF = datetime(2025, 6, 1, tzinfo=UTC)
CODEX = Provider("codex-subscription")
VENDOR = licensed("synthetic-vendor")
PROMPTED = Restrictions(
    available_at=datetime(2025, 1, 1, tzinfo=UTC), labels=frozenset(), approved_for={CODEX}
)
FLOOR = Restrictions(available_at=CUTOFF, labels=frozenset({VENDOR}), approved_for={CODEX})
HEAD = "a" * 64
CHILD = "b" * 64


def role(prompt: str = "You are the researcher.") -> Role:
    return Role(
        Material(prompt, PROMPTED),
        Material("Reply with one JSON object.", PROMPTED),
        RoleAssignment("researcher", MODELS["codex-subscription/gpt-6.1-sol"]),
        "c" * 64,
        FLOOR,
        frozenset(tool.name for tool in TOOLS),
        (),
    )


def reply(action: str = "read_dataset", arguments: str = '{"dataset": "alfred.unrate"}') -> str:
    return json.dumps({"action": action, "arguments": arguments, "rationale": "Look at it"})


def restricted(rng: random.Random) -> Restrictions:
    return Restrictions(
        available_at=CUTOFF - timedelta(days=rng.randrange(400)),
        labels=frozenset(
            rng.sample([VENDOR, licensed("other"), licensed("third")], rng.randrange(3))
        ),
        approved_for=frozenset(
            rng.sample([CODEX, Provider("claude-subscription")], rng.randrange(3))
        ),
    )


def covers(outer: Restrictions, inner: Restrictions) -> bool:
    return (
        outer.available_at >= inner.available_at
        and outer.labels >= inner.labels
        and outer.approved_for <= inner.approved_for
    )


def interventions(*texts: str):
    specification = CampaignSpecification(
        "Does the real-time Sahm rule time the market?",
        (),
        (),
        EvaluationRules("pilot-evaluation/v1", hashlib.sha256(b"rules\n").hexdigest()),
        ResourceLimits(5, 1, 100),
        texts,
    )
    return start_campaign(specification, author="researcher", at=TIME, reason="Pilot protocol")


@pytest.mark.parametrize("tool", [tool.name for tool in TOOLS] + ["no_such_tool", " conclude"])
def test_any_action_but_the_three_words_is_a_tool_call_with_its_arguments(tool):
    assert parse_move({"action": tool, "arguments": "{}", "rationale": "Why"}) == UseTool(
        tool, "{}"
    )


def test_experiment_conclude_and_stop_are_their_own_moves():
    values = {"arguments": '{"claim": "none"}', "rationale": "Why"}
    assert parse_move(values | {"action": "experiment"}) == Experiment('{"claim": "none"}')
    assert parse_move(values | {"action": "conclude"}) == Conclude('{"claim": "none"}')
    assert parse_move(values | {"action": "stop"}) == Stop()


def test_checkpoints_parse_every_n_steps_and_before_conclude():
    revision = interventions(
        "checkpoint every 1 steps",
        f"checkpoint every {STEP_CAP} steps",
        "checkpoint before conclude",
    )
    assert checkpoints(revision) == (Every(1), Every(STEP_CAP), Before())
    assert checkpoints(interventions()) == ()


@pytest.mark.parametrize(
    "text",
    [
        "checkpoint every 0 steps",
        "checkpoint every 05 steps",
        f"checkpoint every {STEP_CAP + 1} steps",
        "checkpoint every 1 step",
        "checkpoint before stop",
        "Checkpoint before conclude",
        "before_trial",
    ],
)
def test_checkpoints_refuse_any_other_intervention(text):
    with pytest.raises(ResearchError, match="neither"):
        checkpoints(interventions("checkpoint before conclude", text))


def test_the_role_prompt_leads_every_request_and_closing_text_ends_the_last():
    transcript = Transcript()
    researcher = role()
    first = transcript.prompt(HEAD, researcher)
    transcript.add_step(HEAD, 1, Material(reply(), PROMPTED), Material('{"rows": []}', FLOOR))
    later = transcript.prompt(HEAD, researcher)
    closing = transcript.prompt(HEAD, researcher, closing=True)
    for request in (first, later, closing):
        assert request.materials[:2] == (researcher.prompt, researcher.protocol)
        assert request.schema == SCHEMA
    assert closing.materials[-1] == Material(CLOSING, PROMPTED)
    assert closing.materials[:-1] == later.materials
    assert CLOSING not in [material.text for material in later.materials]


def test_a_step_shows_its_reply_and_result_under_the_restrictions_of_both():
    transcript = Transcript()
    answered = Material(reply(), PROMPTED)
    read = Material('{"rows": []}', replace(FLOOR, labels=frozenset({VENDOR})))
    transcript.add_step(HEAD, 3, answered, read)
    request = transcript.prompt(HEAD, role())
    step = request.materials[2]
    assert step.text == f"Step 3: {answered.text}\nResult: {read.text}"
    assert step.restrictions == Restrictions(
        available_at=CUTOFF, labels=frozenset({VENDOR}), approved_for={CODEX}
    )
    assert request.restrictions == step.restrictions
    assert transcript.prompt(HEAD, role(), closing=True).restrictions == step.restrictions


def test_a_note_shows_the_researchers_words_under_the_declared_restrictions():
    transcript = Transcript()
    note = Control("annotate", "sundeep", TIME, "Ignore the 2008 episode", FLOOR)
    transcript.add_note(HEAD, note, FLOOR)
    [shown] = transcript.prompt(HEAD, role()).materials[2:]
    assert shown == Material("Researcher annotate: Ignore the 2008 episode", FLOOR)


def test_requests_stay_in_bound_and_never_fall_below_an_entry_shown_earlier():
    rng = random.Random(12)
    for _ in range(4):
        transcript, researcher = Transcript(), role("p" * rng.randrange(ROLE_BYTES - 2**12))
        shown: list[Restrictions] = []
        for number in range(1, 80):
            answered = Material(reply(), restricted(rng))
            size = rng.choice([10, 2**10, 2**14, 2**17, MAX_RESULT_BYTES, BOUND])
            result = Material(json.dumps({"rows": "r" * size}), restricted(rng))
            transcript.add_step(HEAD, number, answered, result)
            closing = rng.random() < 0.2
            request = transcript.prompt(HEAD, researcher, closing=closing)
            assert len(request.to_bytes()) <= BOUND
            assert all(covers(request.restrictions, item) for item in shown)
            newest = request.materials[-2 if closing else -1].text
            if newest.startswith(f"Step {number}: ") or newest == WITHHELD:
                shown.append(derive(answered.restrictions, result.restrictions))
        assert len(shown) > 40


def test_the_bound_holds_a_full_dataset_read_beside_a_role_of_role_bytes():
    row = {
        "arrived_at": "2025-02-08T00:00:00Z",
        "key": "2025-01-01",
        "published_at": "2025-02-08T00:00:00Z",
        "revision": 1,
        "values": {"value": "4.0"},
    }
    count = MAX_RESULT_BYTES // len(canonical_bytes(row)) - 1
    rows = canonical_bytes({"dataset": "alfred.unrate", "rows": [row] * count}).decode()
    assert MAX_RESULT_BYTES - 2**10 < len(rows.encode()) <= MAX_RESULT_BYTES
    padding = ROLE_BYTES - len(Transcript().prompt(HEAD, role(""), closing=True).to_bytes()) - 64
    researcher = role("p" * padding)
    assert ROLE_BYTES - 64 <= len(Transcript().prompt(HEAD, researcher, closing=True).to_bytes())
    transcript = Transcript()
    transcript.add_step(HEAD, 1, Material(reply(), PROMPTED), Material(rows, FLOOR))
    request = transcript.prompt(HEAD, researcher, closing=True)
    assert request.materials[2].text.endswith(rows)
    assert len(request.to_bytes()) <= BOUND


def test_a_step_too_large_for_the_window_is_withheld_under_its_own_restrictions():
    transcript = Transcript()
    result = Material('"' * (BOUND - ROLE_BYTES), replace(FLOOR, labels=frozenset({VENDOR})))
    digest = transcript.add_step(HEAD, 1, Material(reply(), PROMPTED), result)
    [withheld] = transcript.prompt(HEAD, role()).materials[2:]
    assert withheld == Material(WITHHELD, derive(PROMPTED, result.restrictions))
    assert digest == hashlib.sha256(json.dumps(WITHHELD).encode()).hexdigest()


def test_no_request_is_built_when_the_restrictions_of_the_entries_alone_pass_the_bound():
    transcript = Transcript()
    labels = frozenset(licensed(f"vendor-{number:05}") for number in range(BOUND // 20))
    result = Material("{}", replace(FLOOR, labels=labels))
    transcript.add_step(HEAD, 1, Material(reply(), PROMPTED), result)
    assert transcript.prompt(HEAD, role()) is None
    assert transcript.prompt(HEAD, role(), closing=True) is None


def test_the_oldest_steps_are_dropped_for_good_behind_one_notice():
    transcript, researcher = Transcript(), role()
    labelled = replace(FLOOR, labels=frozenset({VENDOR}))
    bulky = json.dumps({"rows": "r" * (BOUND // 3)})
    transcript.add_step(HEAD, 1, Material(reply(), PROMPTED), Material(bulky, labelled))
    for number in (2, 3):
        transcript.add_step(HEAD, number, Material(reply(), PROMPTED), Material(bulky, FLOOR))
    texts = [material.text for material in transcript.prompt(HEAD, researcher).materials]
    assert texts[2] == OMITTED
    assert [text.split(":")[0] for text in texts[3:]] == ["Step 2", "Step 3"]
    transcript.add_step(HEAD, 4, Material(reply(), PROMPTED), Material(bulky, FLOOR))
    request = transcript.prompt(HEAD, researcher)
    assert [material.text.split(":")[0] for material in request.materials[2:]] == [
        OMITTED.split(":")[0],
        "Step 3",
        "Step 4",
    ]
    assert request.materials[2].restrictions.labels == {VENDOR}
    assert request.restrictions.labels == {VENDOR}


def test_the_same_calls_build_byte_identical_requests():
    def built() -> list[bytes]:
        transcript, researcher, requests = Transcript(), role(), []
        for number in range(1, 8):
            result = Material(json.dumps({"rows": "r" * (BOUND // 4)}), FLOOR)
            transcript.add_step(HEAD, number, Material(reply(), PROMPTED), result)
            requests += [transcript.prompt(HEAD, researcher).to_bytes() for _ in range(2)]
        return requests

    first = built()
    assert first == built()
    assert first[0::2] == first[1::2]


def test_a_revision_shows_only_the_entries_made_under_it():
    transcript = Transcript()
    for number in range(1, 5):
        result = Material(json.dumps({"rows": "r" * (BOUND // 3)}), FLOOR)
        transcript.add_step(HEAD, number, Material(reply(), PROMPTED), result)
    assert transcript.prompt(HEAD, role()).materials[2].text == OMITTED
    child = role("You are the researcher on the narrowed question.")
    assert transcript.prompt(CHILD, child).materials == (child.prompt, child.protocol)
    transcript.add_step(CHILD, 5, Material(reply(), PROMPTED), Material("{}", FLOOR))
    [step] = transcript.prompt(CHILD, child).materials[2:]
    assert step.text.startswith("Step 5: ")


def test_a_declaration_inside_the_floor_passes():
    check_declaration(FLOOR, FLOOR)
    check_declaration(replace(FLOOR, labels=frozenset(), approved_for={CODEX, "other"}), FLOOR)


@pytest.mark.parametrize(
    "change, named",
    [
        ({"available_at": CUTOFF + timedelta(seconds=1)}, "after the run's cutoff"),
        ({"approved_for": frozenset()}, "codex-subscription"),
        ({"labels": frozenset({VENDOR, licensed("other")})}, "licensed:other"),
    ],
)
def test_a_declaration_outside_the_floor_is_refused(change, named):
    with pytest.raises(ResearchError, match=named):
        check_declaration(replace(FLOOR, **change), FLOOR)


# The pilot store of tests/test_campaigns.py, copied: a brief, a researcher prompt, two datasets
# with their contracts, and the Codex route under a policy that admits no label.
WRITTEN = datetime(2025, 1, 1, tzinfo=UTC)
START = datetime(2000, 1, 1, tzinfo=UTC)
END = datetime(2022, 6, 1, tzinfo=UTC)
CODEX_MODEL = "codex-subscription/gpt-6.1-sol"
CODEX_POLICY = ProviderPolicy(provider=CODEX, labels=frozenset(), retains=True)
QUESTION = "Does the real-time Sahm rule time the market?"
BRIEF = "The pilot asks whether the real-time Sahm rule times the market.\n"
INSTRUCTIONS = "You are the researcher. Propose one step at a time.\n"
DECLARATION = SourceDeclaration(
    media_type="text/markdown",
    access_domain="repository",
    location="docs/phase0-pilot.md",
    revision="1",
    acquisition="hand-authored test fixture",
    rights="synthetic, unrestricted",
    labels=frozenset(),
    approved_for=frozenset({CODEX}),
    published_at=None,
    source_available_at=WRITTEN,
    availability_evidence="authored for these tests",
    received_at=WRITTEN,
)
# Before the search cutoff, so every read supplies it.
ROW = {
    "arrived_at": "2019-02-08T00:00:00Z",
    "key": "2019-01-01",
    "published_at": "2019-02-08T00:00:00Z",
    "revision": 1,
    "values": {"value": "4.0"},
}
STORED = ("alfred.unrate", "alfred.sahmrealtime")
UNRATE = "alfred.unrate"
FACTORS = "french.ff3_monthly"
FACTOR_FIELDS = ("mkt_rf", "smb", "hml", "rf")
SEARCHED = (UNRATE, FACTORS)
# Monthly history from 2016, so a search candidate decides from July 2017, through 2023, so the
# store holds rows that no read or experiment under the search cutoff may supply.
MONTHS = tuple(signals.shift(date(2016, 1, 1), offset) for offset in range(96))
SAHM_TEXT = signals.canonical(signals.SAHM_RULE)
RISE_TEXT = signals.canonical(
    Threshold(Subtract(Series(UNRATE), Lag(Series(UNRATE), 12)), Fraction(1, 2))
)
LEAKY_TEXT = signals.canonical(Threshold(Lag(Series(UNRATE), 1), Fraction(5))).replace(
    b'"months":1', b'"months":-1'
)
PAST_TEXT = signals.canonical(Threshold(Series(UNRATE, vintage=1), Fraction(5))).replace(
    b'"vintage":1', b'"vintage":-1'
)


def window(start: str, end: str, cutoff: str) -> dict:
    """Return one window of a rules file, in reference.json's schema."""
    return {
        "baseline": "buy_and_hold_market",
        "cutoff": cutoff,
        "evaluation": {
            "alpha": "0.05",
            "minimum_effect": "0.001",
            "subperiods": 2,
            "version": "pilot-evaluation/v1",
        },
        "schema_version": 1,
        "window": {"end": end, "start": start},
    }


# Thirty months each, ending before CUTOFF, so the prompt and brief are written after the search.
RULES = reference.load_rules(
    canonical_bytes(
        {
            "held_back": window("2020-01-01", "2022-06-01", "2022-07-01T00:00:00Z"),
            "schema_version": 1,
            "search": window("2017-07-01", "2019-12-01", "2020-01-01T00:00:00Z"),
        }
    )
)
TOOLSET = tuple(field.name for field in fields(Toolset))
# Inside the pilot's floor: before the cutoff, no label, approved for the Codex route.
NOTED = Restrictions(available_at=WRITTEN, labels=frozenset(), approved_for={CODEX})


def documented(name: str, data: bytes, columns: tuple[str, ...] = ("value",)) -> tuple:
    evidence = Evidence(hashlib.sha256(b"publication").hexdigest(), "Synthetic publication")
    vintages = tuple(Vintage(aspect, evidence) for aspect in ("publication", "revision", "arrival"))
    vintage = hashlib.sha256(data).hexdigest()
    return (
        Documented(name, "researcher", TIME, "Data", columns, (), vintage, (), vintages, (CODEX,)),
        Validated(name, "reviewer", TIME, evidence),
        Approved(name, "lead", TIME, datasets.ADAPTER),
    )


def dataset_file(name: str) -> bytes:
    return canonical_bytes({"dataset": name, "rows": [ROW], "schema_version": 1})


@pytest.fixture
def setup(tmp_path) -> Setup:
    sources = tmp_path / "sources"
    sources.mkdir()
    store = tmp_path / "datasets"
    for name in STORED:
        datasets.save(store, name, dataset_file(name))
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / f"{RULES.digest}.json").write_bytes(RULES.data)
    return Setup(
        sources=sources,
        datasets=store,
        catalog=Catalog(
            tuple(event for name in STORED for event in documented(name, dataset_file(name)))
        ),
        ledger=Ledger(),
        clearances=frozenset(),
        cutoff=CUTOFF,
        policies=(CODEX_POLICY,),
        rules=rules,
        packages=tmp_path / "packages",
        lock=b"synthetic lock\n",
    )


def history(name: str, day: int, values: Callable[[int], dict[str, str]]) -> bytes:
    """Return a dataset file of one row per month of MONTHS, each available on day of the next."""
    rows = []
    for index, month in enumerate(MONTHS):
        following = signals.shift(month, 1)
        available = format_time(datetime(following.year, following.month, day, tzinfo=UTC))
        rows.append(
            {
                "arrived_at": available,
                "key": month.isoformat(),
                "published_at": available,
                "revision": 1,
                "values": values(index),
            }
        )
    return canonical_bytes({"dataset": name, "rows": rows, "schema_version": 1})


@pytest.fixture
def searching(setup) -> Setup:
    """The pilot setup over monthly UNRATE, out the 5th of the next month, and factors."""
    unrate = history(UNRATE, 6, lambda index: {"value": f"{4 + index % 17 / 10:.1f}"})
    factors = history(
        FACTORS,
        1,
        lambda index: {
            "hml": "-0.10",
            "mkt_rf": f"{(index * 37 % 23 * 50 - 550) / 100:.2f}",
            "rf": f"{(10 + index % 7 * 2) / 100:.2f}",
            "smb": "0.10",
        },
    )
    store = setup.datasets.parent / "searched"
    datasets.save(store, UNRATE, unrate)
    datasets.save(store, FACTORS, factors)
    events = (*documented(UNRATE, unrate), *documented(FACTORS, factors, FACTOR_FIELDS))
    return replace(setup, datasets=store, catalog=Catalog(events))


def prompt_source(setup: Setup, text: str = INSTRUCTIONS, revision: str = "1") -> Source:
    declaration = replace(DECLARATION, location="prompts/researcher.md", revision=revision)
    return register(
        setup.sources, name=ROLE, role=PROMPT, data=text.encode(), declaration=declaration
    )


def pilot(
    setup: Setup,
    *texts: str,
    limits: ResourceLimits | None = None,
    fallbacks=(),
    names: tuple[str, ...] = STORED,
) -> CampaignRevision:
    brief = register(
        setup.sources,
        name="pilot-brief",
        role="brief",
        data=BRIEF.encode(),
        declaration=DECLARATION,
    )
    specification = CampaignSpecification(
        QUESTION,
        (brief, prompt_source(setup), *contract_sources(setup.sources, setup.catalog, names)),
        tuple(DataPermission(name, START, END) for name in names),
        EvaluationRules("pilot-evaluation/v1", RULES.digest),
        limits or ResourceLimits(5, 2, 100, max_model_calls=20),
        texts,
        (ModelRole(ROLE, CODEX_MODEL, fallbacks, "active"),),
    )
    return start_campaign(specification, author="researcher", at=TIME, reason="Pilot protocol")


def child_of(parent: CampaignRevision, reason: str = "Narrow the question", **changes):
    specification = replace(parent.specification, **changes)
    return revise(
        parent, specification, author="sundeep", at=TIME + timedelta(days=1), reason=reason
    )


def says(action: str, arguments: object = None, rationale: str = "Next step") -> str:
    return json.dumps(
        {"action": action, "arguments": json.dumps(arguments or {}), "rationale": rationale}
    )


def propose(program: bytes) -> str:
    """Return a reply that asks for an experiment on a program's own JSON document."""
    return json.dumps({"action": "experiment", "arguments": program.decode(), "rationale": "Try"})


def momentum(trial: str | None) -> str:
    """Return the summary of {"program": "momentum"}, which parse rejects before any read."""
    detail = f"Rejected (missing_field): {signals.Rejected('missing_field').message}"
    summary = {"detail": detail, "evaluation": None, "status": "rejected", "trial": trial}
    return canonical_bytes(summary).decode()


SEARCH = says("search_corpus", {"mode": "lexical", "text": "Sahm"})
READ = says("read_dataset", {"dataset": "alfred.unrate"})
CONCLUDE = says("conclude", {"trials": []}, "There are no candidates")
STOP = says("stop", rationale="Nothing more to learn")
NO_RESULT = json.dumps(
    {"classification": "no_result", "citations": [], "explanation": "No candidates"}
)
ACCEPT = json.dumps({"decision": "accept", "citations": [], "explanation": "The evidence agrees"})


class Run(NamedTuple):
    client: Client
    queue: str
    provider: ScriptedProvider
    log: CallLog

    def sent(self) -> list[Request]:
        """Return every request the gateway sent, in order, as its transcript holds it."""
        return [self.log.sent(entry) for entry in self.log.load() if isinstance(entry, CallRecord)]


def serve(
    temporal,
    tmp_path,
    setup,
    script,
    body,
    *,
    authorize=None,
    unsandboxed=False,
    additional_routes=(),
    archive_importer=None,
) -> None:
    calls = tmp_path / "calls"
    calls.mkdir()
    provider = ScriptedProvider(script)
    log = CallLog(calls)
    policy = next(item for item in setup.policies if item.provider == CODEX)
    gateway = Gateway(log, [(policy, provider), *additional_routes])
    authorize = authorize or partial(resolve, setup=setup)
    archive = research.CampaignArchive(tmp_path / "archive", calls, importer=archive_importer)

    async def run() -> None:
        client, queue = await Client.connect(temporal), str(uuid4())
        if unsandboxed:
            # A patched constant of research reaches only a workflow run outside the sandbox,
            # which otherwise imports the module afresh. The registration is research.worker's.
            serving = Worker(
                client,
                task_queue=queue,
                workflows=[research.ResearchLoop],
                activities=[
                    *Activities(gateway, authorize).definitions,
                    *research.Authorized(authorize, archive).definitions,
                ],
                workflow_runner=UnsandboxedWorkflowRunner(),
            )
        else:
            serving = research.worker(client, queue, gateway, authorize, archive=archive)
        async with serving:
            await body(Run(client, queue, provider, log))

    asyncio.run(asyncio.wait_for(run(), 120))


async def reaches(handle: WorkflowHandle, state: str) -> Report:
    while (report := await research.status(handle)).state != state:
        await asyncio.sleep(0.05)
    return report


def texts(request: Request) -> list[str]:
    return [material.text for material in request.materials]


def ledger(report: Report):
    return load_ledger(report.ledger.encode())


def released(report: Report) -> list[str]:
    events = ledger(report).events
    reserved = [event.work.attempt_id for event in events if isinstance(event, Reserved)]
    outcomes = {event.attempt_id: event.outcome for event in events if isinstance(event, Released)}
    return [outcomes.get(attempt, "open") for attempt in reserved]


def charged(report: Report, field: str):
    return sum(
        getattr(event.usage, field) for event in ledger(report).events if isinstance(event, Charged)
    )


def steering(report: Report) -> list[tuple[str, str, str]]:
    return [
        (event["action"], event["actor"], event["reason"])
        for event in json.loads(report.steering)["events"]
    ]


def test_an_empty_shortlist_gets_two_fresh_calls_and_one_evaluation_and_replays(
    temporal, tmp_path, setup
):
    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        ending = await handle.result()
        report, compact = await research.record(handle), await research.status(handle)
        assert ending.outcome == "concluded"
        assessment = report.assessment
        assert assessment.classification == "no_result"
        assert assessment.trials == ()
        assert assessment.search_package == assessment.package == report.package == ""
        assert load_material(assessment.conclusion).text == NO_RESULT
        assert load_material(assessment.critique).text == ACCEPT
        assert not assessment.independent
        assert compact.assessment.conclusion is compact.assessment.critique is None
        assert len(set((*assessment.conclusion_calls, *assessment.critique_calls))) == 2
        selection, conclusion, critique = run.sent()
        assert selection.schema == SCHEMA
        assert conclusion.schema is critique.schema is None
        assert texts(conclusion)[-1] == research.CONCLUSION
        assert texts(critique)[-1] == research.CRITIQUE
        assert NO_RESULT in texts(critique)
        assert charged(report, "model_calls") == 3
        assert "open" not in released(report)
        history = await handle.fetch_history()
        scheduled = [
            event.activity_task_scheduled_event_attributes.activity_type.name
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
        ]
        assert scheduled.count("evaluate") == 1
        await Replayer(workflows=[research.ResearchLoop]).replay_workflow(history)

    serve(temporal, tmp_path, setup, [CONCLUDE, NO_RESULT, ACCEPT], body)


def claim(classification: str, citations=()) -> str:
    return json.dumps(
        {
            "classification": classification,
            "citations": list(citations),
            "explanation": "Held-back interpretation",
        }
    )


def assessment(verdicts, classification, *, decision="accept", citations=None):
    trials = []
    reasons = {
        "supported": "above_minimum",
        "null": "below_minimum",
        "inconclusive": "spans_minimum",
    }
    for verdict in verdicts:
        result = None
        if verdict is not None:
            interval = evaluation.Interval(0.01, 0.001, 0.005, 0.015)
            result = reference.encode_evaluation(
                evaluation.Evaluation(
                    reasons[verdict],
                    5,
                    0.99,
                    interval,
                    interval,
                    (0.01, 0.01),
                    evaluation.Profile(0.001, 2, 0.05),
                )
            ).decode()
        trials.append(
            HeldBack(str(uuid4()), str(uuid4()), "rejected" if result is None else "ran", result)
        )
    ids = [trial.trial for trial in trials] if citations is None else citations
    return Assessment(
        HEAD,
        "",
        HEAD,
        CHILD,
        tuple(trials),
        "",
        conclusion=dump_material(Material(claim(classification, ids), PROMPTED)),
        critique=dump_material(
            Material(
                json.dumps({"decision": decision, "citations": [], "explanation": "Review"}),
                PROMPTED,
            )
        ),
    )


@pytest.mark.parametrize(
    "verdicts, classification, expected",
    [
        (["supported"], "supported", "supported"),
        (["null", "null"], "null", "null"),
        (["inconclusive"], "inconclusive", "inconclusive"),
        ([], "no_result", "no_result"),
        ([], "supported", "inconclusive"),
        ([None], "supported", "inconclusive"),
        ([None], "no_result", "inconclusive"),
        (["null", None], "null", "inconclusive"),
        (["null", "inconclusive"], "null", "inconclusive"),
        (["null"], "supported", "inconclusive"),
        (["supported"], "no_result", "inconclusive"),
    ],
)
def test_interpretations_need_actual_held_back_verdicts(verdicts, classification, expected):
    assert research._classify(assessment(verdicts, classification))[0] == expected


def test_a_critic_can_reject_support_but_cannot_create_a_null():
    assert (
        research._classify(assessment(["supported"], "supported", decision="reject"))[0]
        == "inconclusive"
    )
    for citations in ([], [str(uuid4())], ["foreign", "foreign"]):
        assert (
            research._classify(assessment(["null"], "null", citations=citations))[0]
            == "inconclusive"
        )


@pytest.mark.parametrize(
    "conclusion, critique, expected",
    [
        (claim("supported"), ACCEPT, "inconclusive"),
        ("not json", ACCEPT, "inconclusive"),
        (NO_RESULT, '{"decision":"reject","citations":[],"explanation":"No"}', "inconclusive"),
    ],
)
def test_final_interpretations_keep_both_texts_even_when_the_claim_is_refused(
    temporal, tmp_path, setup, conclusion, critique, expected
):
    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        assert (await handle.result()).outcome == "inconclusive"
        report = await research.record(handle)
        assert report.assessment.classification == expected
        assert load_material(report.assessment.conclusion).text == conclusion
        assert load_material(report.assessment.critique).text == critique
        assert conclusion in texts(run.sent()[-1])
        assert len(run.provider.calls) == 3
        assert "open" not in released(report)

    serve(temporal, tmp_path, setup, [CONCLUDE, conclusion, critique], body)


@pytest.mark.parametrize("positive_claim", [False, True])
def test_a_nonempty_shortlist_is_evaluated_once_with_fixed_multiplicity_and_saved_mapping(
    temporal, tmp_path, searching, positive_claim
):
    revision = pilot(searching, names=SEARCHED)
    search, state = resolve(revision, searching).search, {}

    def script():
        yield propose(SAHM_TEXT)
        summary = json.loads(state["run"].provider.calls[-1].texts[-1].split("\nResult: ")[1])
        state["source"] = summary["trial"]
        yield says("conclude", {"trials": [summary["trial"]]})
        [evidence] = json.loads(state["run"].provider.calls[-1].texts[-2])
        decoded = reference.decode_evaluation(canonical_bytes(evidence["evaluation"]))
        state["verdict"] = decoded.verdict
        yield claim("supported" if positive_claim else decoded.verdict, [evidence["trial"]])
        yield ACCEPT

    async def body(run: Run) -> None:
        state["run"] = run
        handle = await research.start(run.client, revision, run.queue)
        ending = await handle.result()
        report = await research.record(handle)
        if positive_claim:
            assert state["verdict"] != "supported"
        expected = "inconclusive" if positive_claim else state["verdict"]
        assert report.assessment.classification == expected
        assert ending.outcome == ("inconclusive" if expected == "inconclusive" else "concluded")
        assert load_material(report.assessment.critique).text == ACCEPT
        assert report.assessment.conclusion is not None
        [mapping] = report.assessment.trials
        assert mapping.source == state["source"]
        source = reference.open_lineage(search, report.assessment.search_package)
        held = reference.open_lineage(search, report.package)
        assert source.record.trials == held.record.trials[:1]
        assert len(held.record.trials) == 2
        assert str(held.record.trials[-1].trial_id) == mapping.trial
        assert mapping.trial != mapping.source
        assert (
            json.loads(mapping.evaluation)["result"]["trials"]
            == revision.specification.limits.max_trials
            == 5
        )
        assert any(
            event.role == "evaluation"
            for event in ledger(report).events
            if isinstance(event, Charged)
        )
        history = await handle.fetch_history()
        schedules = [
            event.activity_task_scheduled_event_attributes.activity_type.name
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
        ]
        assert schedules.count("evaluate") == 1
        await Replayer(workflows=[research.ResearchLoop]).replay_workflow(history)
        assert len(run.provider.calls) == 4

    serve(temporal, tmp_path, searching, script(), body)


@pytest.mark.parametrize(
    "fallback", [False, True], ids=["explicit-critic", "actual-fallback-author"]
)
def test_final_models_come_from_recorded_active_authority_and_the_actual_shortlist_author(
    temporal, tmp_path, setup, fallback
):
    claude = Provider("claude-subscription")
    policy = ProviderPolicy(provider=claude, labels=frozenset(), retains=True)
    shared = replace(DECLARATION, approved_for=frozenset({CODEX, claude}))
    setup = replace(setup, policies=(CODEX_POLICY, policy))
    revision = pilot(setup)
    prompts = tuple(
        register(
            setup.sources,
            name=name,
            role=PROMPT,
            data=f"You are {name}.".encode(),
            declaration=replace(shared, location=f"prompts/{name}.md"),
        )
        for name in (ROLE, "critic")
    )
    roles = (
        (ModelRole(ROLE, CODEX_MODEL, ("claude-subscription/claude-opus-5-5",), "active"),)
        if fallback
        else (
            ModelRole(ROLE, "claude-subscription/claude-opus-5-5", (), "active"),
            ModelRole("critic", CODEX_MODEL, (), "active"),
        )
    )
    revision = child_of(
        revision,
        sources=tuple(source for source in revision.specification.sources if source.role != PROMPT)
        + prompts,
        model_roles=roles,
    )
    other = ScriptedProvider([CONCLUDE, NO_RESULT])

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert (await handle.result()).outcome == "concluded"
        report = await research.record(handle)
        assert json.loads(report.assessment.proposer)["provider"] == claude
        assert json.loads(report.assessment.critic)["provider"] == CODEX
        assert report.assessment.independent
        records = [entry for entry in run.log.load() if isinstance(entry, CallRecord)]
        assert records[-2].model.provider == claude
        assert records[-1].model.provider == CODEX
        assert records[-1].role == "critic"
        assert len(other.calls) == 2
        assert len(run.provider.calls) == (2 if fallback else 1)
        assert charged(report, "model_calls") == len(records)
        assert "open" not in released(report)

    serve(
        temporal,
        tmp_path,
        setup,
        [Fault.FAIL, ACCEPT] if fallback else [ACCEPT],
        body,
        additional_routes=((policy, other),),
    )


@pytest.mark.parametrize("case", ["unapproved", "policy", "unrelated", "same_provider"])
def test_critique_uses_an_authorized_role_and_preserves_a_same_provider_critic(
    temporal, tmp_path, setup, case
):
    claude = Provider("claude-subscription")
    policy = ProviderPolicy(provider=claude, labels=frozenset(), retains=True)
    labelled = frozenset({VENDOR}) if case == "policy" else frozenset()
    setup = replace(
        setup,
        clearances=labelled,
        policies=(replace(CODEX_POLICY, labels=labelled), policy),
    )
    base = pilot(setup)
    name = "summarizer" if case == "unrelated" else "critic"
    researcher = register(
        setup.sources,
        name=ROLE,
        role=PROMPT,
        data=INSTRUCTIONS.encode(),
        declaration=replace(
            DECLARATION,
            location="prompts/restricted-researcher.md",
            labels=labelled,
            approved_for={CODEX, claude} if case == "policy" else {CODEX},
        ),
    )
    critic = register(
        setup.sources,
        name=name,
        role=PROMPT,
        data=b"x" * BOUND if case == "unrelated" else b"You are the configured critic.",
        declaration=replace(
            DECLARATION, location=f"prompts/{name}.md", approved_for={CODEX, claude}
        ),
    )
    revision = child_of(
        base,
        sources=tuple(item for item in base.specification.sources if item.role != PROMPT)
        + (researcher, critic),
        model_roles=(
            *base.specification.model_roles,
            ModelRole(
                name,
                CODEX_MODEL if case == "same_provider" else "claude-subscription/claude-opus-5-5",
                (),
                "active",
            ),
        ),
    )
    other = ScriptedProvider([])

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert (await handle.result()).outcome == "concluded"
        report = await research.record(handle)
        assert not report.assessment.independent
        assert json.loads(report.assessment.critic)["provider"] == CODEX
        assert texts(run.sent()[-1])[0] == (
            "You are the configured critic." if case == "same_provider" else INSTRUCTIONS
        )
        assert other.calls == []
        assert len(run.provider.calls) == 3
        assert "open" not in released(report)
        await Replayer(workflows=[research.ResearchLoop]).replay_workflow(
            await handle.fetch_history()
        )

    serve(
        temporal,
        tmp_path,
        setup,
        [CONCLUDE, NO_RESULT, ACCEPT],
        body,
        additional_routes=((policy, other),),
    )


def test_large_prompts_are_serialized_once_per_role_with_multiple_models(temporal, tmp_path, setup):
    claude = Provider("claude-subscription")
    policy = ProviderPolicy(provider=claude, labels=frozenset(), retains=True)
    setup = replace(setup, policies=(CODEX_POLICY, policy))
    base = pilot(setup, fallbacks=("claude-subscription/claude-opus-5-5",))
    prompts = tuple(
        register(
            setup.sources,
            name=name,
            role=PROMPT,
            data=b"\\" * 25000,
            declaration=replace(DECLARATION, location=f"prompts/large-{name}.md"),
        )
        for name in (ROLE, "critic")
    )
    revision = child_of(
        base,
        sources=tuple(item for item in base.specification.sources if item.role != PROMPT) + prompts,
        model_roles=(
            *base.specification.model_roles,
            ModelRole("critic", CODEX_MODEL, ("claude-subscription/claude-opus-5-5",), "active"),
        ),
    )

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert (await handle.result()).outcome == "concluded"
        assert len(run.provider.calls) == 3

    serve(
        temporal,
        tmp_path,
        setup,
        [CONCLUDE, NO_RESULT, ACCEPT],
        body,
        additional_routes=((policy, ScriptedProvider([])),),
    )


@pytest.mark.parametrize("stage", ["conclusion", "critique"])
@pytest.mark.parametrize("temporal_cancel", [False, True])
def test_cancellation_during_a_final_call_preserves_evidence_and_completed_text(
    temporal, tmp_path, setup, stage, temporal_cancel
):
    script = (
        [CONCLUDE, Fault.BLOCK] if stage == "conclusion" else [CONCLUDE, NO_RESULT, Fault.BLOCK]
    )
    revision = pilot(setup)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert await asyncio.to_thread(run.provider.blocking.wait, 10)
        from quant_research.__main__ import live_receipts

        observed = await research.observe(handle, client=run.client)
        receipts = live_receipts(observed.ledger, run.log.load(), settled=observed.settled)
        assert receipts.agreed
        assert sorted(check.state for check in receipts.calls) == ["matched"] * (
            len(script) - 1
        ) + ["pending"]
        assert observed.report.assessment.classification == "inconclusive"
        assert (
            observed.report.assessment.proposer
            == (await research.status(handle)).assessment.proposer
        )
        assert observed.report.assessment.evidence is None
        with pytest.raises(WorkflowUpdateFailedError) as refused:
            await research.steer(handle, child_of(revision, question="Another question"))
        assert "closing asks" in str(refused.value.cause)
        if temporal_cancel:
            await handle.cancel()
        else:
            await research.control(handle, "cancel", actor="sundeep", reason="End interpretation")
        assert (await handle.result()).outcome == "cancelled"
        report = await research.record(handle)
        assert report.assessment.evidence is not None
        assert report.assessment.critique is None
        assert (report.assessment.conclusion is not None) == (stage == "critique")
        assert len(run.provider.calls) == len(script)
        assert "open" not in released(report)

    serve(temporal, tmp_path, setup, script, body)


@pytest.mark.parametrize("temporal_cancel", [False, True])
def test_cancel_during_held_back_evaluation_never_repeats_it_or_calls_models(
    temporal, tmp_path, setup, monkeypatch, temporal_cancel
):
    entered, gate = threading.Event(), threading.Event()
    original = reference.assess
    calls = []

    def waiting(*args, **kwargs):
        calls.append(args)
        entered.set()
        gate.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(reference, "assess", waiting)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            if temporal_cancel:
                await handle.cancel()
            else:
                await research.control(handle, "cancel", actor="sundeep", reason="End evaluation")
            assert (await handle.result()).outcome == "cancelled"
            report = await research.record(handle)
            assert report.assessment.conclusion is report.assessment.critique is None
            assert len(calls) == len(run.provider.calls) == 1
            assert "open" not in released(report)
            from quant_research.__main__ import live_receipts

            observed = await research.observe(handle, client=run.client)
            evaluated = {
                event.attempt_id
                for event in observed.ledger.events
                if isinstance(event, Charged) and event.role == "evaluation"
            }
            assert evaluated and not (evaluated & observed.settled)
            assert len(observed.settled) == 1
            assert live_receipts(observed.ledger, run.log.load(), settled=observed.settled).agreed
            missing = live_receipts(observed.ledger, (), settled=observed.settled)
            assert missing.unclosed == tuple(map(str, sorted(observed.settled)))
            await Replayer(workflows=[research.ResearchLoop]).replay_workflow(
                await handle.fetch_history()
            )
        finally:
            gate.set()

    serve(temporal, tmp_path, setup, [CONCLUDE], body)


@pytest.mark.parametrize(
    "failure", ["second_measurement", "lost_return", "control_cancel", "temporal_cancel", "timeout"]
)
def test_a_failed_evaluation_retains_published_evidence_and_a_key_for_late_saves(
    temporal, tmp_path, searching, monkeypatch, failure
):
    revision = pilot(searching, names=SEARCHED)
    search = resolve(revision, searching).search
    entered, gate, finished = threading.Event(), threading.Event(), threading.Event()
    state, measurements, dispatches = {}, [], []
    examine, assess, append = reference._examine, reference.assess, reference._append
    if failure == "timeout":
        monkeypatch.setattr(research, "EXPERIMENT_TIMEOUT", timedelta(seconds=5))

    def measure(*args, **kwargs):
        if args[1].digest == RULES.held_back.digest:
            measurements.append(args)
            if len(measurements) == 2:
                if failure == "second_measurement":
                    raise RuntimeError("second measurement failed")
                if failure in {"control_cancel", "temporal_cancel", "timeout"}:
                    entered.set()
                    assert gate.wait(15)
        return examine(*args, **kwargs)

    def evaluate(*args, **kwargs):
        dispatches.append(args)
        result = assess(*args, **kwargs)
        if failure == "lost_return":
            raise RuntimeError("final evaluation return lost")
        return result

    def publish(*args, **kwargs):
        package = append(*args, **kwargs)
        if len(package.record.trials) == 4:
            finished.set()
        return package

    monkeypatch.setattr(reference, "_examine", measure)
    monkeypatch.setattr(reference, "assess", evaluate)
    monkeypatch.setattr(reference, "_append", publish)

    def script():
        sources = []
        for program in (SAHM_TEXT, RISE_TEXT):
            yield propose(program)
            summary = json.loads(state["run"].provider.calls[-1].texts[-1].split("\nResult: ")[1])
            sources.append(summary["trial"])
        yield says("conclude", {"trials": sources})

    async def body(run: Run) -> None:
        state["run"] = run
        handle = await research.start(run.client, revision, run.queue)
        try:
            if failure in {"control_cancel", "temporal_cancel", "timeout"}:
                assert await asyncio.to_thread(entered.wait, 10)
                if failure == "control_cancel":
                    await research.control(
                        handle, "cancel", actor="sundeep", reason="End evaluation"
                    )
                elif failure == "temporal_cancel":
                    await handle.cancel()
            ending = await handle.result()
            report, compact = await research.record(handle), await research.status(handle)
            cancelled = failure in {"control_cancel", "temporal_cancel"}
            assert ending.outcome == ("cancelled" if cancelled else "inconclusive")
            assert report.assessment.classification == "inconclusive"
            assert report.assessment.conclusion is report.assessment.critique is None
            assert report.assessment.recovery == compact.assessment.recovery is not None
            assert len(report.assessment.trials) == 2
            assert len(dispatches) == 1
            assert len(run.provider.calls) == 3
            assert "open" not in released(report)
            recovered = reference.recover_assessment(search, report.assessment.recovery)
            completed = 2 if failure == "lost_return" else 1
            assert len(json.loads(recovered.summary)) == completed
            if cancelled:
                assert report.assessment.evidence is None
                assert {item.status for item in report.assessment.trials} == {"pending"}
            else:
                assert report.assessment.package == report.package == recovered.package
                assert load_material(report.assessment.evidence) == Material(
                    recovered.summary, recovered.restrictions
                )
                assert (
                    sum(item.status == "pending" for item in report.assessment.trials)
                    == 2 - completed
                )
            if failure in {"control_cancel", "temporal_cancel", "timeout"}:
                gate.set()
                assert await asyncio.to_thread(finished.wait, 10)
                later = reference.recover_assessment(search, report.assessment.recovery)
                assert len(json.loads(later.summary)) == 2
                assert later.package != recovered.package
                assert (await research.record(handle)).assessment == report.assessment
                from quant_research.packages import open_package
                from quant_research.report import campaign_snapshot

                [path] = (tmp_path / "archive" / "packages" / "unselected").iterdir()
                first = open_package(path)
                archive = research.CampaignArchive(tmp_path / "archive", tmp_path / "calls")
                closed = research.ClosedCampaign(
                    first.read_artifact("campaign-original-report"),
                    (file_bytes(revision),),
                    first.read_artifact("campaign-history"),
                )
                same = archive.publish(closed, key=campaign_snapshot(first)["key"])
                assert same.digest == first.digest
                refreshed = archive.publish(closed, key="explicit-late-recovery")
                assert refreshed.digest != first.digest
                assert len(refreshed.record.trials) == 4
                assert campaign_snapshot(refreshed)["recovered_head"] == later.package
                assert refreshed.read_artifact("campaign-original-report") == closed.report
                assert refreshed.read_artifact("campaign-ledger") == first.read_artifact(
                    "campaign-ledger"
                )
                assert (
                    campaign_snapshot(first)["report"]["assessment"]
                    == campaign_snapshot(refreshed)["report"]["assessment"]
                )
            history = await handle.fetch_history()
            schedules = [
                event.activity_task_scheduled_event_attributes
                for event in history.events
                if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
            ]
            [evaluation] = [item for item in schedules if item.activity_type.name == "evaluate"]
            assert evaluation.retry_policy.maximum_attempts == 1
            assert (
                evaluation.start_to_close_timeout.ToTimedelta() == research.EXPERIMENT_TIMEOUT * 2
            )
            assert sum(
                item.activity_type.name == "recover_assessment" for item in schedules
            ) == int(not cancelled)
            await Replayer(
                workflows=[research.ResearchLoop], workflow_runner=UnsandboxedWorkflowRunner()
            ).replay_workflow(history)
            assert len(dispatches) == 1
            assert len(run.provider.calls) == 3
        finally:
            gate.set()

    serve(temporal, tmp_path, searching, script(), body, unsandboxed=True)


@pytest.mark.parametrize("stage", ["conclusion", "critique"])
def test_a_lost_final_call_record_keeps_numerical_evidence_and_charges_unknown_usage(
    temporal, tmp_path, searching, monkeypatch, stage
):
    revision = pilot(searching, names=SEARCHED)
    state = {}
    append = CallLog.append

    def lose_record(log, entry):
        if state.get("lose") and isinstance(entry, CallRecord) and entry.outcome == "completed":
            state["lose"] = False
            state["lost"] = entry.call_id
            raise RuntimeError("completed final call record lost")
        return append(log, entry)

    monkeypatch.setattr(CallLog, "append", lose_record)

    def script():
        yield propose(SAHM_TEXT)
        summary = json.loads(state["run"].provider.calls[-1].texts[-1].split("\nResult: ")[1])
        yield says("conclude", {"trials": [summary["trial"]]})
        [evidence] = json.loads(state["run"].provider.calls[-1].texts[-2])
        result = reference.decode_evaluation(canonical_bytes(evidence["evaluation"]))
        state["conclusion"] = claim(result.verdict, [evidence["trial"]])
        state["lose"] = stage == "conclusion"
        yield state["conclusion"]
        state["lose"] = stage == "critique"
        yield ACCEPT

    async def body(run: Run) -> None:
        state["run"] = run
        handle = await research.start(run.client, revision, run.queue)
        assert (await handle.result()).outcome == "inconclusive"
        report = await research.record(handle)
        assert report.assessment.classification == "inconclusive"
        [mapping] = report.assessment.trials
        assert mapping.status != "pending"
        assert mapping.evaluation is not None
        assert report.assessment.evidence is not None
        assert report.assessment.critique is None
        if stage == "critique":
            assert load_material(report.assessment.conclusion).text == state["conclusion"]
        else:
            assert report.assessment.conclusion is None
        [unknown] = [
            entry
            for entry in run.log.load()
            if isinstance(entry, CallRecord) and entry.call_id == state["lost"]
        ]
        assert unknown.outcome == "unknown"
        assert unknown.usage.model_calls == 1
        assert unknown.usage.unreported == {"input_tokens", "output_tokens", "cost_usd"}
        [charge] = [
            event
            for event in ledger(report).events
            if isinstance(event, Charged) and event.id == unknown.call_id
        ]
        assert charge.usage == unknown.usage
        expected = 3 if stage == "conclusion" else 4
        assert charged(report, "model_calls") == len(run.provider.calls) == expected
        assert set(report.spent["unreported"].split()) == unknown.usage.unreported
        assert "open" not in released(report)
        history = await handle.fetch_history()
        assert (
            sum(
                event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
                and event.activity_task_scheduled_event_attributes.activity_type.name == "evaluate"
                for event in history.events
            )
            == 1
        )
        await Replayer(workflows=[research.ResearchLoop]).replay_workflow(history)
        assert len(run.provider.calls) == expected

    serve(temporal, tmp_path, searching, script(), body)


@pytest.mark.parametrize(
    "stage, fault",
    [
        ("conclusion", Fault.FAIL),
        ("conclusion", Fault.RATE_LIMIT),
        ("critique", Fault.FAIL),
        ("critique", Fault.RATE_LIMIT),
    ],
)
def test_a_failed_final_call_ends_inconclusive_without_retry_or_rate_limit_pause(
    temporal, tmp_path, setup, stage, fault
):
    script = [CONCLUDE, fault] if stage == "conclusion" else [CONCLUDE, NO_RESULT, fault]

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        ending = await handle.result()
        report = await research.record(handle)
        assert ending.outcome == report.assessment.classification == "inconclusive"
        assert len(run.provider.calls) == len(script)
        assert steering(report) == []
        assert "open" not in released(report)

    serve(temporal, tmp_path, setup, script, body)


def test_a_budget_closing_shortlist_keeps_the_budget_outcome_after_both_fresh_calls(
    temporal, tmp_path, setup
):
    revision = pilot(setup, limits=ResourceLimits(5, 2, 100, max_model_calls=4))

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        ending = await handle.result()
        report = await research.record(handle)
        assert (ending.outcome, ending.reason) == ("inconclusive", "model_calls")
        assert report.assessment.classification == "inconclusive"
        assert report.assessment.reason == "model_calls"
        assert load_material(report.assessment.conclusion).text == NO_RESULT
        assert load_material(report.assessment.critique).text == ACCEPT
        assert charged(report, "model_calls") == len(run.provider.calls) == 4
        assert len(report.steps) == 2
        assert "open" not in released(report)

    serve(temporal, tmp_path, setup, [READ, CONCLUDE, NO_RESULT, ACCEPT], body)


def test_required_final_material_is_never_silently_dropped_to_fit_the_critic(
    temporal, tmp_path, setup, monkeypatch
):
    monkeypatch.setattr(research, "BOUND", 32 * 2**10)
    large = json.dumps(
        {"classification": "no_result", "citations": [], "explanation": "x" * (40 * 2**10)}
    )

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        ending = await handle.result()
        report = await research.record(handle)
        assert (ending.outcome, ending.reason) == ("inconclusive", "prompt")
        assert load_material(report.assessment.conclusion).text == large
        assert report.assessment.critique is None
        assert len(run.provider.calls) == 2
        assert "Mandatory critique" in report.assessment.detail
        assert "open" not in released(report)

    serve(temporal, tmp_path, setup, [CONCLUDE, large], body, unsandboxed=True)


@pytest.mark.parametrize("stage, calls", [("selection", 1), ("conclusion", 2), ("critique", 3)])
def test_a_completed_reply_racing_cancel_is_kept_without_scheduling_the_next_stage(
    temporal, tmp_path, setup, monkeypatch, stage, calls
):
    original = research.ask_model

    async def racing(revision, assignment, request, attempt_id, **kwargs):
        asked = await original(revision, assignment, request, attempt_id, **kwargs)
        current = (
            "selection"
            if request.schema == SCHEMA
            else "conclusion"
            if texts(request)[-1] == research.CONCLUSION
            else "critique"
        )
        if current == stage:
            workflow.instance().control("cancel", "sundeep", "Cancel arrived with reply", None)
        return asked

    monkeypatch.setattr(research, "ask_model", racing)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        assert (await handle.result()).outcome == "cancelled"
        report = await research.record(handle)
        assert len(run.provider.calls) == calls
        assert (report.assessment.conclusion is not None) == (calls >= 2)
        assert (report.assessment.critique is not None) == (calls == 3)
        assert "open" not in released(report)
        history = await handle.fetch_history()
        scheduled = [
            event.activity_task_scheduled_event_attributes.activity_type.name
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
        ]
        assert scheduled.count("evaluate") == (0 if stage == "selection" else 1)

    serve(temporal, tmp_path, setup, [CONCLUDE, NO_RESULT, ACCEPT], body, unsandboxed=True)


@pytest.mark.parametrize("invalid", ["duplicate", "foreign", "extra-field"])
def test_an_invalid_shortlist_never_reads_held_back_data_or_asks_for_interpretation(
    temporal, tmp_path, setup, invalid
):
    state = {}

    def script():
        yield propose(LEAKY_TEXT)
        source = json.loads(state["run"].provider.calls[-1].texts[-1].split("\nResult: ")[1])[
            "trial"
        ]
        selection = (
            {"trials": [source, source]}
            if invalid == "duplicate"
            else {"trials": [str(uuid4())]}
            if invalid == "foreign"
            else {"trials": [source], "claim": "supported"}
        )
        yield says("conclude", selection)

    async def body(run: Run) -> None:
        state["run"] = run
        handle = await research.start(run.client, pilot(setup), run.queue)
        ending = await handle.result()
        report = await research.record(handle)
        assert (ending.outcome, ending.reason) == ("inconclusive", "shortlist")
        assert report.assessment.conclusion is report.assessment.critique is None
        assert report.assessment.package == report.assessment.search_package
        assert len(run.provider.calls) == 2
        assert "open" not in released(report)
        history = await handle.fetch_history()
        assert all(
            event.activity_task_scheduled_event_attributes.activity_type.name != "evaluate"
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
        )

    serve(temporal, tmp_path, setup, script(), body)


def test_an_oversized_active_critic_prompt_is_refused_before_opening_crosses_a_payload(
    temporal, tmp_path, setup
):
    base = pilot(setup)
    critic = register(
        setup.sources,
        name="critic",
        role=PROMPT,
        data=b"c" * BOUND,
        declaration=replace(DECLARATION, location="prompts/critic.md"),
    )
    revision = child_of(
        base,
        sources=(*base.specification.sources, critic),
        model_roles=(
            *base.specification.model_roles,
            ModelRole("critic", CODEX_MODEL, (), "active"),
        ),
    )

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert await handle.result() == Ending("inconclusive", "prompt", 0, None)
        assert run.provider.calls == []

    serve(temporal, tmp_path, setup, [], body)


def test_a_pause_after_the_fresh_conclusion_holds_critique_until_resume_without_reevaluation(
    temporal, tmp_path, setup, monkeypatch
):
    original = research.ask_model

    async def paused(revision, assignment, request, attempt_id, **kwargs):
        asked = await original(revision, assignment, request, attempt_id, **kwargs)
        if texts(request)[-1] == research.CONCLUSION:
            workflow.instance().control("pause", "sundeep", "Review held-back conclusion", None)
        return asked

    monkeypatch.setattr(research, "ask_model", paused)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        held = await reaches(handle, "paused")
        assert held.paused_by == "sundeep: Review held-back conclusion"
        assert len(run.provider.calls) == 2
        await asyncio.sleep(0.2)
        assert len(run.provider.calls) == 2
        await research.control(handle, "resume", actor="sundeep", reason="Continue critique")
        assert (await handle.result()).outcome == "concluded"
        report = await research.record(handle)
        assert load_material(report.assessment.conclusion).text == NO_RESULT
        assert load_material(report.assessment.critique).text == ACCEPT
        assert len(run.provider.calls) == 3
        history = await handle.fetch_history()
        schedules = [
            event.activity_task_scheduled_event_attributes.activity_type.name
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
        ]
        assert schedules.count("evaluate") == 1

    serve(temporal, tmp_path, setup, [CONCLUDE, NO_RESULT, ACCEPT], body, unsandboxed=True)


def test_a_scripted_campaign_searches_reads_pauses_resumes_and_ends_at_its_dispatch_limit(
    temporal, tmp_path, setup
):
    limits = ResourceLimits(5, 2, 100, max_model_calls=6)
    revision = pilot(setup, "checkpoint every 2 steps", limits=limits)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        paused = await reaches(handle, "paused")
        assert paused.paused_by == "runner: Checkpoint every 2 steps after step 2"
        assert [(step.action, step.result) for step in paused.steps] == [
            ("search_corpus", "ran"),
            ("read_dataset", "ran"),
        ]
        await asyncio.sleep(0.5)
        assert len(run.provider.calls) == 2
        assert (
            await research.control(
                handle, "resume", actor="sundeep", reason="Both reads look right"
            )
            == "running"
        )
        ending = await handle.result()
        report = await research.record(handle)
        assert (ending.outcome, ending.reason, ending.steps) == ("inconclusive", "model_calls", 4)
        assert load_material(ending.final).text == STOP
        assert report.ending == ending
        assert [(step.action, step.result) for step in report.steps] == [
            ("search_corpus", "ran"),
            ("read_dataset", "ran"),
            ("read_dataset", "ran"),
            ("stop", "stopped"),
        ]
        assert steering(report) == [
            ("pause", "runner", "Checkpoint every 2 steps after step 2"),
            ("resume", "sundeep", "Both reads look right"),
        ]
        assert charged(report, "model_calls") == len(run.provider.calls) == 4
        assert released(report) == [
            "successful",
            "inconclusive",
            "inconclusive",
            "successful",
            "successful",
            "successful",
        ]
        first, second, third, closing = run.sent()
        assert texts(first)[2:] == []
        assert texts(third)[2].startswith(f"Step 1: {SEARCH}\nResult: ")
        assert texts(third)[3].startswith(f"Step 2: {READ}\nResult: ")
        assert '"dataset":"alfred.unrate"' in texts(third)[3]
        assert texts(closing)[-1] == CLOSING

    serve(temporal, tmp_path, setup, [SEARCH, READ, READ, STOP], body)


def test_the_step_cap_ends_the_loop_through_one_closing_ask_and_bounds_the_status(
    temporal, tmp_path, setup
):
    experiment = says("experiment", {"program": "momentum"})
    revision = pilot(setup, limits=ResourceLimits(1, 2, 600, max_model_calls=2 * STEP_CAP + 2))

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        ending = await handle.result()
        status, report = await research.status(handle), await research.record(handle)
        # Measured: a 31965-byte status, and a 146229-byte record read in one part.
        assert len(DataConverter.default.payload_converter.to_payload(status).data) < 2**19
        assert report.steps == status.steps
        assert (ending.outcome, ending.reason, ending.steps) == (
            "inconclusive",
            "steps",
            STEP_CAP + 1,
        )
        assert load_material(ending.final).text == STOP
        assert len(status.steps) == len(run.provider.calls) == STEP_CAP + 1
        assert [step.result for step in status.steps] == (
            ["rejected"] + ["refused"] * (STEP_CAP - 1) + ["stopped"]
        )
        assert [step.trial is not None for step in status.steps].count(True) == 1
        spent = research.TRIALS_SPENT.format(limit=1)
        assert texts(run.sent()[2])[-1] == f"Step 2: {experiment}\nResult: {spent}"

    serve(temporal, tmp_path, setup, [experiment] * STEP_CAP + [STOP], body)


def test_the_history_guard_ends_the_loop_through_a_closing_ask_and_replays(
    temporal, tmp_path, setup, monkeypatch
):
    # Measured: this history passes 50 KB during the third read step, so the fourth ask closes.
    monkeypatch.setattr(research, "HISTORY_BYTES", 50_000)
    revision = pilot(setup)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        ending = await handle.result()
        assert (ending.outcome, ending.reason) == ("inconclusive", "history")
        assert ending.steps == len(run.provider.calls) == 4
        assert load_material(ending.final).text == STOP
        assert texts(run.sent()[-1])[-1] == CLOSING
        replayer = Replayer(
            workflows=[research.ResearchLoop], workflow_runner=UnsandboxedWorkflowRunner()
        )
        await replayer.replay_workflow(await handle.fetch_history())

    serve(temporal, tmp_path, setup, [READ] * 3 + [STOP], body, unsandboxed=True)


@pytest.mark.parametrize("move, outcome", [(CONCLUDE, "concluded"), (STOP, "stopped")])
def test_a_conclusion_or_a_stop_ends_the_loop_with_its_reply_as_restricted_material(
    temporal, tmp_path, setup, move, outcome
):
    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        ending = await handle.result()
        report = await research.record(handle)
        request, *_ = run.sent()
        assert (ending.outcome, ending.reason, ending.steps) == (outcome, "", 1)
        assert load_material(ending.final) == Material(
            NO_RESULT if outcome == "concluded" else move, request.restrictions
        )
        assert [(step.action, step.result) for step in report.steps] == [
            (json.loads(move)["action"], outcome)
        ]
        assert released(report) == [
            "inconclusive",
            *(
                ["successful", "successful"]
                if outcome == "concluded"
                else ["inconclusive", "inconclusive"]
            ),
            "successful",
        ]
        assert report.state == outcome

    serve(
        temporal,
        tmp_path,
        setup,
        [move, *([NO_RESULT, ACCEPT] if outcome == "concluded" else [])],
        body,
    )


def test_a_tool_result_and_an_unknown_tool_refusal_reach_the_next_request(
    temporal, tmp_path, setup
):
    unknown = says("no_such_tool")
    experiment = says("experiment", {"program": "momentum"})
    revision = pilot(setup)
    toolset = resolve(revision, setup).toolset
    read = toolset.call("read_dataset", '{"dataset": "alfred.unrate"}').material.text
    refusal = toolset.call("no_such_tool", "{}").material.text

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        ending = await handle.result()
        report = await research.record(handle)
        _, after_read, after_unknown, after_experiment = run.sent()
        assert texts(after_read)[-1] == f"Step 1: {READ}\nResult: {read}"
        assert '"refusal":"unknown_tool"' in refusal
        assert texts(after_unknown)[-1] == f"Step 2: {unknown}\nResult: {refusal}"
        trial = report.steps[2].trial
        assert texts(after_experiment)[-1] == f"Step 3: {experiment}\nResult: {momentum(trial)}"
        assert [(step.action, step.result) for step in report.steps] == [
            ("read_dataset", "ran"),
            ("unknown", "ran"),
            ("experiment", "rejected"),
            ("stop", "stopped"),
        ]
        shown = [step.digest for step in report.steps]
        assert shown[:3] == [
            hashlib.sha256(json.dumps(text).encode()).hexdigest()
            for text in texts(after_experiment)[2:]
        ]
        assert shown[3] == ""
        assert ending.outcome == "stopped"
        roles = [event.role for event in ledger(report).events if isinstance(event, Charged)]
        assert (roles.count("tools"), roles.count("experiments")) == (2, 1)

    serve(temporal, tmp_path, setup, [READ, unknown, experiment, STOP], body)


@pytest.mark.parametrize(
    "limits, script, reason",
    [
        (
            ResourceLimits(5, 2, 100, max_model_calls=20, max_wall_seconds=2400),
            [READ, STOP],
            "wall_seconds",
        ),
        (
            ResourceLimits(5, 2, 100, max_model_calls=20, max_input_tokens=1),
            [READ, STOP],
            "input_tokens",
        ),
        (
            ResourceLimits(5, 2, 100, max_model_calls=20, max_output_tokens=1),
            [READ, STOP],
            "output_tokens",
        ),
        (
            ResourceLimits(5, 2, 100, max_model_calls=20),
            [Fault.FAIL, Fault.FAIL, STOP],
            "failures_per_run",
        ),
    ],
    ids=["wall", "input-tokens", "output-tokens", "attempts"],
)
def test_each_limit_ends_inconclusive_through_one_closing_ask(
    temporal, tmp_path, setup, limits, script, reason
):
    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup, limits=limits), run.queue)
        ending = await handle.result()
        report = await research.record(handle)
        assert (ending.outcome, ending.reason) == ("inconclusive", reason)
        assert load_material(ending.final).text == STOP
        assert len(run.provider.calls) == len(script) == ending.steps
        assert texts(run.sent()[-1])[-1] == CLOSING
        assert "open" not in released(report)
        if reason == "failures_per_run":
            assert [step.result for step in report.steps] == ["failed", "failed", "stopped"]
            first, second, closing = run.sent()
            assert texts(first) == texts(second) == texts(closing)[:-1]

    serve(temporal, tmp_path, setup, script, body)


def test_tool_compute_is_charged_as_observed_and_refuses_the_next_ask_once_spent(
    temporal, tmp_path, setup
):
    @dataclass(frozen=True)
    class Slow(Toolset):
        def call(self, tool: str, arguments: str):
            time.sleep(1.2)
            return super().call(tool, arguments)

    def authorize(revision: CampaignRevision):
        authority = resolve(revision, setup)
        tools = authority.toolset
        return replace(authority, toolset=Slow(**{name: getattr(tools, name) for name in TOOLSET}))

    revision = pilot(setup, limits=ResourceLimits(5, 2, 1, max_model_calls=20))

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        ending = await handle.result()
        report = await research.record(handle)
        assert (ending.outcome, ending.reason, ending.steps) == (
            "inconclusive",
            "compute_seconds",
            2,
        )
        assert charged(report, "compute_seconds") >= Decimal("1.2")
        assert [step.action for step in report.steps] == ["read_dataset", "stop"]

    serve(temporal, tmp_path, setup, [READ, STOP], body, authorize=authorize)


def test_a_checkpoint_before_conclude_holds_the_reply_and_a_resume_ends_the_loop(
    temporal, tmp_path, setup
):
    async def body(run: Run) -> None:
        handle = await research.start(
            run.client, pilot(setup, "checkpoint before conclude"), run.queue
        )
        held = await reaches(handle, "paused")
        assert held.paused_by == "runner: Checkpoint before conclude holds step 1"
        assert held.steps == ()
        await research.control(handle, "resume", actor="sundeep", reason="The finding stands")
        ending = await handle.result()
        assert (ending.outcome, ending.steps) == ("concluded", 1)
        assert len(run.provider.calls) == 3

    serve(temporal, tmp_path, setup, [CONCLUDE, NO_RESULT, ACCEPT], body)


def test_a_held_conclusion_is_set_aside_when_declared_text_arrives_during_the_hold(
    temporal, tmp_path, setup
):
    async def body(run: Run) -> None:
        handle = await research.start(
            run.client, pilot(setup, "checkpoint before conclude"), run.queue
        )
        await reaches(handle, "paused")
        note = "Check the 2008 vintage before concluding"
        await research.control(handle, "annotate", actor="sundeep", reason=note, restrictions=NOTED)
        await research.control(handle, "resume", actor="sundeep", reason="Look again")
        held = await reaches(handle, "paused")
        assert held.paused_by == "runner: Checkpoint before conclude holds step 2"
        await research.control(handle, "resume", actor="sundeep", reason="Now it stands")
        ending = await handle.result()
        report = await research.record(handle)
        assert (ending.outcome, ending.steps) == ("concluded", 2)
        assert [step.result for step in report.steps] == ["set_aside", "concluded"]
        assert texts(run.sent()[1])[2:] == [
            f"Researcher annotate: {note}",
            f"Step 1: {CONCLUDE}\nResult: {research.SET_ASIDE}",
        ]

    serve(temporal, tmp_path, setup, [CONCLUDE, CONCLUDE, NO_RESULT, ACCEPT], body)


def test_a_rate_limited_ask_pauses_the_loop_and_a_resume_asks_again(temporal, tmp_path, setup):
    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        paused = await reaches(handle, "paused")
        assert paused.paused_by == "runner: Rate limited on step 1"
        assert [(step.action, step.result) for step in paused.steps] == [("", "rate_limited")]
        await research.control(handle, "resume", actor="sundeep", reason="The cap reset")
        ending = await handle.result()
        report = await research.record(handle)
        assert ending.outcome == "concluded"
        limited, answered = report.steps
        assert limited.attempt != answered.attempt
        assert released(report) == [
            "inconclusive",
            "successful",
            "successful",
            "inconclusive",
            "successful",
        ]
        first, second, *_ = run.sent()
        assert first.to_bytes() == second.to_bytes()

    serve(temporal, tmp_path, setup, [Fault.RATE_LIMIT, CONCLUDE, NO_RESULT, ACCEPT], body)


def test_requests_carry_every_materials_restrictions_and_withhold_undeclared_steering_text(
    temporal, tmp_path, setup
):
    revision = pilot(setup, "checkpoint every 1 steps")
    experiment = says("experiment", {"program": "momentum"})
    prompted = resolve(revision, setup).prompts[ROLE].restrictions
    read = resolve(revision, setup).toolset.call("read_dataset", '{"dataset": "alfred.unrate"}')
    # Later than the prompt, so it shows in the derived restrictions; the read is available at
    # the search cutoff, before both.
    noted = replace(NOTED, available_at=datetime(2025, 3, 1, tzinfo=UTC))

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        await reaches(handle, "paused")
        for action in ("annotate", "pause"):
            await research.control(handle, action, actor="sundeep", reason=f"UNDECLARED {action}")
        await research.control(
            handle, "annotate", actor="sundeep", reason="Weigh the revisions", restrictions=noted
        )
        await research.control(handle, "resume", actor="sundeep", reason="UNDECLARED resume")
        await reaches(handle, "paused")
        await research.control(handle, "resume", actor="sundeep", reason="UNDECLARED again")
        await handle.result()
        report = await research.record(handle)
        first, second, third = run.sent()
        assert not any("UNDECLARED" in text for request in run.sent() for text in texts(request))
        assert texts(second)[2:] == [
            f"Step 1: {experiment}\nResult: {momentum(report.steps[0].trial)}",
            "Researcher annotate: Weigh the revisions",
        ]
        assert texts(third)[4] == f"Step 2: {READ}\nResult: {read.material.text}"
        assert first.restrictions == prompted
        assert second.restrictions == derive(prompted, noted) != prompted
        assert third.restrictions == derive(prompted, noted, read.restrictions)
        assert read.restrictions.available_at == RULES.search.cutoff

    serve(temporal, tmp_path, setup, [experiment, READ, STOP], body)


@pytest.mark.parametrize(
    "change, named",
    [
        ({"available_at": CUTOFF + timedelta(days=1)}, "after the run's cutoff"),
        ({"labels": frozenset({VENDOR})}, "more than the role's providers receive"),
        ({"approved_for": frozenset()}, "must be approved for"),
    ],
    ids=["late", "labelled", "unapproved"],
)
def test_a_declaration_outside_the_floor_is_refused_and_writes_nothing(
    temporal, tmp_path, setup, change, named
):
    revision = pilot(setup, "checkpoint every 1 steps")
    outside = replace(NOTED, **change)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        paused = await reaches(handle, "paused")
        with pytest.raises(WorkflowUpdateFailedError) as annotated:
            await research.control(
                handle, "annotate", actor="sundeep", reason="Note", restrictions=outside
            )
        with pytest.raises(WorkflowUpdateFailedError) as steered:
            await research.steer(
                handle, child_of(revision, question="Narrower?"), restrictions=outside
            )
        for refused in (annotated, steered):
            assert named in refused.value.cause.message
        assert await research.status(handle) == paused
        await research.control(handle, "cancel", actor="sundeep", reason="Done")
        report = await research.record(handle)
        assert [action for action, _, _ in steering(report)] == ["pause", "cancel"]

    serve(temporal, tmp_path, setup, [READ], body)


def test_refused_controls_and_steers_write_nothing_and_fail_no_workflow_task(
    temporal, tmp_path, setup
):
    revision = pilot(setup, "checkpoint every 1 steps")
    limits = replace(revision.specification.limits, max_model_calls=30)
    rules = EvaluationRules("pilot-evaluation/v2", hashlib.sha256(b"other rules\n").hexdigest())
    inactive = (ModelRole(ROLE, CODEX_MODEL, (), "inactive"),)
    longer = (ModelRole(ROLE, CODEX_MODEL, (CODEX_MODEL,), "active"),)
    refusals = [
        (
            {"action": "resume", "actor": "runner", "reason": "Not the researcher"},
            "reserved for the research loop",
        ),
        (
            {"action": "annotate", "actor": "sundeep", "reason": "x" * (research.NOTE_BYTES + 1)},
            "at most",
        ),
        (
            {
                "action": "annotate",
                "actor": "sundeep",
                "reason": "Note",
                "restrictions": replace(
                    NOTED, approved_for=NOTED.approved_for | {f"other-{n}" for n in range(400)}
                ),
            },
            "at most",
        ),
    ]
    children = [
        (child_of(revision, limits=limits), "keeps its starting ['limits']"),
        (child_of(revision, evaluation=rules), "keeps its starting ['evaluation']"),
        (child_of(revision, interventions=("before_trial",)), "neither"),
        (child_of(revision, model_roles=inactive), "cannot be opened: role"),
        (child_of(revision, model_roles=longer), "chain of 2 passes the 1 held"),
        (child_of(pilot(setup), question="Narrower?"), "revises a revision the loop lacks"),
    ]

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        paused = await reaches(handle, "paused")
        for arguments, named in refusals:
            with pytest.raises(WorkflowUpdateFailedError) as refused:
                await research.control(handle, **arguments)
            assert named in refused.value.cause.message
        with pytest.raises(WorkflowUpdateFailedError) as damaged:
            await handle.execute_update(
                research.ResearchLoop.control, args=["annotate", "sundeep", "Note", "{}"]
            )
        assert "Invalid declaration" in damaged.value.cause.message
        for child, named in children:
            with pytest.raises(WorkflowUpdateFailedError) as refused:
                await research.steer(handle, child)
            assert named in refused.value.cause.message
        assert await research.status(handle) == paused
        await research.control(handle, "resume", actor="sundeep", reason="Carry on")
        assert (await handle.result()).outcome == "stopped"
        report = await research.record(handle)
        assert [action for action, _, _ in steering(report)] == ["pause", "resume"]
        failed = EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED
        assert all(event.event_type != failed for event in (await handle.fetch_history()).events)

    serve(temporal, tmp_path, setup, [READ, STOP], body)


def test_a_steer_runs_later_asks_under_the_child_alone_and_the_history_replays(
    temporal, tmp_path, setup
):
    parent = pilot(setup, "checkpoint every 1 steps")
    narrowed = "You are the researcher on the delayed rule. Read before you conclude.\n"
    brief, _, *contracts = parent.specification.sources
    child = child_of(
        parent,
        "Ask about the delayed rule",
        question="Does the rule survive a one-month delay?",
        sources=(brief, prompt_source(setup, narrowed, "2"), *contracts),
        interventions=("checkpoint before conclude",),
    )

    async def body(run: Run) -> None:
        handle = await research.start(run.client, parent, run.queue)
        await reaches(handle, "paused")
        assert await research.steer(handle, child, restrictions=NOTED) == "paused"
        await research.control(
            handle, "annotate", actor="sundeep", reason="Mind the delay", restrictions=NOTED
        )
        await research.control(handle, "resume", actor="sundeep", reason="Go on")
        limited = await reaches(handle, "paused")
        assert limited.paused_by == "runner: Rate limited on step 2"
        await research.control(handle, "resume", actor="sundeep", reason="Try again")
        held = await reaches(handle, "paused")
        assert held.paused_by == "runner: Checkpoint before conclude holds step 4"
        assert held.revision == child.revision_id
        assert held.authority == hashlib.sha256(resolve(child, setup).record()).hexdigest()
        await research.control(handle, "resume", actor="sundeep", reason="Done")
        assert (await handle.result()).outcome == "stopped"
        report = await research.record(handle)
        before, steered, again, stopped = run.sent()
        assert texts(before)[0] == INSTRUCTIONS
        assert texts(steered)[0] == narrowed
        assert texts(steered)[2:] == [
            "Researcher steer: Ask about the delayed rule",
            "Researcher annotate: Mind the delay",
        ]
        assert again.to_bytes() == steered.to_bytes()
        assert texts(stopped)[4].startswith(f"Step 3: {READ}")
        assert [step.revision for step in report.steps] == [parent.revision_id] + [
            child.revision_id
        ] * 3
        assert [action for action, _, _ in steering(report)] == [
            "pause",
            "steer",
            "annotate",
            "resume",
            "pause",
            "resume",
            "pause",
            "resume",
        ]
        await Replayer(workflows=[research.ResearchLoop]).replay_workflow(
            await handle.fetch_history()
        )

    serve(temporal, tmp_path, setup, [READ, Fault.RATE_LIMIT, READ, STOP], body)


def test_a_temporal_cancel_while_paused_is_recorded_and_ends_the_loop(temporal, tmp_path, setup):
    revision = pilot(setup, "checkpoint every 1 steps")

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        await reaches(handle, "paused")
        await handle.cancel()
        assert await handle.result() == Ending("cancelled", "workflow", 1, None)
        report = await research.record(handle)
        assert steering(report)[-1] == (
            "cancel",
            "runner",
            "The workflow was cancelled after step 1",
        )
        assert released(report) == ["inconclusive", "inconclusive", "inconclusive", "successful"]

    serve(temporal, tmp_path, setup, [READ], body)


def test_a_researcher_cancel_during_an_ask_ends_the_loop_with_no_closing_ask(
    temporal, tmp_path, setup
):
    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        assert await asyncio.to_thread(run.provider.blocking.wait, 10)
        from quant_research.__main__ import live_receipts

        observed = await research.observe(handle, client=run.client)
        receipts = live_receipts(observed.ledger, run.log.load(), settled=observed.settled)
        assert receipts.agreed
        assert [check.state for check in receipts.calls] == ["pending"]
        assert observed.report.spent["model_calls"] == "0"
        await research.control(handle, "cancel", actor="sundeep", reason="Wrong question")
        assert await handle.result() == Ending("cancelled", "researcher", 1, None)
        report = await research.record(handle)
        assert [(step.action, step.result) for step in report.steps] == [("", "cancelled")]
        assert len(run.provider.calls) == 1
        assert released(report) == ["inconclusive", "inconclusive", "inconclusive", "failed"]
        observed = await research.observe(handle, client=run.client)
        receipts = live_receipts(observed.ledger, run.log.load(), settled=observed.settled)
        assert receipts.agreed
        assert [check.state for check in receipts.calls] == ["matched"]
        assert observed.report.spent == report.spent

    serve(temporal, tmp_path, setup, [Fault.BLOCK], body)


def test_observation_requires_close_after_cancellation_before_a_claim(
    temporal, tmp_path, setup, monkeypatch
):
    from quant_research.__main__ import live_receipts
    from quant_research.providers import Closed, Unanswered

    entered = threading.Event()

    def before_claim(self, *args, cancel, **kwargs):
        entered.set()
        assert cancel.wait(10)
        return Unanswered(())

    monkeypatch.setattr(Gateway, "call", before_claim)

    async def body(run):
        handle = await research.start(run.client, pilot(setup), run.queue)
        assert await asyncio.to_thread(entered.wait, 10)
        assert run.log.load() == ()
        await handle.cancel()
        assert (await handle.result()).outcome == "cancelled"
        observed = await research.observe(handle, client=run.client)
        [attempt] = observed.settled
        assert run.log.load() == (Closed(attempt),)
        assert observed.report.spent["model_calls"] == "0"
        assert live_receipts(observed.ledger, run.log.load(), settled=observed.settled).agreed
        for entries in ((), (Closed(uuid4()),)):
            found = live_receipts(observed.ledger, entries, settled=observed.settled)
            assert found.unclosed == (str(attempt),)
            assert not found.agreed
        await Replayer(workflows=[research.ResearchLoop]).replay_workflow(
            await handle.fetch_history()
        )

    serve(temporal, tmp_path, setup, [], body)


def test_a_stop_in_flight_keeps_its_revision_after_a_steer(temporal, tmp_path, setup, monkeypatch):
    entered = threading.Event()
    original = research.ask_model

    async def waiting(revision, *args, **kwargs):
        entered.set()
        await workflow.wait_condition(lambda: workflow.instance()._head != revision.revision_id)
        return await original(revision, *args, **kwargs)

    monkeypatch.setattr(research, "ask_model", waiting)
    parent = pilot(setup)
    child = child_of(parent, question="Another question")

    async def body(run: Run) -> None:
        handle = await research.start(run.client, parent, run.queue)
        assert await asyncio.to_thread(entered.wait, 10)
        await research.steer(handle, child)
        assert (await handle.result()).outcome == "stopped"
        report = await research.record(handle)
        assert report.revision == child.revision_id
        assert report.steps[-1].revision == report.assessment.revision == parent.revision_id

    serve(temporal, tmp_path, setup, [STOP], body, unsandboxed=True)


def test_a_researcher_cancel_abandons_a_first_opening_that_hangs(temporal, tmp_path, setup):
    gate = threading.Event()

    def authorize(revision: CampaignRevision):
        gate.wait(60)
        return resolve(revision, setup)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        try:
            cancelled = await research.control(
                handle, "cancel", actor="sundeep", reason="Wrong question"
            )
            assert cancelled == "cancelled"
            assert await asyncio.wait_for(handle.result(), 10) == Ending(
                "cancelled", "researcher", 0, None
            )
        finally:
            gate.set()
        assert run.provider.calls == []

    serve(temporal, tmp_path, setup, [], body, authorize=authorize)


def test_a_steer_waiting_on_its_opening_is_refused_once_the_loop_ends(temporal, tmp_path, setup):
    revision = pilot(setup, "checkpoint before conclude")
    child = child_of(revision, question="Narrower?")
    entered, gate = threading.Event(), threading.Event()

    def authorize(opened: CampaignRevision):
        if opened.revision_id == child.revision_id:
            entered.set()
            gate.wait(60)
        return resolve(opened, setup)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        await reaches(handle, "paused")
        steered = asyncio.create_task(research.steer(handle, child))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            await research.control(handle, "resume", actor="sundeep", reason="Conclude")
            ending = await asyncio.wait_for(handle.result(), 10)
            with pytest.raises(WorkflowUpdateFailedError) as refused:
                await steered
        finally:
            gate.set()
        assert (ending.outcome, ending.steps) == ("concluded", 1)
        assert "cannot be opened" in refused.value.cause.message
        report = await research.record(handle)
        assert [action for action, _, _ in steering(report)] == ["pause", "resume"]
        assert report.revision == revision.revision_id

    serve(temporal, tmp_path, setup, [CONCLUDE, NO_RESULT, ACCEPT], body, authorize=authorize)


def test_a_request_that_cannot_fit_ends_the_loop_inconclusive(temporal, tmp_path, setup):
    labels = frozenset(licensed(f"vendor-{number:05}") for number in range(BOUND // 20))

    @dataclass(frozen=True)
    class Labelled(Toolset):
        def call(self, tool: str, arguments: str):
            used = super().call(tool, arguments)
            return replace(used, restrictions=replace(used.restrictions, labels=labels))

    def authorize(revision: CampaignRevision):
        authority = resolve(revision, setup)
        tools = authority.toolset
        return replace(
            authority, toolset=Labelled(**{name: getattr(tools, name) for name in TOOLSET})
        )

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        assert await handle.result() == Ending("inconclusive", "prompt", 1, None)
        report = await research.record(handle)
        assert [(step.action, step.result) for step in report.steps] == [("read_dataset", "ran")]
        assert len(run.provider.calls) == 1
        assert released(report) == [
            "inconclusive",
            "inconclusive",
            "inconclusive",
            "successful",
            "inconclusive",
        ]

    serve(temporal, tmp_path, setup, [READ, STOP], body, authorize=authorize)


def test_a_role_prompt_past_a_payload_ends_the_loop_at_its_opening(temporal, tmp_path, setup):
    base = pilot(setup)
    large = prompt_source(setup, "p" * 3 * 2**20, revision="large")
    sources = tuple(large if s.name == ROLE else s for s in base.specification.sources)
    revision = child_of(base, reason="Long instructions", sources=sources)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert await asyncio.wait_for(handle.result(), 30) == Ending(
            "inconclusive", "prompt", 0, None
        )
        assert run.provider.calls == []

    serve(temporal, tmp_path, setup, [], body)


def test_start_joins_its_own_loop_and_refuses_another_revision_runner_or_a_closed_campaign(
    temporal, tmp_path, setup
):
    revision = pilot(setup, "checkpoint before conclude")
    other = pilot(setup)

    async def body(run: Run) -> None:
        first = await research.start(run.client, revision, run.queue)
        await reaches(first, "paused")
        again = await research.start(run.client, revision, run.queue)
        assert again.result_run_id == first.result_run_id
        with pytest.raises(ResearchError, match="another revision"):
            await research.start(run.client, child_of(revision, question="Narrower?"), run.queue)
        jobs = await execution.start(run.client, other, (), f"{run.queue}-jobs")
        with pytest.raises(ResearchError, match="another runner"):
            await research.start(run.client, other, run.queue)
        await jobs.terminate()
        await research.control(first, "resume", actor="sundeep", reason="Done")
        await first.result()
        with pytest.raises(ResearchError, match="already run"):
            await research.start(run.client, revision, run.queue)

    serve(temporal, tmp_path, setup, [STOP], body)


def test_a_loop_started_around_start_with_a_bad_checkpoint_ends_inconclusive(
    temporal, tmp_path, setup
):
    revision = pilot(setup, "before_trial")

    async def body(run: Run) -> None:
        handle = await run.client.start_workflow(
            research.ResearchLoop.run,
            file_bytes(revision).decode(),
            id=str(revision.campaign_id),
            task_queue=run.queue,
        )
        assert await handle.result() == Ending("inconclusive", "checkpoint", 0, None)
        assert run.provider.calls == []

    serve(temporal, tmp_path, setup, [], body)


def test_start_refuses_a_revision_the_loop_cannot_run(setup):
    inactive = (ModelRole(ROLE, CODEX_MODEL, (), "inactive"),)
    refused = [
        (pilot(setup, "before_trial"), "neither"),
        (child_of(pilot(setup), model_roles=inactive), "no active researcher role"),
        (pilot(setup, limits=ResourceLimits(5, 2, 100)), "max_model_calls of at least 4"),
        (
            pilot(setup, limits=ResourceLimits(5, 2, 100, 3), fallbacks=(CODEX_MODEL,)),
            "max_model_calls of at least 6",
        ),
        (
            pilot(setup, limits=ResourceLimits(5, 2, 100, 20, max_wall_seconds=1199)),
            "max_wall_seconds of at least 2400",
        ),
    ]
    for revision, named in refused:
        with pytest.raises(ResearchError, match=named):
            asyncio.run(research.start(None, revision, "unused"))


def test_a_scripted_search_records_three_trials_rejects_the_leaky_one_and_reads_no_later_row(
    temporal, tmp_path, searching
):
    limits = ResourceLimits(5, 2, 600, max_model_calls=20)
    revision = pilot(searching, names=SEARCHED, limits=limits)
    search = resolve(revision, searching).search
    script = [propose(SAHM_TEXT), propose(LEAKY_TEXT), propose(RISE_TEXT), READ, STOP]

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert (await handle.result()).outcome == "stopped"
        report = await research.record(handle)
        assert [(step.action, step.result) for step in report.steps] == [
            ("experiment", "ran"),
            ("experiment", "rejected"),
            ("experiment", "ran"),
            ("read_dataset", "ran"),
            ("stop", "stopped"),
        ]
        trials = [step.trial for step in report.steps]
        assert len(set(trials[:3])) == 3
        assert trials[3:] == [None, None]
        head = reference.open_lineage(search, report.package).record
        assert head.trial_id is None
        assert [str(trial.trial_id) for trial in head.trials] == trials[:3]
        assert {trial.lineage_id for trial in head.trials} == {revision.campaign_id}
        for prior, later in itertools.pairwise(head.trials):
            link = LineageLink(
                prior.trial_id, prior.specification.spec_id, "follows", reference.LINK
            )
            assert later.links == (link,)
        statuses = [trial.attempts[0].outcome.status for trial in head.trials]
        assert statuses[1] == "rejected"
        assert {statuses[0], statuses[2]} <= {"successful", "null", "inconclusive"}
        assert head.trials[1].specification.input_digests == ()
        sent = run.sent()
        shown = [json.loads(texts(request)[-1].split("\nResult: ")[1]) for request in sent[1:5]]
        assert [summary["status"] for summary in shown[:3]] == statuses
        assert shown[1]["detail"].startswith("Rejected (negative_lag): ")
        assert [shown[index]["evaluation"]["result"]["trials"] for index in (0, 2)] == [1, 3]
        # The store holds UNRATE through 2023; November 2019 came out on 5 December.
        assert MONTHS[-1] == date(2023, 12, 1)
        assert max(row["key"] for row in shown[3]["rows"]) == "2019-11-01"
        roles = [event.role for event in ledger(report).events if isinstance(event, Charged)]
        assert roles.count("experiments") == 3

    serve(temporal, tmp_path, searching, script, body)


@pytest.mark.parametrize("change", ["missing", "cutoff"])
def test_a_revision_whose_rules_cannot_be_resolved_ends_inconclusive_before_any_ask(
    temporal, tmp_path, setup, change
):
    revision = pilot(setup)
    if change == "missing":
        (setup.rules / f"{RULES.digest}.json").unlink()
    else:
        setup = replace(setup, cutoff=RULES.held_back.cutoff - timedelta(seconds=1))

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert await handle.result() == Ending("inconclusive", "rules", 0, None)
        report = await research.record(handle)
        assert (report.authority, report.package, run.provider.calls) == ("", "", [])
        assert not ledger(report).events

    serve(temporal, tmp_path, setup, [], body)


def test_the_trial_limit_refuses_an_experiment_without_scheduling_it(temporal, tmp_path, setup):
    revision = pilot(setup, limits=ResourceLimits(2, 2, 600, max_model_calls=20))
    search = resolve(revision, setup).search
    script = [propose(LEAKY_TEXT), propose(PAST_TEXT), propose(SAHM_TEXT), STOP]

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        await handle.result()
        report = await research.record(handle)
        assert [(step.result, step.trial is not None) for step in report.steps] == [
            ("rejected", True),
            ("rejected", True),
            ("refused", False),
            ("stopped", False),
        ]
        first, _, _, last = run.sent()
        spent = research.TRIALS_SPENT.format(limit=2)
        assert texts(last)[-1] == f"Step 3: {script[2]}\nResult: {spent}"
        head = reference.open_lineage(search, report.package).record
        assert [str(trial.trial_id) for trial in head.trials] == [
            step.trial for step in report.steps[:2]
        ]
        events = (await handle.fetch_history()).events
        scheduled = [
            event.activity_task_scheduled_event_attributes.activity_type.name
            for event in events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
        ]
        assert scheduled.count("experiment") == 2
        # The protocol states the search window, its cutoff, the universe, and the trial limit.
        protocol = texts(first)[1]
        assert "on decisions from July 2017 to December 2019." in protocol
        assert (
            "data available by 2020-01-01T00:00:00Z. Programs may read alfred.unrate." in protocol
        )
        assert "the campaign runs at most 2.\n" in protocol
        assert SAHM_TEXT.decode().strip() in protocol
        assert "alfred.sahmrealtime" not in protocol

    serve(temporal, tmp_path, setup, script, body)


def test_a_failed_experiment_adds_no_entry_and_is_forfeited_as_a_failed_trial(
    temporal, tmp_path, setup, monkeypatch
):
    extend = reference.extend

    def failing_once(*args):
        monkeypatch.setattr(reference, "extend", extend)
        raise RuntimeError("SECRET backing text")

    monkeypatch.setattr(reference, "extend", failing_once)
    revision = pilot(setup, limits=ResourceLimits(5, 2, 600, max_model_calls=20))
    search = resolve(revision, setup).search
    experiment = says("experiment", {"program": "momentum"})

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert (await handle.result()).outcome == "stopped"
        report = await research.record(handle)
        failed, rejected, _ = report.steps
        assert [(step.result, step.digest == "") for step in report.steps] == [
            ("failed", True),
            ("rejected", False),
            ("stopped", True),
        ]
        _, second, third = run.sent()
        assert texts(second)[2:] == []
        assert texts(third)[2:] == [f"Step 2: {experiment}\nResult: {momentum(rejected.trial)}"]
        assert released(report) == [
            "inconclusive",
            "inconclusive",
            "inconclusive",
            "failed",
            "successful",
            "successful",
        ]
        roles = [event.role for event in ledger(report).events if isinstance(event, Charged)]
        assert roles.count("experiments") == 2
        forfeited, later = reference.open_lineage(search, report.package).record.trials
        assert [str(forfeited.trial_id), str(later.trial_id)] == [failed.trial, rejected.trial]
        outcome = forfeited.attempts[0].outcome
        assert (outcome.status, outcome.detail) == ("failed", reference.UNFINISHED)
        assert forfeited.specification.input_digests == ()
        assert later.links[0].trial_id == forfeited.trial_id
        assert report.unrecorded is None
        replayer = Replayer(
            workflows=[research.ResearchLoop], workflow_runner=UnsandboxedWorkflowRunner()
        )
        await replayer.replay_workflow(await handle.fetch_history())

    serve(temporal, tmp_path, setup, [experiment, experiment, STOP], body, unsandboxed=True)


def test_a_failed_forfeit_ends_the_loop_with_the_candidate_unrecorded(
    temporal, tmp_path, setup, monkeypatch
):
    def failing(*args):
        raise RuntimeError("No lineage")

    monkeypatch.setattr(reference, "extend", failing)
    monkeypatch.setattr(reference, "forfeit", failing)
    revision = pilot(setup, limits=ResourceLimits(5, 2, 600, max_model_calls=20))
    experiment = says("experiment", {"program": "momentum"})

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        assert await handle.result() == Ending("inconclusive", "lineage", 1, None)
        report = await research.record(handle)
        [step] = report.steps
        assert (step.result, report.package, len(run.provider.calls)) == ("failed", "", 1)
        assert released(report) == ["inconclusive", "inconclusive", "inconclusive", "failed"]
        unrecorded = report.unrecorded
        assert (unrecorded.revision, unrecorded.trial, unrecorded.program) == (
            revision.revision_id,
            step.trial,
            '{"program": "momentum"}',
        )
        assert unrecorded.started <= unrecorded.ended

    serve(temporal, tmp_path, setup, [experiment], body)


def test_a_cancel_during_a_forfeit_that_fails_ends_the_loop_cancelled(
    temporal, tmp_path, setup, monkeypatch
):
    entered, gate = threading.Event(), threading.Event()

    def failing(*args):
        raise RuntimeError("No result")

    def blocking(*args):
        entered.set()
        gate.wait(60)
        raise RuntimeError("No lineage")

    monkeypatch.setattr(reference, "extend", failing)
    monkeypatch.setattr(reference, "forfeit", blocking)
    revision = pilot(setup, limits=ResourceLimits(5, 2, 600, max_model_calls=20))
    experiment = says("experiment", {"program": "momentum"})

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            await research.control(handle, "cancel", actor="sundeep", reason="Wrong question")
        finally:
            gate.set()
        assert await asyncio.wait_for(handle.result(), 30) == Ending(
            "cancelled", "researcher", 1, None
        )
        report = await research.record(handle)
        [step] = report.steps
        assert (step.result, report.package, report.unrecorded.trial) == ("failed", "", step.trial)

    serve(temporal, tmp_path, setup, [experiment], body)


def test_a_cancel_during_an_experiment_ends_the_loop_and_keeps_its_candidate_unrecorded(
    temporal, tmp_path, setup, monkeypatch
):
    entered, gate = threading.Event(), threading.Event()

    def blocking(*args):
        entered.set()
        gate.wait(60)
        raise RuntimeError("Released after the loop ended")

    monkeypatch.setattr(reference, "extend", blocking)
    revision = pilot(setup, limits=ResourceLimits(5, 2, 600, max_model_calls=20))
    experiment = says("experiment", {"program": "momentum"})

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            await research.control(handle, "cancel", actor="sundeep", reason="Wrong question")
            ending = await asyncio.wait_for(handle.result(), 10)
        finally:
            gate.set()
        assert ending == Ending("cancelled", "researcher", 1, None)
        report = await research.record(handle)
        [step] = report.steps
        assert (step.action, step.result, report.package) == ("experiment", "cancelled", "")
        unrecorded = report.unrecorded
        assert (unrecorded.revision, unrecorded.trial, unrecorded.program) == (
            revision.revision_id,
            step.trial,
            '{"program": "momentum"}',
        )
        assert not (setup.packages / str(revision.campaign_id)).exists()

    serve(temporal, tmp_path, setup, [experiment], body)


@pytest.mark.parametrize("dataset_name_length", [47, 128])
def test_campaign_archive_scopes_models_to_pinned_revisions_and_preserves_supersession(
    tmp_path, setup, monkeypatch, dataset_name_length
):
    from quant_research.packages import Outcome, PackageRecord
    from quant_research.report import campaign_snapshot

    name = "synthetic." + "x" * (dataset_name_length - len("synthetic."))
    vintage_bytes = dataset_file(name)
    vintage = datasets.save(setup.datasets, name, vintage_bytes)
    setup = replace(
        setup, catalog=Catalog((*setup.catalog.events, *documented(name, vintage_bytes)))
    )
    parent = pilot(setup)
    inactive_prompt = register(
        setup.sources,
        name="critic",
        role=PROMPT,
        data=b"Critique the research when activated.",
        declaration=replace(DECLARATION, location="prompts/critic.md"),
    )
    paper = register(
        setup.sources,
        name="paper",
        role="paper",
        data=b"The model is $r_t = \\beta m_t$.\n",
        declaration=replace(
            DECLARATION,
            location="paper.md",
            source_available_at=RULES.search.cutoff,
            approved_for={CODEX, Provider("claude-subscription")},
        ),
    )
    locator = publish_locations(
        setup.sources, open_generation(setup.sources, paper.locator).occurrence.id, MARKDOWN_LOCATOR
    )
    ref = locator.ref(locator.locations[0])
    located = Located(ref, locator.occurrence.id, locator.digest, locator.locations[0])
    locator_files = {
        str(path.relative_to(setup.sources)): path.read_bytes()
        for path in (setup.sources / "locations" / locator.id).iterdir()
    }
    contract = b"Synthetic monthly observations with canonical first-of-month keys.\n"
    monthly = register(
        setup.sources,
        name="long-dataset-contract",
        role="contract",
        data=contract,
        declaration=replace(
            DECLARATION,
            location=f"{datasets.ADAPTER}/contract/{name}",
            revision=(
                "Part 0, period mapping iso-month-start/closed-label/v1, "
                f"text SHA-256 {hashlib.sha256(contract).hexdigest()}"
            ),
        ),
    )
    parent = start_campaign(
        replace(
            parent.specification,
            sources=(
                *parent.specification.sources,
                inactive_prompt,
                paper,
                monthly,
            ),
            data=(*parent.specification.data, DataPermission(name, START, END)),
            model_roles=(*parent.specification.model_roles, ModelRole("critic", "unused/model")),
        ),
        author="researcher",
        at=TIME,
        reason="Preserve inactive role instructions",
    )
    child = child_of(
        parent,
        question="A revised question?",
        model_roles=(ModelRole(ROLE, "claude-subscription/claude-opus-5-5", (), "active"),),
    )
    setup = replace(
        setup,
        policies=(
            *setup.policies,
            ProviderPolicy(
                provider=Provider("claude-subscription"), labels=frozenset(), retains=True
            ),
        ),
    )
    calls = tmp_path / "calls"
    calls.mkdir()
    archive = research.CampaignArchive(tmp_path / "archive", calls)
    authority_digests = {}
    corpora = {}
    for revision in (parent, child):
        authority = resolve(revision, setup)
        authority_digests[revision.revision_id] = hashlib.sha256(authority.record()).hexdigest()
        corpus = f"corpora/{authority.corpus.id}.json"
        corpora[corpus] = (setup.sources / corpus).read_bytes()
        assert (
            authority.toolset.call("lookup_location", json.dumps({"ref": ref})).outcome == located
        )
        archive.pin(authority)
    trial, run, first, second = (uuid4() for _ in range(4))
    decisions = (
        acceptance.Accepted(first, None, "reviewer", "Reviewed first result", TIME),
        acceptance.Accepted(second, first, "reviewer", "Reviewed replacement", TIME),
    )
    accepted = acceptance.AcceptanceLog(
        (
            acceptance.Submitted(trial, run, first, None, TIME),
            acceptance.Completed(first, Outcome("successful", "First"), TIME),
            decisions[0],
            acceptance.Submitted(trial, run, second, first, TIME),
            acceptance.Completed(second, Outcome("successful", "Replacement"), TIME),
            decisions[1],
        )
    )
    report = Report(
        workflow_id="campaign",
        run_id="run",
        state="inconclusive",
        paused_by=None,
        revision=child.revision_id,
        authority="",
        cap=STEP_CAP,
        steps=(),
        limits={},
        spent={},
        ending=Ending("inconclusive", "evaluation", 0, None),
        package="",
        unrecorded=None,
        steering='{"events":[]}',
        ledger=accounting.dump_ledger(accounting.Ledger({})).decode(),
        assessment=Assessment(child.revision_id, "", "", "", (), "", recovery="a" * 64),
    )
    recoveries = []

    def recover(search, recovery):
        recoveries.append((search.question, search.lineage, recovery))

    monkeypatch.setattr(reference, "recover_assessment", recover)
    closed = research.ClosedCampaign(
        canonical_bytes(asdict(report)),
        (file_bytes(child), file_bytes(parent)),
        b'{"events":[]}',
        acceptance=acceptance.dump_log(accepted),
    )
    package = archive.publish(closed, key="explicit-snapshot")
    assert package.record == PackageRecord(None, (), ())
    assert package.read_artifact(f"campaign-vintage-{vintage}") == vintage_bytes
    assert json.loads(package.read_artifact(f"campaign-vintage-{vintage}"))["dataset"] == name
    catalog = Catalog.from_bytes(package.read_artifact(f"campaign-{parent.revision_id}-catalog"))
    [entry] = catalog.search(name)
    assert (entry.name, entry.documentation.vintage) == (name, vintage)
    snapshot = campaign_snapshot(package)
    assert recoveries == [(child.specification.question, child.campaign_id, "a" * 64)]
    assert snapshot["question"] == child.specification.question
    checks = snapshot["cutoff_checks"]
    assert len(checks) == 4
    assert {(item["revision"], item["model"]["model"], item["period"]) for item in checks} == {
        (parent.revision_id, "gpt-6.1-sol", "search"),
        (parent.revision_id, "gpt-6.1-sol", "held_back"),
        (child.revision_id, "claude-opus-5-5", "search"),
        (child.revision_id, "claude-opus-5-5", "held_back"),
    }
    expected_models = {
        parent.revision_id: {
            "provider": "codex-subscription",
            "model": "gpt-6.1-sol",
            "version": None,
            "knowledge_cutoff": None,
        },
        child.revision_id: {
            "provider": "claude-subscription",
            "model": "claude-opus-5-5",
            "version": None,
            "knowledge_cutoff": None,
        },
    }
    assert all(item["model"] == expected_models[item["revision"]] for item in checks)
    assert package.read_artifact("campaign-acceptance") == closed.acceptance
    assert snapshot["acceptance_decisions"] == [
        {
            "attempt": str(decision.attempt_id),
            "supersedes": None if decision.supersedes is None else str(decision.supersedes),
            "actor": decision.actor,
            "reason": decision.reason,
            "at": format_time(decision.at),
        }
        for decision in decisions
    ]
    restored_sources = tmp_path / "restored-sources"
    for artifact in package.artifact_records:
        if artifact.role == "campaign-source-map":
            for relative, name in json.loads(package.read_artifact(artifact.name)).items():
                path = restored_sources / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(package.read_artifact(name))
    setup.sources.rename(tmp_path / "original-sources")
    assert {path: (restored_sources / path).read_bytes() for path in corpora} == corpora
    assert set(path.rsplit("/", 1)[1] for path in locator_files) == {
        "manifest.json",
        "manifest.sha256",
        "publication.json",
    }
    assert {path: (restored_sources / path).read_bytes() for path in locator_files} == locator_files
    for artifact in package.artifact_records:
        if artifact.role == "campaign-revision":
            restored_revision = parse_file(package.read_artifact(artifact.name))
            restored = resolve(restored_revision, replace(setup, sources=restored_sources))
            assert (
                restored.toolset.call("lookup_location", json.dumps({"ref": ref})).outcome
                == located
            )
            archived = package.read_artifact(f"campaign-{restored_revision.revision_id}-authority")
            assert restored.record() == archived
            assert (
                hashlib.sha256(restored.record()).hexdigest()
                == authority_digests[restored_revision.revision_id]
            )


def test_terminal_archive_freezes_multipart_bytes_before_completion_without_a_reader(
    temporal, tmp_path, setup
):
    from quant_research.packages import open_package
    from quant_research.report import campaign_snapshot

    async def body(run: Run) -> None:
        revision = pilot(setup, "checkpoint before conclude")
        handle = await research.start(run.client, revision, run.queue)
        await reaches(handle, "paused")
        for index in range(70):
            await research.control(
                handle, "annotate", actor="sundeep", reason=f"{index} " + "x" * 4000
            )
        await research.control(handle, "resume", actor="sundeep", reason="Finish")
        ending = await handle.result()
        [path] = (tmp_path / "archive" / "packages" / "unselected").iterdir()
        package = open_package(path)
        original = package.read_artifact("campaign-original-report")
        assert len(original) > research.PART
        parts = [
            await handle.query(research.ResearchLoop.part, offset)
            for offset in range(0, len(original), research.PART)
        ]
        assert original == b"".join(parts)
        assert json.loads(original)["ending"] == {
            "outcome": ending.outcome,
            "reason": ending.reason,
            "steps": ending.steps,
            "final": ending.final,
        }
        snapshot = campaign_snapshot(package)
        assert snapshot["report"]["assessment"]["classification"] == "no_result"
        assert len(snapshot["cutoff_checks"]) == 2
        assert all(item["status"] == "cutoff_unknown" for item in snapshot["cutoff_checks"])
        assert package.record.trials == ()
        assert {item.role for item in package.artifact_records} >= {
            "campaign-history",
            "campaign-ledger",
            "campaign-acceptance",
            "campaign-call",
            "campaign-transcript",
            "campaign-vintage",
            "campaign-source",
            "campaign-rules",
        }
        assert json.loads(package.read_artifact("campaign-acceptance"))["events"] == []
        source_map = json.loads(
            next(
                package.read_artifact(item.name)
                for item in package.artifact_records
                if item.role == "campaign-source-map"
            )
        )
        [corpus] = [name for name in source_map if name.startswith("corpora/")]
        assert package.read_artifact(source_map[corpus]) == (setup.sources / corpus).read_bytes()
        from temporalio.service import RPCError

        with pytest.raises((WorkflowUpdateFailedError, RPCError)):
            await research.control(handle, "annotate", actor="sundeep", reason="Too late")
        archived_history = json.loads(package.read_artifact("campaign-history"))
        assert all(
            event.get("activityTaskScheduledEventAttributes", {})
            .get("activityType", {})
            .get("name")
            != "archive_campaign"
            for event in archived_history["events"]
        )

    serve(temporal, tmp_path, setup, [CONCLUDE, NO_RESULT, ACCEPT], body)


def test_terminal_archive_survives_cancel_and_lost_reply_and_reuses_its_selection(
    temporal, tmp_path, setup
):
    entered, release = threading.Event(), threading.Event()
    imported = []
    registrations = []
    resolutions = []

    def authorize(revision):
        resolutions.append(revision.revision_id)
        return resolve(revision, setup)

    def importer(package):
        imported.append((package.digest, (package.path / "manifest.json").read_bytes()))
        [entry] = prospective.load_log(tmp_path / "archive" / "prospective").report()
        assert entry.registration.package == package.digest
        registrations.append(entry.registration)
        if len(imported) == 1:
            entered.set()
            assert release.wait(15)
            raise RuntimeError("configured importer reply lost")

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        assert await asyncio.to_thread(entered.wait, 15)
        resolved_before_retry = tuple(resolutions)
        await handle.cancel()
        setup.datasets.rename(tmp_path / "unavailable-datasets")
        setup.sources.rename(tmp_path / "unavailable-sources")
        (tmp_path / "calls").rename(tmp_path / "unavailable-calls")
        release.set()
        assert (await handle.result()).outcome == "stopped"
        assert len(imported) == 2 and imported[0] == imported[1]
        assert registrations == [registrations[0], registrations[0]]
        assert registrations[0].procedure.completion.selection == "unfrozen"
        assert tuple(resolutions) == resolved_before_retry
        assert len(run.provider.calls) == 1
        assert not setup.packages.exists()
        assert len(list((tmp_path / "archive" / "selections").iterdir())) == 1
        assert len(list((tmp_path / "archive" / "receipts").iterdir())) == 1
        report = await research.record(handle)
        assert report.ending.outcome == "stopped"
        assert report.state == "stopped"

    serve(temporal, tmp_path, setup, [STOP], body, authorize=authorize, archive_importer=importer)


def test_opening_failure_archives_zero_trials_without_resolving_authority_again(
    temporal, tmp_path, setup
):
    from quant_research.packages import open_package

    calls = []

    def unavailable(revision):
        calls.append(revision.revision_id)
        raise RuntimeError("Authority unavailable")

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        assert (await handle.result()).reason == "role"
        [path] = (tmp_path / "archive" / "packages" / "unselected").iterdir()
        package = open_package(path)
        assert package.record.trials == ()
        [entry] = prospective.load_log(tmp_path / "archive" / "prospective").report()
        assert entry.registration.package == package.digest
        assert entry.registration.procedure.shortlist is None
        assert entry.registration.procedure.models == ()
        assert entry.registration.procedure.completion.selection == "incomplete"
        assert entry.registration.procedure.completion.horizon == prospective.Horizon(1)
        assert entry.registration.procedure.completion.configured_models == (
            ("researcher", "codex-subscription/gpt-6.1-sol", ()),
        )
        assert len(calls) == 1
        assert not run.provider.calls
        assert any(item.role == "campaign-unavailable" for item in package.artifact_records)
        assert (
            json.loads(package.read_artifact("campaign-original-report"))["ending"]["outcome"]
            == "inconclusive"
        )

    serve(temporal, tmp_path, setup, [], body, authorize=unavailable)


def test_unexpected_implementation_failure_archives_available_state_then_fails(
    temporal, tmp_path, setup, monkeypatch
):
    from temporalio.client import WorkflowFailureError

    from quant_research.packages import open_package

    async def broken(loop, revision, role, attempt):
        raise RuntimeError("private implementation detail")

    monkeypatch.setattr(research.ResearchLoop, "_step", broken)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, pilot(setup), run.queue)
        with pytest.raises(WorkflowFailureError) as failed:
            await handle.result()
        with pytest.raises(WorkflowFailureError) as observed:
            await research.observe(handle, client=run.client)
        assert observed.value.cause.message == "Unexpected research implementation failure"
        cause = failed.value.cause
        assert cause.message == "Unexpected research implementation failure"
        assert cause.type == "RuntimeError"
        assert cause.non_retryable
        assert cause.details == ()
        assert cause.cause is None
        history = await handle.fetch_history()
        failure_event = history.events[-1].workflow_execution_failed_event_attributes
        assert b"private implementation detail" not in failure_event.failure.SerializeToString()
        [path] = (tmp_path / "archive" / "packages" / "unselected").iterdir()
        package = open_package(path)
        failure = package.read_artifact("campaign-operational-failure")
        assert b"RuntimeError" in failure
        assert b"private implementation detail" not in failure
        report = json.loads(package.read_artifact("campaign-original-report"))
        assert report["ending"] is None
        events = load_ledger(report["ledger"].encode()).events
        assert len(events) == 4
        assert all(isinstance(event, Reserved) for event in events)
        assert not run.provider.calls
        assert package.record.trials == ()

    serve(temporal, tmp_path, setup, [], body, unsandboxed=True)


def test_cancellation_during_admitted_handler_drain_keeps_the_fixed_ending(
    temporal, tmp_path, setup, monkeypatch
):
    revision = pilot(setup, "checkpoint before conclude")
    child = child_of(revision, question="Narrower?")
    entered = threading.Event()
    admitted = research.ResearchLoop._admit

    async def delayed(loop, opened):
        role = await admitted(loop, opened)
        if opened.revision_id == child.revision_id:
            entered.set()
            await workflow.sleep(0.5)
        return role

    monkeypatch.setattr(research.ResearchLoop, "_admit", delayed)

    async def body(run: Run) -> None:
        handle = await research.start(run.client, revision, run.queue)
        await reaches(handle, "paused")
        steered = asyncio.create_task(research.steer(handle, child))
        assert await asyncio.to_thread(entered.wait, 10)
        await research.control(handle, "resume", actor="sundeep", reason="Conclude")
        await reaches(handle, "concluded")
        await handle.cancel()
        assert (await handle.result()).outcome == "concluded"
        with pytest.raises(WorkflowUpdateFailedError):
            await steered
        report = await research.record(handle)
        assert report.revision == revision.revision_id
        assert [action for action, _, _ in steering(report)] == ["pause", "resume"]
        assert len(list((tmp_path / "archive" / "receipts").iterdir())) == 1

    serve(temporal, tmp_path, setup, [CONCLUDE, NO_RESULT, ACCEPT], body, unsandboxed=True)


def test_oversized_steer_is_refused_before_client_send(tmp_path, setup):
    from quant_research.steering import REVISION_BYTES, SteeringError

    child = child_of(pilot(setup), question="x" * REVISION_BYTES)
    with pytest.raises(SteeringError, match="65536 UTF-8 bytes"):
        asyncio.run(research.steer(None, child))


@pytest.mark.parametrize("character", ["x", "é", "😀", '"'])
def test_raw_steer_admission_bounds_bytes_before_acceptance_and_opening(
    temporal, tmp_path, setup, character
):
    from quant_research.steering import REVISION_BYTES

    revision = pilot(setup, "checkpoint every 1 steps")
    child = child_of(revision, question=character)
    padding = REVISION_BYTES - len(file_bytes(child))
    child = replace(
        child, specification=replace(child.specification, question=character + "x" * padding)
    )
    assert len(file_bytes(child)) == REVISION_BYTES
    oversized = replace(
        child,
        specification=replace(child.specification, question=child.specification.question + "x"),
    )

    async def body(run):
        handle = await research.start(run.client, revision, run.queue)
        paused = await reaches(handle, "paused")
        before = await handle.fetch_history()
        with pytest.raises(WorkflowUpdateFailedError) as refused:
            await handle.execute_update(
                research.ResearchLoop.steer, args=[file_bytes(oversized).decode(), None]
            )
        assert "65536 UTF-8 bytes" in refused.value.cause.message
        after = await handle.fetch_history()
        meaningful = {
            EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_ACCEPTED,
            EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED,
            EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED,
        }
        assert [(e.event_type, e.event_id) for e in after.events if e.event_type in meaningful] == [
            (e.event_type, e.event_id) for e in before.events if e.event_type in meaningful
        ]
        assert await research.status(handle) == paused
        assert await research.steer(handle, child) == "paused"
        assert (await research.status(handle)).revision == child.revision_id
        await research.control(handle, "cancel", actor="sundeep", reason="Boundary checked")
        await research.record(handle)
        await Replayer(workflows=[research.ResearchLoop]).replay_workflow(
            await handle.fetch_history()
        )

    serve(temporal, tmp_path, setup, [READ], body)


@pytest.mark.parametrize("stage", ["before_settlement", "tool_running"])
def test_observation_distinguishes_closed_receipts_from_workflow_settlement(
    temporal, tmp_path, setup, monkeypatch, stage
):
    from quant_research.__main__ import live_receipts

    original = research.ask_model if stage == "before_settlement" else research.call_tool

    async def held(*args, **kwargs):
        result = await original(*args, **kwargs)
        loop = workflow.instance()
        loop.control("pause", "researcher", "Check settlement", None)
        await workflow.wait_condition(lambda: loop._steering.status != "paused")
        return result

    monkeypatch.setattr(
        research, "ask_model" if stage == "before_settlement" else "call_tool", held
    )

    async def body(run):
        handle = await research.start(run.client, pilot(setup), run.queue)
        await reaches(handle, "paused")
        observed = await research.observe(handle, client=run.client)
        found = live_receipts(observed.ledger, run.log.load(), settled=observed.settled)
        assert found.agreed
        assert len(observed.report.steps) == 0
        assert [check.state for check in found.calls] == [
            "pending" if stage == "before_settlement" else "matched"
        ]
        assert observed.report.spent["model_calls"] == (
            "0" if stage == "before_settlement" else "1"
        )
        await research.control(handle, "cancel", actor="sundeep", reason="Finish inspection")
        report = await research.record(handle)
        observed = await research.observe(handle, client=run.client)
        assert observed.report.spent == report.spent
        assert live_receipts(observed.ledger, run.log.load(), settled=observed.settled).agreed

    serve(temporal, tmp_path, setup, [READ], body, unsandboxed=True)


def test_observation_prefix_survives_advancement_concurrent_readers_and_worker_eviction(
    temporal, tmp_path, setup, monkeypatch
):
    from temporalio.client import WorkflowQueryFailedError

    monkeypatch.setattr(research, "PART", 1024)
    calls = tmp_path / "calls"
    calls.mkdir()
    log = CallLog(calls)
    gateway = Gateway(log, [(CODEX_POLICY, ScriptedProvider([READ, READ]))])
    archive = research.CampaignArchive(tmp_path / "archive", calls)
    authorize = partial(resolve, setup=setup)
    original_query = WorkflowHandle.query
    pinned_queries = []

    async def queried(handle, query, *args, **kwargs):
        if query in (research.ResearchLoop.observation, research.ResearchLoop.ledger_part):
            pinned_queries.append(handle.run_id)
        return await original_query(handle, query, *args, **kwargs)

    monkeypatch.setattr(WorkflowHandle, "query", queried)

    async def body():
        client, queue = await Client.connect(temporal), str(uuid4())

        def worker(cached=0):
            return Worker(
                client,
                task_queue=queue,
                workflows=[research.ResearchLoop],
                activities=[
                    *Activities(gateway, authorize).definitions,
                    *research.Authorized(authorize, archive).definitions,
                ],
                workflow_runner=UnsandboxedWorkflowRunner(),
                max_cached_workflows=cached,
            )

        async with worker():
            handle = await research.start(client, pilot(setup, "checkpoint every 1 steps"), queue)
            await reaches(handle, "paused")
            description = await handle.describe()
            pinned = client.get_workflow_handle_for(
                research.ResearchLoop.run, handle.id, run_id=description.run_id
            )
            head = await pinned.query(research.ResearchLoop.observation)
            first = await pinned.query(
                research.ResearchLoop.ledger_part, args=[head.event_count, 0]
            )
            assert len(first) == research.PART
            await research.control(handle, "resume", actor="sundeep", reason="Another step")
            while len((await research.status(handle)).steps) != 2:
                await asyncio.sleep(0.05)
            await reaches(handle, "paused")
            current = await research.observe(handle, client=client)
            assert current.report.spent["model_calls"] == "2"
            assert len(current.ledger.events) > head.event_count
        async with worker(1):
            before = (await handle.fetch_history()).events
            parts = [first]
            while len(parts[-1]) == research.PART:
                part = await pinned.query(
                    research.ResearchLoop.ledger_part,
                    args=[head.event_count, len(parts) * research.PART],
                )
                assert (
                    len(DataConverter.default.payload_converter.to_payload(part).data)
                    <= research.PART
                )
                parts.append(part)
            original = load_ledger(b"".join(parts))
            assert len(original.events) == head.event_count
            assert sum(line.usage.model_calls for line in original.report()) == 1
            assert b"".join(parts) == accounting.dump_ledger(
                current.ledger, event_count=head.event_count
            )
            readers = await asyncio.gather(
                *(research.observe(handle, client=client) for _ in range(3))
            )
            assert all(reader.report == current.report for reader in readers)
            assert all(
                accounting.dump_ledger(reader.ledger) == accounting.dump_ledger(current.ledger)
                for reader in readers
            )
            assert (await handle.fetch_history()).events == before
            for count, offset in [
                (head.event_count, -1),
                (len(current.ledger.events) + 1, 0),
            ]:
                with pytest.raises(WorkflowQueryFailedError):
                    await pinned.query(research.ResearchLoop.ledger_part, args=[count, offset])
            assert all(run_id == description.run_id for run_id in pinned_queries)
            await research.control(handle, "cancel", actor="sundeep", reason="Done")
            report = await research.record(handle)
            observed = await research.observe(handle, client=client)
            assert accounting.dump_ledger(observed.ledger).decode() == report.ledger
            await Replayer(
                workflows=[research.ResearchLoop], workflow_runner=UnsandboxedWorkflowRunner()
            ).replay_workflow(await handle.fetch_history())

    asyncio.run(asyncio.wait_for(body(), 120))


@pytest.mark.parametrize("ending", [False, True])
@pytest.mark.parametrize("trial_count", [STEP_CAP, STEP_CAP + 1])
def test_observation_wire_head_and_binary_pages_bound_large_metadata_and_receipts(
    tmp_path, setup, monkeypatch, ending, trial_count
):
    from types import SimpleNamespace

    from quant_research.accounting import Budget, Usage, Work
    from quant_research.research import StepRecord, Unrecorded
    from quant_research.steering import SteeringLog

    revision = pilot(setup)
    huge = 10**4000
    revision = replace(
        revision,
        specification=replace(
            revision.specification, limits=ResourceLimits(huge, huge, huge, huge, huge, huge, huge)
        ),
    )
    loop = research.ResearchLoop(file_bytes(revision).decode())
    monkeypatch.setattr(
        research.workflow,
        "info",
        lambda: SimpleNamespace(
            workflow_id=str(revision.campaign_id), run_id=str(revision.campaign_id)
        ),
    )
    loop._steering = SteeringLog(revision.campaign_id).record(
        "pause", actor="\x01" * research.NOTE_BYTES, reason="\x02" * research.NOTE_BYTES, at=TIME
    )
    loop._steps = [
        StepRecord(
            index,
            format_time(TIME),
            revision.revision_id,
            str(uuid4()),
            "reproduce_experiment",
            "failed",
            "a" * 64,
            str(uuid4()),
        )
        for index in range(STEP_CAP + 1)
    ]
    loop._unrecorded = Unrecorded(
        revision.revision_id,
        str(uuid4()),
        "雪" * research.BOUND,
        format_time(TIME),
        format_time(TIME),
    )
    loop._assessment = Assessment(
        revision.revision_id,
        "done",
        "a" * 64,
        "b" * 64,
        tuple(
            research.HeldBack(str(uuid4()), str(uuid4()), "completed", "v" * research.BOUND)
            for _ in range(trial_count)
        ),
        "s" * research.BOUND,
        evidence="e" * research.BOUND,
        conclusion="c" * research.BOUND,
        critique="x" * research.BOUND,
        proposer=json.dumps({"provider": "codex-subscription", "model": '"\\\x01' * 512}),
        critic=json.dumps({"provider": "claude-subscription", "model": '"\\\x02' * 512}),
        independent=True,
        conclusion_calls=(str(uuid4()),),
        critique_calls=(str(uuid4()),),
        recovery="r" * 64,
    )
    if ending:
        loop._ending = Ending("stopped", "", STEP_CAP + 1, "f" * research.BOUND)
    loop._ledger = accounting.Ledger({revision.campaign_id: Budget()})
    work = Work(revision.campaign_id, uuid4(), uuid4(), uuid4())
    loop._ledger.reserve(work, Usage())
    for _ in range(1000):
        loop._ledger.charge(
            uuid4(),
            work.attempt_id,
            "researcher",
            Usage(model_calls=1, cost_usd=Decimal("0." + "1" * 1000)),
        )
    loop._settled = {uuid4() for _ in range(STEP_CAP + 4)} | {work.attempt_id}
    head = loop.observation()
    converter = DataConverter.default.payload_converter
    assert len(converter.to_payload(head).data) < research.PART
    assert head.report.spent == {}
    assert head.report.assessment == replace(
        loop._assessment,
        trials=tuple(replace(trial, evaluation=None) for trial in loop._assessment.trials),
        shortlist="",
        evidence=None,
        conclusion=None,
        critique=None,
    )
    assert head.report.unrecorded.program == ""
    assert head.report.ending is None or head.report.ending.final is None
    content = accounting.dump_ledger(loop._ledger)
    assert len(content) > research.PART
    parts = [
        loop.ledger_part(head.event_count, start) for start in range(0, len(content), research.PART)
    ]
    assert b"".join(parts) == content
    assert all(len(converter.to_payload(part).data) <= research.PART for part in parts)
    assert converter.to_payload(parts[0]).metadata == {"encoding": b"binary/plain"}
    assert accounting.dump_ledger(load_ledger(b"".join(parts))) == content


def test_observation_reconstructs_exact_spend_larger_than_a_wire_page(setup, monkeypatch):
    from types import SimpleNamespace

    from temporalio.client import WorkflowExecutionStatus

    revision = pilot(setup)
    loop = research.ResearchLoop(file_bytes(revision).decode())
    info = SimpleNamespace(workflow_id=str(revision.campaign_id), run_id=str(uuid4()))
    monkeypatch.setattr(research.workflow, "info", lambda: info)
    loop._ledger = accounting.Ledger({revision.campaign_id: accounting.Budget()})
    work = accounting.Work(revision.campaign_id, uuid4(), uuid4(), uuid4())
    loop._ledger.reserve(work, accounting.Usage())
    for amount in ("1", "1E-500000"):
        loop._ledger.charge(
            uuid4(), work.attempt_id, "researcher", accounting.Usage(cost_usd=Decimal(amount))
        )
    converter = DataConverter.default.payload_converter
    head = loop.observation()
    assert head.report.spent == {}
    assert len(converter.to_payload(head).data) < research.PART
    expected = "1." + "0" * 499999 + "1"
    assert loop.status().spent["cost_usd"] == expected
    assert len(expected.encode()) > research.PART

    class Pinned:
        id = info.workflow_id

        async def query(self, query, *, args=()):
            result = query(loop, *args)
            return converter.from_payload(
                converter.to_payload(result),
                research._Observation if query == research.ResearchLoop.observation else bytes,
            )

    class Reader:
        def get_workflow_handle_for(self, run, workflow_id, *, run_id):
            assert (workflow_id, run_id) == (info.workflow_id, info.run_id)
            return Pinned()

    observed = asyncio.run(
        research.observe(
            Pinned(),
            client=Reader(),
            description=SimpleNamespace(run_id=info.run_id, status=WorkflowExecutionStatus.RUNNING),
        )
    )
    assert observed.report.spent == loop.status().spent
    assert observed.report.spent["cost_usd"] == expected
    assert accounting.dump_ledger(observed.ledger) == accounting.dump_ledger(loop._ledger)


@pytest.mark.parametrize("character", ['"', "é", "😀", "\x01"])
def test_opening_measures_the_entire_encoded_result_at_its_exact_byte_limit(
    tmp_path, setup, monkeypatch, character
):
    from temporalio.exceptions import ApplicationError
    from temporalio.testing import ActivityEnvironment

    base = pilot(setup)
    critic = register(
        setup.sources,
        name="critic",
        role=PROMPT,
        data=b"Critique",
        declaration=replace(DECLARATION, location="critic.txt"),
    )
    revision = child_of(
        base,
        sources=(*base.specification.sources, critic),
        model_roles=(
            *base.specification.model_roles,
            ModelRole("critic", CODEX_MODEL, (), "active"),
        ),
    )
    authority = resolve(revision, setup)
    calls = tmp_path / "calls"
    calls.mkdir()
    archive = research.CampaignArchive(tmp_path / "archive", calls)
    monkeypatch.setattr(archive, "pin", lambda authority: None)
    env = ActivityEnvironment()

    async def opening(text):
        adjusted = replace(
            authority,
            prompts=authority.prompts | {"critic": replace(authority.prompts["critic"], text=text)},
        )
        return await env.run(
            research.Authorized(lambda _: adjusted, archive).opening, file_bytes(revision).decode()
        )

    async def body():
        text = character * 1000
        opened = await opening(text)
        converter = DataConverter.default.payload_converter
        size = len(converter.to_payload(opened).data)
        text += "x" * (BOUND - size)
        exact = await opening(text)
        assert len(converter.to_payload(exact).data) == BOUND
        with pytest.raises(ApplicationError) as refused:
            await opening(text + "x")
        assert refused.value.type == "PromptTooLarge"
        assert refused.value.non_retryable

    asyncio.run(body())


def test_oversized_child_opening_leaves_head_unchanged_and_records_no_large_completion(
    temporal, tmp_path, setup
):
    base = pilot(setup, "checkpoint every 1 steps")
    critic = register(
        setup.sources,
        name="critic",
        role=PROMPT,
        data=b'"' * BOUND,
        declaration=replace(DECLARATION, location="critic.txt"),
    )
    child = child_of(
        base,
        sources=(*base.specification.sources, critic),
        model_roles=(
            *base.specification.model_roles,
            ModelRole("critic", CODEX_MODEL, (), "active"),
        ),
    )

    async def body(run):
        handle = await research.start(run.client, base, run.queue)
        paused = await reaches(handle, "paused")
        with pytest.raises(WorkflowUpdateFailedError) as refused:
            await research.steer(handle, child)
        assert "cannot be opened: prompt" in refused.value.cause.message
        assert await research.status(handle) == paused
        await research.control(handle, "cancel", actor="sundeep", reason="Bounded opening checked")
        await research.record(handle)
        history = await handle.fetch_history()
        assert any(
            event.activity_task_failed_event_attributes.failure.application_failure_info.type
            == "PromptTooLarge"
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED
        )
        assert all(
            len(payload.data) <= BOUND
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED
            for payload in event.activity_task_completed_event_attributes.result.payloads
        )
        assert all(
            event.event_type != EventType.EVENT_TYPE_WORKFLOW_TASK_FAILED
            for event in history.events
        )
        await Replayer(workflows=[research.ResearchLoop]).replay_workflow(history)

    serve(temporal, tmp_path, setup, [READ], body)


def test_observation_refuses_a_terminated_workflow(temporal, tmp_path, setup):
    from temporalio.client import WorkflowFailureError

    async def body(run):
        handle = await research.start(
            run.client, pilot(setup, "checkpoint every 1 steps"), run.queue
        )
        await reaches(handle, "paused")
        await handle.terminate("Observation test")
        with pytest.raises(WorkflowFailureError):
            await research.observe(handle, client=run.client)

    serve(temporal, tmp_path, setup, [READ], body)


def test_observation_pages_the_largest_openable_fallback_chain(
    temporal, tmp_path, setup, monkeypatch
):
    from temporalio.exceptions import ApplicationError
    from temporalio.testing import ActivityEnvironment

    from quant_research.__main__ import live_receipts

    base = pilot(
        setup, "checkpoint every 1 steps", limits=ResourceLimits(5, 2, 100, max_model_calls=10000)
    )
    probe_calls = tmp_path / "probe-calls"
    probe_calls.mkdir()
    archive = research.CampaignArchive(tmp_path / "probe-archive", probe_calls)
    monkeypatch.setattr(archive, "pin", lambda authority: None)
    authorize = research.Authorized(partial(resolve, setup=setup), archive)
    env = ActivityEnvironment()

    def revision(count):
        roles = (ModelRole(ROLE, CODEX_MODEL, (CODEX_MODEL,) * (count - 1), "active"),)
        return replace(base, specification=replace(base.specification, model_roles=roles))

    async def largest():
        low, high = 1, 3000
        while low + 1 < high:
            middle = (low + high) // 2
            try:
                await env.run(authorize.opening, file_bytes(revision(middle)).decode())
            except ApplicationError as error:
                assert error.type == "PromptTooLarge"
                high = middle
            else:
                low = middle
        return low

    count = asyncio.run(largest())
    assert count > 500
    calls = tmp_path / "calls"
    calls.mkdir()
    log = CallLog(calls)
    gateway = Gateway(log, [(CODEX_POLICY, ScriptedProvider([Fault.FAIL] * (count - 1) + [READ]))])
    archive = research.CampaignArchive(tmp_path / "archive", calls)

    async def body():
        client, queue = await Client.connect(temporal), str(uuid4())
        async with research.worker(
            client, queue, gateway, partial(resolve, setup=setup), archive=archive
        ):
            handle = await research.start(client, revision(count), queue)
            await reaches(handle, "paused")
            observed = await research.observe(handle, client=client)
            content = accounting.dump_ledger(observed.ledger)
            assert len(content) > research.PART
            assert observed.report.spent["model_calls"] == str(count)
            assert len(observed.report.steps) == 1
            checks = live_receipts(observed.ledger, log.load(), settled=observed.settled)
            assert checks.agreed
            assert len(checks.calls) == count
            assert {check.state for check in checks.calls} == {"matched"}
            await research.control(
                handle, "cancel", actor="sundeep", reason="Largest chain checked"
            )
            report = await research.record(handle)
            assert report.spent == observed.report.spent
            await Replayer(workflows=[research.ResearchLoop]).replay_workflow(
                await handle.fetch_history()
            )

    asyncio.run(asyncio.wait_for(body(), 300))
