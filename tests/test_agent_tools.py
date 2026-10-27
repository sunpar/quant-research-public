from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import get_args
from uuid import uuid4

import pytest

from quant_research import datasets, worker
from quant_research.access import MemoryPin, Provider, Restrictions, derive, issue_context, licensed
from quant_research.agent_tools import (
    TOOLS,
    Call,
    Generated,
    Lessons,
    Located,
    Outcome,
    Reason,
    Recalled,
    Refusal,
    ToolError,
    Toolset,
)
from quant_research.catalog import (
    Approved,
    Catalog,
    Documented,
    Evidence,
    Revoked,
    Validated,
    Vintage,
)
from quant_research.datasets import ADAPTER, DatasetError, Description, Listing, Supplied
from quant_research.findings import (
    Declaration,
    External,
    Ledger,
    Lesson,
    Proposal,
    Review,
    ReviewError,
    Scope,
    record_id,
)
from quant_research.locators import MARKDOWN_LOCATOR, LocatorGeneration, publish_locations
from quant_research.model_provenance import UNKNOWN, ModelRecord, RoleAssignment
from quant_research.providers import (
    Answer,
    CallLog,
    Gateway,
    Lease,
    ProviderPolicy,
    Request,
    ScriptedProvider,
    Unanswered,
)
from quant_research.retrieval import (
    Corpus,
    Query,
    Retrieval,
    RetrievalError,
    publish_corpus,
    retrieve,
)
from quant_research.sources import (
    TEXT_PIPELINE,
    Generation,
    SourceDeclaration,
    SourceError,
    publish_generation,
    register_source,
)
from quant_research.worker import Limits, LoggedRun, RunLog, RunRecord

TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
EARLIER = TIME - timedelta(days=1)
LATER = TIME + timedelta(days=1)
VENDOR = licensed("synthetic-vendor")
FIRM = Provider("firm-proxy")
SUBSCRIPTION = Provider("claude-subscription")
PROVIDERS = (FIRM, SUBSCRIPTION)
SECRET = "zq-sentinel-7731"
OPEN_CLAIM = "We find momentum earns 5% per month."
OPEN_EQUATION = r"$r_t = \beta m_t$"
DECLARATION = SourceDeclaration(
    media_type="text/markdown",
    access_domain="synthetic",
    location="open.md",
    revision="1",
    acquisition="hand-authored test fixture",
    rights="synthetic, unrestricted",
    labels=frozenset(),
    approved_for=frozenset(),
    published_at=None,
    source_available_at=EARLIER,
    availability_evidence="authored for these tests",
    received_at=EARLIER,
)
SOURCES = {
    "open": (f"{OPEN_CLAIM}\n\nThe model is {OPEN_EQUATION}.\n", {}),
    "restricted": (
        f"We find momentum in {SECRET} earns 7% per month.\n",
        {"labels": frozenset({VENDOR})},
    ),
    "future": (
        f"We find momentum in {SECRET} earns 9% per month.\n",
        {"source_available_at": LATER},
    ),
    "elsewhere": (f"We find momentum in {SECRET} earns 3% per month.\n", {}),
}
VINTAGES = tuple(
    Vintage(aspect, Evidence(hashlib.sha256(aspect.encode()).hexdigest(), f"Synthetic {aspect}"))
    for aspect in ("publication", "revision", "arrival")
)
UNSTORED = "0" * 64
LIMITS = Limits(timeout=20.0, memory=128 * 2**20, pids=32, scratch=8 * 2**20, output=2**16)


def lifecycle(name, *steps, access=(), adapter=ADAPTER, vintage=UNSTORED, approved_for=()):
    events = {
        "documented": Documented(
            name,
            "researcher",
            TIME,
            "Data Engineering",
            ("close",),
            access,
            vintage,
            (),
            VINTAGES,
            approved_for,
        ),
        "validated": Validated(name, "reviewer", TIME, VINTAGES[0].evidence),
        "approved": Approved(name, "lead", TIME, adapter),
        "revoked": Revoked(name, "lead", TIME, "Adapter leaked restatements"),
    }
    return tuple(events[step] for step in steps)


def row(key, revision, at, close):
    return {
        "arrived_at": at,
        "key": key,
        "published_at": at,
        "revision": revision,
        "values": {"close": close},
    }


PRICES = [
    row("A", 1, "2026-10-20T00:00:00Z", "11"),
    row("A", 2, "2026-10-22T00:00:00Z", SECRET),
    row("B", 1, "2026-10-20T00:00:00Z", "12"),
]
RESTRICTED_ROWS = [row("A", 1, "2026-10-20T00:00:00Z", SECRET)]
QUOTES = [row("A", 1, "2026-10-20T00:00:00Z", "21")]
SCOPE = Scope(
    "US large-cap equities",
    "one month",
    datetime(2010, 1, 1, tzinfo=UTC),
    datetime(2024, 12, 31, tzinfo=UTC),
)


def remembered(statement, evidence_id):
    return Lesson(statement, "agent", SCOPE, (External(evidence_id, "supports"),))


OPEN_LESSON = remembered("Momentum earns a premium after costs", "artifact:open-backtest")
VENDOR_LESSON = remembered(f"Momentum in {SECRET} earns a premium", "artifact:synthetic-vendor")
LATER_LESSON = remembered(f"Momentum in {SECRET} persists", "artifact:later-backtest")
FRESH_LESSON = remembered(f"Momentum in {SECRET} returns", "artifact:open-backtest")


def declare(evidence_id, *labels, available_at=EARLIER):
    restrictions = Restrictions(available_at=available_at, labels=frozenset(labels))
    return Declaration(evidence_id, restrictions, "ingestion", TIME, "Synthetic restrictions")


def admit(lesson):
    return (
        Proposal(lesson, TIME),
        *(
            Review(record_id(lesson), action, "reviewer", TIME, "Checked the evidence")
            for action in ("review", "admit")
        ),
    )


REMEMBERED = (
    declare("artifact:open-backtest"),
    declare("artifact:synthetic-vendor", VENDOR),
    declare("artifact:later-backtest", available_at=LATER),
    *admit(OPEN_LESSON),
    *admit(VENDOR_LESSON),
    *admit(LATER_LESSON),
)
PIN = Ledger(REMEMBERED).head
LEDGER = Ledger((*REMEMBERED, *admit(FRESH_LESSON)))


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def fixture(name, rows) -> bytes:
    return canonical({"dataset": name, "rows": rows, "schema_version": 1})


def vintage(name, rows) -> str:
    return hashlib.sha256(fixture(name, rows)).hexdigest()


APPROVED = ("documented", "validated", "approved")
CATALOG = Catalog(
    (
        *lifecycle("vendor.prices", *APPROVED, vintage=vintage("vendor.prices", PRICES)),
        *lifecycle(
            "vendor.restricted",
            *APPROVED,
            access=(VENDOR,),
            vintage=vintage("vendor.restricted", RESTRICTED_ROWS),
            approved_for=(FIRM,),
        ),
        *lifecycle(
            "firm.prices", *APPROVED, vintage=vintage("firm.prices", QUOTES), approved_for=(FIRM,)
        ),
        *lifecycle(
            "shared.prices",
            *APPROVED,
            vintage=vintage("shared.prices", QUOTES),
            approved_for=PROVIDERS,
        ),
        *lifecycle("vendor.documented", "documented"),
        *lifecycle("vendor.validated", "documented", "validated"),
        *lifecycle("vendor.revoked", *APPROVED, "revoked"),
        *lifecycle("vendor.adapter", *APPROVED, adapter="other.adapter-v1"),
    )
)


@dataclass(frozen=True)
class World:
    sources: Path
    datasets: Path
    corpus: Corpus
    elsewhere: Corpus
    generations: dict[str, Generation]
    locators: dict[str, LocatorGeneration]

    def ref(self, source: str, index: int = 0) -> str:
        generation = self.locators[source]
        return generation.ref(generation.locations[index])


@pytest.fixture
def world(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    generations, locators = {}, {}
    for name, (text, declared) in SOURCES.items():
        declaration = replace(DECLARATION, location=f"{name}.md", **declared)
        occurrence = register_source(sources, text.encode(), declaration)
        generations[name] = publish_generation(sources, occurrence.id, TEXT_PIPELINE)
        locators[name] = publish_locations(sources, occurrence.id, MARKDOWN_LOCATOR)
    corpus = publish_corpus(
        sources, [generations[name].id for name in ("open", "restricted", "future")]
    )
    elsewhere = publish_corpus(sources, [generations["elsewhere"].id])
    store = tmp_path / "datasets"
    for name, rows in (
        ("vendor.prices", PRICES),
        ("vendor.restricted", RESTRICTED_ROWS),
        ("firm.prices", QUOTES),
        ("shared.prices", QUOTES),
    ):
        datasets.save(store, name, fixture(name, rows))
    return World(sources, store, corpus, elsewhere, generations, locators)


def tools(
    world,
    *clearances,
    cutoff=TIME,
    corpus=None,
    ledger=LEDGER,
    memory=PIN,
    generated=None,
    providers=(),
    scope=None,
    catalog=CATALOG,
) -> Toolset:
    context = issue_context(
        clearances=clearances,
        cutoff=cutoff,
        generation=(corpus or world.corpus).id,
        providers=providers,
        memory=memory,
        scope=scope,
    )
    return Toolset(context, world.sources, world.datasets, catalog, ledger, generated)


def generated(directory: Path, limits=LIMITS) -> Generated:
    """Return an empty run log and signal directory under directory."""
    for name in ("runs", "signals"):
        (directory / name).mkdir(exist_ok=True)
    return Generated(RunLog(directory / "runs"), directory / "signals", limits)


def unreachable(tmp_path) -> Toolset:
    """Return tools whose stores and review log lack what any call needs, so such a call raises."""
    context = issue_context(
        clearances=(), cutoff=TIME, generation="0" * 64, providers=(), memory=MemoryPin("f" * 64)
    )
    missing = tmp_path / "missing"
    return Toolset(context, missing, missing, CATALOG, Ledger(), generated(tmp_path))


def invoke(toolset, tool, **arguments):
    return toolset.call(tool, json.dumps(arguments))


def texts(call) -> tuple[str, ...]:
    assert isinstance(call.outcome, Retrieval), call.outcome
    return tuple(unit.text for unit in call.outcome.units)


def without_arguments(call) -> dict:
    record = json.loads(call.to_bytes())
    del record["arguments"]
    return record


def keywords(schema: dict) -> set[str]:
    nested = schema.get("properties", {}).values()
    return set(schema).union(*(keywords(item) for item in nested))


def test_every_tool_takes_only_required_string_arguments_the_model_may_set(tmp_path):
    names = [tool.name for tool in TOOLS]
    assert len(set(names)) == len(names) == 7
    properties: set[str] = set()
    for tool in TOOLS:
        schema = tool.input_schema
        assert json.loads(json.dumps(schema)) == schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"] == list(schema["properties"])
        assert all(item["type"] == "string" for item in schema["properties"].values())
        assert keywords(schema) <= {
            "type",
            "properties",
            "required",
            "additionalProperties",
            "enum",
            "description",
        }
        properties |= set(schema["properties"])
        toolset = unreachable(tmp_path)
        assert invoke(toolset, tool.name, zq="zq").outcome == Refusal("invalid_arguments")
        if schema["properties"]:
            assert invoke(toolset, tool.name).outcome == Refusal("invalid_arguments")
    assert properties == {"mode", "text", "dataset", "ref", "source"}


@pytest.mark.parametrize("name", ["search", "Search_Corpus", "search_corpus ", "", "\x00"])
def test_an_unknown_tool_name_is_refused_before_anything_runs(tmp_path, name):
    call = invoke(unreachable(tmp_path), name, mode="lexical", text="momentum")
    assert call.outcome == Refusal("unknown_tool")


HIDDEN = {
    "restricted": lambda world: tools(world, VENDOR),
    "future": lambda world: tools(world, cutoff=LATER),
    "elsewhere": lambda world: tools(world, corpus=world.elsewhere),
}


@pytest.mark.parametrize("source", HIDDEN)
@pytest.mark.parametrize("mode", ["identifier", "lexical"])
def test_search_omits_material_the_run_may_not_read_and_records_no_trace_of_it(world, source, mode):
    call = invoke(tools(world), "search_corpus", mode=mode, text="momentum")
    assert texts(call) == (OPEN_CLAIM,)
    assert SECRET.encode() not in call.to_bytes()
    assert b"synthetic-vendor" not in call.to_bytes()
    wider = invoke(HIDDEN[source](world), "search_corpus", mode=mode, text="momentum")
    assert SOURCES[source][0].strip() in texts(wider)


@pytest.mark.parametrize("source", HIDDEN)
def test_a_ref_the_run_may_not_read_is_indistinguishable_from_a_nonexistent_ref(world, source):
    hidden = world.ref(source)
    nonexistent = "f" * 64 + hidden[64:]
    toolset = tools(world)
    denied = invoke(toolset, "lookup_location", ref=hidden)
    missing = invoke(toolset, "lookup_location", ref=nonexistent)
    assert denied.outcome == missing.outcome == Refusal("unknown_ref")
    assert without_arguments(denied) == without_arguments(missing)
    assert SECRET.encode() not in denied.to_bytes()
    wider = invoke(HIDDEN[source](world), "lookup_location", ref=hidden)
    assert isinstance(wider.outcome, Located)
    assert SECRET in wider.outcome.location.text


def test_lookup_returns_the_located_element_of_a_pinned_source(world):
    generation = world.locators["open"]
    for index, text in enumerate((OPEN_CLAIM, OPEN_EQUATION)):
        ref = world.ref("open", index)
        call = invoke(tools(world), "lookup_location", ref=ref)
        assert call.outcome == Located(
            ref, generation.occurrence.id, generation.digest, generation.locations[index]
        )
        assert call.outcome.location.text == text
        assert SOURCES["open"][0][call.outcome.location.start : call.outcome.location.end] == text


def test_lookup_resolves_a_ref_of_a_pinned_generation_without_chunks(world):
    declaration = replace(DECLARATION, location="chunkless.md")
    occurrence = register_source(world.sources, SOURCES["open"][0].encode(), declaration)
    chunkless = publish_generation(
        world.sources, occurrence.id, replace(TEXT_PIPELINE, split=lambda text: [])
    )
    assert chunkless.chunks == ()
    generation = publish_locations(world.sources, occurrence.id, MARKDOWN_LOCATOR)
    corpus = publish_corpus(world.sources, [chunkless.id])
    ref = generation.ref(generation.locations[0])
    call = invoke(tools(world, corpus=corpus), "lookup_location", ref=ref)
    assert call.outcome == Located(ref, occurrence.id, generation.digest, generation.locations[0])


def test_an_uncleared_dataset_gets_the_refusal_of_a_name_that_does_not_exist(world):
    toolset = tools(world, providers=PROVIDERS)
    call = invoke(toolset, "read_dataset", dataset="vendor.restricted")
    missing = invoke(toolset, "read_dataset", dataset="zq.missing")
    assert call.outcome == missing.outcome == Refusal("unknown_dataset")
    assert without_arguments(call) == without_arguments(missing)
    assert call.material == missing.material
    for text in (call.outcome.message, call.to_bytes().decode()):
        for secret in ("synthetic-vendor", "licensed", SECRET, "2026-10"):
            assert secret not in text
    cleared = invoke(tools(world, VENDOR), "read_dataset", dataset="vendor.restricted")
    assert isinstance(cleared.outcome, Supplied)
    assert cleared.outcome.rows[0].values == {"close": SECRET}


def test_a_dataset_revision_after_the_cutoff_is_not_supplied(world):
    call = invoke(tools(world), "read_dataset", dataset="vendor.prices")
    assert isinstance(call.outcome, Supplied)
    assert [(row.key, row.revision, row.values) for row in call.outcome.rows] == [
        ("A", 1, {"close": "11"}),
        ("B", 1, {"close": "12"}),
    ]
    assert SECRET.encode() not in call.to_bytes()
    later = invoke(tools(world, cutoff=LATER), "read_dataset", dataset="vendor.prices")
    assert [(row.key, row.revision) for row in later.outcome.rows] == [("A", 2), ("B", 1)]


RECALLED = {
    "restricted": (VENDOR_LESSON, lambda world: tools(world, VENDOR)),
    "future": (LATER_LESSON, lambda world: tools(world, cutoff=LATER)),
    "after_pin": (FRESH_LESSON, lambda world: tools(world, memory=LEDGER.head)),
}


@pytest.mark.parametrize("case", RECALLED)
def test_read_lessons_omits_lessons_the_run_may_not_read_and_records_no_trace_of_them(world, case):
    hidden, wider = RECALLED[case]
    call = invoke(tools(world), "read_lessons")
    assert call.outcome == Lessons(PIN, (Recalled(record_id(OPEN_LESSON), OPEN_LESSON),))
    for secret in (SECRET, "synthetic-vendor", record_id(hidden)):
        assert secret.encode() not in call.to_bytes()
    recalled = invoke(wider(world), "read_lessons")
    assert Recalled(record_id(hidden), hidden) in recalled.outcome.lessons
    assert SECRET.encode() in recalled.to_bytes()


def test_a_lessons_record_replays_while_the_bound_log_keeps_the_pinned_prefix(world):
    toolset = tools(world)
    call = invoke(toolset, "read_lessons")
    record = call.to_bytes()
    suspension = Review(record_id(OPEN_LESSON), "suspend", "reviewer", LATER, "Source corrected")
    grown = LEDGER.apply(suspension)
    current = invoke(tools(world, ledger=grown, memory=grown.head), "read_lessons")
    assert current.outcome == Lessons(
        grown.head, (Recalled(record_id(FRESH_LESSON), FRESH_LESSON),)
    )
    replayed = replace(toolset, ledger=grown).replay(record)
    assert replayed == call
    assert replayed.material == call.material
    assert replayed.to_bytes() == record
    reordered = Ledger((REMEMBERED[1], REMEMBERED[0], *REMEMBERED[2:]))
    for ledger in (reordered, Ledger(REMEMBERED[:-1])):
        other = replace(toolset, ledger=ledger)
        with pytest.raises(ReviewError):
            other.replay(record)
        with pytest.raises(ReviewError):
            other.call("read_lessons", "{}")


WIDENING = {
    "cutoff": "2099-01-01T00:00:00Z",
    "generation": "f" * 64,
    "clearances": str(VENDOR),
    "context": "f" * 64,
    "providers": "any",
    "memory": "f" * 64,
    "pin": "f" * 64,
}
ALLOWED = [
    ("search_corpus", {"mode": "lexical", "text": "momentum"}),
    ("read_dataset", {"dataset": "vendor.prices"}),
    ("describe_dataset", {"dataset": "vendor.prices"}),
    ("lookup_location", {"ref": "f" * 64 + ":claim:0-36"}),
    ("read_lessons", {}),
    ("run_signal", {"source": "pass", "dataset": "vendor.prices"}),
]


@pytest.mark.parametrize("key", WIDENING)
@pytest.mark.parametrize(("tool", "arguments"), ALLOWED, ids=[tool for tool, _ in ALLOWED])
def test_an_argument_naming_run_authority_is_refused_before_anything_runs(
    tmp_path, key, tool, arguments
):
    toolset = unreachable(tmp_path)
    with pytest.raises((RetrievalError, SourceError, DatasetError, ReviewError)):
        toolset.call(tool, json.dumps(arguments))
    call = toolset.call(tool, json.dumps({**arguments, key: WIDENING[key]}))
    assert call.outcome == Refusal("invalid_arguments")


@pytest.mark.parametrize(
    "name", ["../outside", "vendor/prices", "Vendor.Prices", "..", "vendor.prices/../../outside"]
)
def test_a_dataset_name_that_is_not_a_logical_name_opens_no_file(tmp_path, name):
    store = tmp_path / "datasets"
    target = store / name / f"{UNSTORED}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Opening the target would raise, because garbage is not a fixture.
    target.write_bytes(b"garbage")
    context = issue_context(clearances=(), cutoff=TIME, generation="0" * 64, providers=())
    catalog = Catalog(lifecycle(name, *APPROVED))
    toolset = Toolset(context, tmp_path / "missing", store, catalog, Ledger())
    assert invoke(toolset, "read_dataset", dataset=name).outcome == Refusal("unknown_dataset")


REF = "a" * 64 + ":claim:0-5"


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("search_corpus", "not json"),
        ("search_corpus", ""),
        ("search_corpus", '["lexical", "momentum"]'),
        ("search_corpus", '"momentum"'),
        ("search_corpus", '{"mode":"lexical","mode":"identifier","text":"momentum"}'),
        ("search_corpus", '{"mode":"lexical","text":NaN}'),
        ("search_corpus", '{"mode":"lexical","text":1}'),
        ("search_corpus", '{"mode":"lexical","text":true}'),
        ("search_corpus", '{"mode":"lexical","text":null}'),
        ("search_corpus", '{"mode":"lexical","text":{"text":"momentum"}}'),
        ("search_corpus", '{"mode":"lexical","text":["momentum"]}'),
        ("search_corpus", '{"mode":"lexical"}'),
        ("search_corpus", "{}"),
        ("search_corpus", '{"mode":"lexical","text":" \\t"}'),
        ("search_corpus", '{"mode":"lexical","text":"a\\u0000b"}'),
        ("search_corpus", '{"mode":"lexical","text":"\\ud800"}'),
        ("search_corpus", '{"mode":"fuzzy","text":"momentum"}'),
        ("search_corpus", '{"mode":"Lexical","text":"momentum"}'),
        ("search_corpus", "\ufeff" + '{"mode":"lexical","text":"momentum"}'),
        ("read_dataset", '{"dataset":7}'),
        ("read_dataset", '{"dataset":"vendor.prices","dataset":"vendor.restricted"}'),
        *(
            ("lookup_location", json.dumps({"ref": ref}))
            for ref in (
                REF.upper(),
                REF.replace(":0-", ":00-"),
                REF.replace("-5", "-05"),
                REF.replace("claim", "theorem"),
                REF.replace("-5", "-0"),
                REF.replace("0-5", "5-5"),
                REF.replace("0-5", "9-5"),
                REF + ":claim",
                REF[1:],
                REF + "\n",
                " " + REF,
                REF.replace("0-5", "-1-5"),
                REF.replace("-5", "-" + "9" * 20),
                REF.replace("-5", "-" + "1" * 5000),
                REF.replace("0-5", "\u0660-\u0665"),
                "",
            )
        ),
        ("lookup_location", json.dumps({"ref": REF, "kind": "claim"})),
    ],
)
def test_arguments_outside_the_schema_are_refused_before_anything_runs(tmp_path, tool, arguments):
    call = unreachable(tmp_path).call(tool, arguments)
    assert call.outcome == Refusal("invalid_arguments")
    assert call.arguments == arguments


CASES = {
    "retrieval": (
        "search_corpus",
        lambda world: {"mode": "lexical", "text": "momentum"},
        Retrieval,
    ),
    "read": ("read_dataset", lambda world: {"dataset": "vendor.prices"}, Supplied),
    "location": ("lookup_location", lambda world: {"ref": world.ref("open")}, Located),
    "lessons": ("read_lessons", lambda world: {}, Lessons),
    "listing": ("list_datasets", lambda world: {}, Listing),
    "description": ("describe_dataset", lambda world: {"dataset": "vendor.prices"}, Description),
    "unknown_tool": ("zq_search", lambda world: {"text": "zq"}, Refusal),
    "invalid_arguments": ("search_corpus", lambda world: {"mode": "zq", "text": "zq"}, Refusal),
    "unknown_ref": ("lookup_location", lambda world: {"ref": world.ref("restricted")}, Refusal),
    "unknown_dataset": ("read_dataset", lambda world: {"dataset": "zq.missing"}, Refusal),
    "documented": ("read_dataset", lambda world: {"dataset": "vendor.documented"}, Refusal),
    "validated": ("read_dataset", lambda world: {"dataset": "vendor.validated"}, Refusal),
    "revoked": ("read_dataset", lambda world: {"dataset": "vendor.revoked"}, Refusal),
    "other_adapter": ("read_dataset", lambda world: {"dataset": "vendor.adapter"}, Refusal),
    "uncleared_dataset": ("read_dataset", lambda world: {"dataset": "vendor.restricted"}, Refusal),
    "generated_code_off": (
        "run_signal",
        lambda world: {"source": "zq", "dataset": "vendor.prices"},
        Refusal,
    ),
}
REASONS = [
    reason
    for part in get_args(Reason.__value__)
    for reason in get_args(getattr(part, "__value__", part))
]


@pytest.mark.smoke
@pytest.mark.parametrize("case", CASES)
def test_every_outcome_records_its_context_and_replays_byte_identically(world, case):
    tool, arguments, kind = CASES[case]
    toolset = tools(world)
    call = toolset.call(tool, json.dumps(arguments(world)))
    assert type(call.outcome) is kind
    record = call.to_bytes()
    assert json.loads(record)["context"] == toolset.context.identity
    assert toolset.replay(record) == call
    assert tools(world).replay(record).to_bytes() == record


@pytest.mark.parametrize("reason", REASONS)
def test_every_refusal_has_a_fixed_message_that_quotes_no_argument(world, reason):
    tool, arguments, _ = CASES[reason]
    values = arguments(world)
    call = tools(world).call(tool, json.dumps(values))
    assert call.outcome == Refusal(reason)
    assert call.outcome.message == Refusal(reason).message
    assert all(text not in call.outcome.message for text in (tool, *values.values()))


def test_every_reason_has_its_own_message_and_dataset_reasons_keep_their_wording():
    messages = [Refusal(reason).message for reason in REASONS]
    assert all(messages) and len(set(messages)) == len(messages)
    for reason in get_args(datasets.Reason.__value__):
        assert Refusal(reason).message == datasets.Refused(reason).message


def test_replay_refuses_a_record_from_another_context_before_anything_runs(world, tmp_path):
    toolset = tools(world, cutoff=LATER)
    narrowed = replace(toolset, context=toolset.context.narrow(cutoff=TIME))
    for tool, arguments in (
        ("search_corpus", {"mode": "lexical", "text": "momentum"}),
        ("zq_search", {"text": "zq"}),
    ):
        record = invoke(toolset, tool, **arguments).to_bytes()
        for other in (narrowed, unreachable(tmp_path)):
            with pytest.raises(ToolError):
                other.replay(record)


def test_replay_refuses_a_record_with_any_byte_changed(world):
    toolset = tools(world)
    record = invoke(toolset, "read_dataset", dataset="vendor.prices").to_bytes()
    for index in range(len(record)):
        changed = record[:index] + bytes([record[index] ^ 1]) + record[index + 1 :]
        with pytest.raises(ToolError):
            toolset.replay(changed)


def revintaged(world) -> Catalog:
    """Store a changed vendor.prices fixture and return a catalog that approves it."""
    changed = [row("A", 1, "2026-10-20T00:00:00Z", "13"), *PRICES[1:]]
    stored = datasets.save(world.datasets, "vendor.prices", fixture("vendor.prices", changed))
    return Catalog(
        (
            *CATALOG.events,
            Revoked("vendor.prices", "lead", TIME, "New vintage"),
            *lifecycle("vendor.prices", *APPROVED, vintage=stored),
        )
    )


def test_replay_refuses_a_read_after_a_new_vintage_is_approved(world):
    record = invoke(tools(world), "read_dataset", dataset="vendor.prices").to_bytes()
    with pytest.raises(ToolError):
        replace(tools(world), catalog=revintaged(world)).replay(record)


@pytest.mark.parametrize(
    "record",
    [
        b"",
        b"garbage",
        b"\xff",
        b"[]\n",
        canonical({"schema_version": 1, "tool": "zq_search", "arguments": "{}"}),
        canonical(
            {
                "schema_version": 1,
                "tool": ["zq_search"],
                "arguments": "{}",
                "context": "0" * 64,
                "outcome": {"refusal": "unknown_tool"},
            }
        ),
    ],
)
def test_replay_refuses_a_malformed_record(world, record):
    with pytest.raises(ToolError):
        tools(world).replay(record)


def test_a_run_without_generated_code_is_refused_before_any_read(tmp_path):
    toolset = replace(unreachable(tmp_path), generated=None)
    call = invoke(toolset, "run_signal", source="pass", dataset="vendor.prices")
    assert call.outcome == Refusal("generated_code_off")


def test_identical_logged_runs_each_replay_without_a_container(world, tmp_path):
    bound = generated(tmp_path)
    toolset = tools(world, generated=bound)
    arguments = json.dumps({"source": "pass", "dataset": "vendor.prices"})
    read = invoke(toolset, "read_dataset", dataset="vendor.prices").outcome
    source = hashlib.sha256(b"pass").hexdigest()
    record = RunRecord("sha256:" + "a" * 64, source, read.record, LIMITS)
    runs = [bound.log.append(worker.Refused("failed", record)) for _ in range(2)]
    context = toolset.context.identity
    records = [
        Call("run_signal", arguments, context, run, read.restrictions).to_bytes() for run in runs
    ]
    assert [toolset.replay(item).outcome for item in records] == runs
    assert bound.log.load() == tuple(runs)


@pytest.mark.parametrize(
    "dataset", ["vendor.restricted", "zq.missing"], ids=["uncleared", "unknown"]
)
def test_a_refused_read_starts_no_container_and_logs_no_run(world, tmp_path, dataset):
    bound = generated(tmp_path)
    toolset = tools(world, generated=bound)
    call = invoke(toolset, "run_signal", source="pass", dataset=dataset)
    assert call.outcome == Refusal("unknown_dataset")
    assert toolset.replay(call.to_bytes()) == call
    assert bound.log.load() == ()
    assert list(bound.signals.iterdir()) == []


SIGNAL = """\
import json

rows = json.load(open("/job/slice.json"))["rows"]
signal = {row["key"]: float(row["values"]["close"]) for row in rows}
open("/output/signal.json", "w").write(json.dumps(signal))
"""
NOISY = """\
import sys

marker = "zq-" + "printed"
print(marker)
print(marker, file=sys.stderr)
open("/output/signal.json", "w").write(marker)
"""


def test_run_signal_logs_each_container_run_and_replays_from_the_log(world, tmp_path, image):
    bound = generated(tmp_path)
    toolset = tools(world, generated=bound)
    read = invoke(toolset, "read_dataset", dataset="vendor.prices").outcome
    signal = invoke(toolset, "run_signal", source=SIGNAL, dataset="vendor.prices")
    noisy = invoke(toolset, "run_signal", source=NOISY, dataset="vendor.prices")
    assert bound.log.load() == (signal.outcome, noisy.outcome)
    for call, source in ((signal, SIGNAL), (noisy, NOISY)):
        assert isinstance(call.outcome, LoggedRun)
        assert call.outcome.record.image == image
        assert call.outcome.record.source == hashlib.sha256(source.encode()).hexdigest()
        assert call.outcome.record.read == read.record
        assert call.outcome.record.limits == LIMITS
        assert str(call.outcome.run_id).encode() in call.to_bytes()
    values = {"A": 11.0, "B": 12.0}
    assert signal.outcome.outcome == "signal"
    assert signal.outcome.output == hashlib.sha256(canonical(values)).hexdigest()
    [path] = bound.signals.iterdir()
    assert path.name == f"{signal.outcome.output}.json"
    assert json.loads(path.read_bytes()) == values
    assert (noisy.outcome.outcome, noisy.outcome.output) == ("malformed", None)
    assert b"zq-printed" not in noisy.to_bytes()
    assert SECRET.encode() not in signal.to_bytes() + noisy.to_bytes()

    records = [call.to_bytes() for call in (signal, noisy)]
    assert [toolset.replay(record) for record in records] == [signal, noisy]
    assert len(bound.log.load()) == 2
    assert list(bound.signals.iterdir()) == [path]
    swapped = json.loads(records[0])
    swapped["arguments"] = noisy.arguments
    with pytest.raises(ToolError):
        toolset.replay(canonical(swapped))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for other in (
        replace(toolset, generated=generated(elsewhere)),
        replace(toolset, generated=replace(bound, limits=replace(LIMITS, timeout=30.0))),
        replace(toolset, generated=None),
        replace(toolset, catalog=revintaged(world)),
    ):
        for record in records:
            with pytest.raises(ToolError):
                other.replay(record)


FIRM_POLICY = ProviderPolicy(provider=FIRM, labels=frozenset(), retains=False)
SUBSCRIPTION_POLICY = ProviderPolicy(provider=SUBSCRIPTION, labels=frozenset(), retains=True)
FIRM_MODEL = ModelRecord("firm-proxy", "firm-model", UNKNOWN, UNKNOWN)
SUBSCRIPTION_MODEL = ModelRecord("claude-subscription", "subscription-model", UNKNOWN, UNKNOWN)
WITHHELD = {
    "refusal": "result_too_large",
    "message": "The result is larger than 262144 bytes, so it is withheld",
}


def floor(context) -> Restrictions:
    return Restrictions(
        available_at=context.cutoff, labels=frozenset(), approved_for=context.providers
    )


def recalled(lesson) -> dict:
    return {
        "record_id": record_id(lesson),
        "statement": lesson.statement,
        "author": "agent",
        "scope": {
            "universe": "US large-cap equities",
            "horizon": "one month",
            "start": "2010-01-01T00:00:00Z",
            "end": "2024-12-31T00:00:00Z",
            "assumptions": [],
        },
        "evidence": [{"evidence_id": lesson.evidence[0].evidence_id, "relation": "supports"}],
    }


UNRATE = "alfred.unrate"
CONTRACTED = Catalog((*CATALOG.events, *lifecycle(UNRATE, *APPROVED, approved_for=(FIRM,))))


def contracted(world) -> tuple[tuple[Generation, ...], Corpus]:
    """Publish alfred.unrate's shipped contract and a corpus of it and the open source."""
    parts = datasets.publish_contract(world.sources, CONTRACTED, UNRATE)
    corpus = publish_corpus(
        world.sources, [world.generations["open"].id, *(part.id for part in parts)]
    )
    return parts, corpus


def units(generations) -> list[dict]:
    return [
        {
            "occurrence_id": generation.occurrence.id,
            "generation_id": generation.id,
            "start": chunk.start,
            "end": chunk.end,
            "text": chunk.text,
        }
        for generation in generations
        for chunk in generation.chunks
    ]


def member(world, name, text, **declared) -> Generation:
    declaration = replace(DECLARATION, location=f"{name}.md", **declared)
    occurrence = register_source(world.sources, text.encode(), declaration)
    return publish_generation(world.sources, occurrence.id, TEXT_PIPELINE)


def example_search(world, tmp_path):
    toolset = tools(world, VENDOR, providers=PROVIDERS)
    call = invoke(toolset, "search_corpus", mode="identifier", text="momentum earns")
    generation = world.generations["open"]
    unit = {
        "occurrence_id": generation.occurrence.id,
        "generation_id": generation.id,
        "start": 0,
        "end": len(OPEN_CLAIM),
        "text": OPEN_CLAIM,
    }
    return toolset, call, {"units": [unit]}, Restrictions(available_at=TIME, labels={VENDOR})


def example_read(world, tmp_path):
    toolset = tools(world, providers=PROVIDERS)
    call = invoke(toolset, "read_dataset", dataset="firm.prices")
    restrictions = Restrictions(available_at=TIME, labels=frozenset(), approved_for={FIRM})
    return toolset, call, {"dataset": "firm.prices", "rows": QUOTES}, restrictions


def example_location(world, tmp_path):
    toolset = tools(world, VENDOR, providers=PROVIDERS)
    ref = world.ref("restricted")
    call = invoke(toolset, "lookup_location", ref=ref)
    location = {
        "ref": ref,
        "occurrence_id": world.locators["restricted"].occurrence.id,
        "kind": "claim",
        "start": 0,
        "end": 56,
        "text": SOURCES["restricted"][0].strip(),
    }
    return toolset, call, location, Restrictions(available_at=TIME, labels={VENDOR})


def example_lessons(world, tmp_path):
    toolset = tools(world, VENDOR, providers=PROVIDERS)
    call = invoke(toolset, "read_lessons")
    lessons = {"lessons": [recalled(OPEN_LESSON), recalled(VENDOR_LESSON)]}
    return toolset, call, lessons, Restrictions(available_at=TIME, labels={VENDOR})


def example_run(world, tmp_path):
    bound = generated(tmp_path)
    toolset = tools(world, generated=bound, providers=PROVIDERS)
    read = invoke(toolset, "read_dataset", dataset="firm.prices")
    source = hashlib.sha256(b"pass").hexdigest()
    run = bound.log.append(
        worker.Refused(
            "failed", RunRecord("sha256:" + "a" * 64, source, read.outcome.record, LIMITS)
        )
    )
    arguments = json.dumps({"source": "pass", "dataset": "firm.prices"})
    record = Call("run_signal", arguments, toolset.context.identity, run, read.restrictions)
    call = toolset.replay(record.to_bytes())
    text = {
        "run_id": str(run.run_id),
        "reason": "failed",
        "message": "The candidate exited with an error",
    }
    restrictions = Restrictions(available_at=TIME, labels=frozenset(), approved_for={FIRM})
    return toolset, call, text, restrictions


def example_refusal(world, tmp_path):
    toolset = tools(world, providers=PROVIDERS)
    call = invoke(toolset, "read_dataset", dataset="vendor.restricted")
    text = {"refusal": "unknown_dataset", "message": "No dataset this run may read has this name"}
    restrictions = Restrictions(available_at=TIME, labels=frozenset(), approved_for=PROVIDERS)
    return toolset, call, text, restrictions


def example_listing(world, tmp_path):
    toolset = tools(world, VENDOR, providers=PROVIDERS)
    call = invoke(toolset, "list_datasets")
    listed = ("firm.prices", "shared.prices", "vendor.prices", "vendor.restricted")
    text = {
        "datasets": [
            {"dataset": name, "owner": "Data Engineering", "fields": ["close"]} for name in listed
        ]
    }
    return toolset, call, text, Restrictions(available_at=TIME, labels={VENDOR})


def example_description(world, tmp_path):
    parts, corpus = contracted(world)
    toolset = tools(world, providers=PROVIDERS, corpus=corpus, catalog=CONTRACTED)
    call = invoke(toolset, "describe_dataset", dataset=UNRATE)
    text = {"dataset": UNRATE, "units": units(parts)}
    restrictions = Restrictions(available_at=TIME, labels=frozenset(), approved_for={FIRM})
    return toolset, call, text, restrictions


EXAMPLES = {
    Retrieval: example_search,
    Supplied: example_read,
    Listing: example_listing,
    Description: example_description,
    Located: example_location,
    Lessons: example_lessons,
    LoggedRun: example_run,
    Refusal: example_refusal,
}


@pytest.mark.parametrize("kind", get_args(Outcome.__value__), ids=lambda kind: kind.__name__)
def test_every_outcome_kind_renders_as_material_under_what_it_read(world, tmp_path, kind):
    toolset, call, text, restrictions = EXAMPLES[kind](world, tmp_path)
    assert type(call.outcome) is kind
    assert json.loads(call.material.text) == text
    assert call.material.restrictions == restrictions
    assert (restrictions == floor(toolset.context)) is (kind is Refusal)
    record = call.to_bytes()
    replayed = toolset.replay(record)
    assert replayed.to_bytes() == record
    assert replayed.material == call.material


def test_a_search_goes_only_where_every_member_it_read_may_go(world):
    shared = member(world, "shared", "We find carry earns 2% per month.\n", approved_for=PROVIDERS)
    firm = member(world, "firm", "Value spreads widen in recessions.\n", approved_for={FIRM})
    corpus = publish_corpus(world.sources, [shared.id, firm.id])
    toolset = tools(world, corpus=corpus, providers=PROVIDERS)
    matching = invoke(toolset, "search_corpus", mode="identifier", text="carry")
    nonmatching = invoke(toolset, "search_corpus", mode="lexical", text="momentum")
    assert texts(matching) == ("We find carry earns 2% per month.",)
    assert texts(nonmatching) == ()
    for call in (matching, nonmatching):
        assert call.material.restrictions.approved_for == {FIRM}
        assert FIRM_POLICY.denials(toolset.context, call.material.restrictions) == ()
        assert SUBSCRIPTION_POLICY.denials(toolset.context, call.material.restrictions) == (
            "provider_not_approved",
        )


def test_each_dataset_read_renders_under_its_own_approvals(world):
    toolset = tools(world, providers=PROVIDERS)
    approvals = {
        name: invoke(toolset, "read_dataset", dataset=name).material.restrictions.approved_for
        for name in ("vendor.prices", "firm.prices", "shared.prices")
    }
    assert approvals == {
        "vendor.prices": set(),
        "firm.prices": {FIRM},
        "shared.prices": {FIRM, SUBSCRIPTION},
    }


def test_an_empty_result_keeps_what_it_read_and_a_refusal_gets_the_floor(world, tmp_path):
    cleared = tools(world, VENDOR, providers=PROVIDERS)
    search = invoke(cleared, "search_corpus", mode="identifier", text="carry")
    assert texts(search) == ()
    assert search.material.restrictions == Restrictions(available_at=TIME, labels={VENDOR})
    early = tools(world, VENDOR, cutoff=TIME - timedelta(days=2), providers=PROVIDERS)
    lessons = invoke(early, "read_lessons")
    assert lessons.outcome.lessons == ()
    assert lessons.material.restrictions == floor(early.context)
    read = invoke(early, "read_dataset", dataset="vendor.restricted")
    assert read.outcome.rows == ()
    assert read.material.restrictions == Restrictions(
        available_at=early.context.cutoff, labels={VENDOR}, approved_for={FIRM}
    )
    uncleared = tools(world, generated=generated(tmp_path), providers=PROVIDERS)
    refused = invoke(uncleared, "run_signal", source="pass", dataset="vendor.restricted")
    assert refused.outcome == Refusal("unknown_dataset")
    assert refused.material.restrictions == floor(uncleared.context)
    toolset = tools(world, providers=PROVIDERS)
    for reason in REASONS:
        tool, arguments, _ = CASES[reason]
        call = toolset.call(tool, json.dumps(arguments(world)))
        assert call.outcome == Refusal(reason)
        assert call.material.restrictions == floor(toolset.context)


def test_the_floor_keeps_every_send_decision_for_a_permitted_provider(world):
    toolset = tools(world, providers=(FIRM,))
    refused = invoke(toolset, "zq_search").material.restrictions
    for labels, available_at, approved_for in product(
        ((), (VENDOR,)), (EARLIER, TIME, LATER), ((), (FIRM,), (SUBSCRIPTION,), PROVIDERS)
    ):
        item = Restrictions(available_at=available_at, labels=labels, approved_for=approved_for)
        derived = derive(refused, item)
        assert FIRM_POLICY.denials(toolset.context, derived) == FIRM_POLICY.denials(
            toolset.context, item
        )
        denials = SUBSCRIPTION_POLICY.denials(toolset.context, derived)
        assert "provider_not_permitted" in denials
        assert set(SUBSCRIPTION_POLICY.denials(toolset.context, item)) <= set(denials)


@pytest.mark.parametrize("mode", ["identifier", "lexical"])
def test_a_search_returns_at_most_20_units_and_retrieve_still_returns_every_match(world, mode):
    notes = "\n\n".join(f"Momentum note {index}." for index in range(25))
    corpus = publish_corpus(world.sources, [member(world, "notes", f"{notes}\n").id])
    toolset = tools(world, corpus=corpus)
    call = invoke(toolset, "search_corpus", mode=mode, text="Momentum")
    every = retrieve(world.sources, toolset.context, Query(mode, "Momentum")).units
    assert len(every) == 25
    assert call.outcome.units == every[:20]
    assert len(json.loads(call.material.text)["units"]) == 20
    assert toolset.replay(call.to_bytes()) == call


def test_a_result_over_the_byte_cap_is_withheld_under_the_reads_full_restrictions(world):
    def approve(name, value):
        rows = [row("A", 1, "2026-10-20T00:00:00Z", value)]
        datasets.save(world.datasets, name, fixture(name, rows))
        events = lifecycle(
            name, *APPROVED, access=(VENDOR,), vintage=vintage(name, rows), approved_for=(FIRM,)
        )
        return rows, events

    empty = canonical(
        {"dataset": "vendor.limit", "rows": [row("A", 1, "2026-10-20T00:00:00Z", "")]}
    )
    room = 2**18 - len(empty)
    fill = "\u00e9" * (room // 2) + "x" * (room % 2)
    limit_rows, limit_events = approve("vendor.limit", fill)
    _, large_events = approve("vendor.large", fill + "x")
    catalog = Catalog((*CATALOG.events, *limit_events, *large_events))
    toolset = replace(tools(world, VENDOR, providers=PROVIDERS), catalog=catalog)
    shown = invoke(toolset, "read_dataset", dataset="vendor.limit")
    assert len(shown.material.text.encode()) == 2**18
    assert json.loads(shown.material.text) == {"dataset": "vendor.limit", "rows": limit_rows}
    withheld = invoke(toolset, "read_dataset", dataset="vendor.large")
    assert json.loads(withheld.material.text) == WITHHELD
    assert withheld.material.restrictions == Restrictions(
        available_at=TIME, labels={VENDOR}, approved_for={FIRM}
    )
    assert withheld.outcome.rows[0].values == {"close": fill + "x"}
    assert toolset.replay(withheld.to_bytes()) == withheld


def gateway(directory, routes):
    (directory / "calls").mkdir()
    router = Gateway(CallLog(directory / "calls"), routes)

    def send(context, model, material):
        return router.call(
            context,
            RoleAssignment("researcher", model),
            Request((material,)),
            lease=Lease(uuid4(), (uuid4(),), 1),
            cancel=threading.Event(),
        )

    return send


def test_a_firm_only_read_is_refused_on_a_subscription_route_and_sent_on_the_firm_route(
    world, tmp_path
):
    toolset = tools(world, providers=PROVIDERS)
    material = invoke(toolset, "read_dataset", dataset="firm.prices").material
    assert json.loads(material.text) == {"dataset": "firm.prices", "rows": QUOTES}
    subscription, firm = ScriptedProvider([]), ScriptedProvider(["Noted"])
    send = gateway(tmp_path, [(SUBSCRIPTION_POLICY, subscription), (FIRM_POLICY, firm)])
    refused = send(toolset.context, SUBSCRIPTION_MODEL, material)
    assert isinstance(refused, Unanswered)
    assert [(record.outcome, record.reasons) for record in refused.records] == [
        ("refused", ("provider_not_approved",))
    ]
    assert subscription.calls == []
    answered = send(toolset.context, FIRM_MODEL, material)
    assert isinstance(answered, Answer) and answered.material.text == "Noted"
    assert [call.texts for call in firm.calls] == [(material.text,)]


def test_a_declaration_after_the_pin_only_tightens_lessons_at_the_new_head(world):
    toolset = tools(world, providers=PROVIDERS)
    call = invoke(toolset, "read_lessons")
    assert call.outcome.lessons == (Recalled(record_id(OPEN_LESSON), OPEN_LESSON),)
    assert call.material.restrictions == Restrictions(available_at=TIME, labels=frozenset())
    grown = LEDGER.apply(declare("artifact:open-backtest", VENDOR, available_at=LATER))
    replayed = replace(toolset, ledger=grown).replay(call.to_bytes())
    assert replayed == call
    assert replayed.material == call.material
    assert replayed.to_bytes() == call.to_bytes()
    current = invoke(tools(world, ledger=grown, memory=grown.head), "read_lessons")
    assert current.outcome == Lessons(grown.head, ())
    cleared = invoke(
        tools(world, VENDOR, cutoff=LATER, ledger=grown, memory=grown.head), "read_lessons"
    )
    assert cleared.outcome.lessons == (
        Recalled(record_id(OPEN_LESSON), OPEN_LESSON),
        Recalled(record_id(VENDOR_LESSON), VENDOR_LESSON),
        Recalled(record_id(LATER_LESSON), LATER_LESSON),
        Recalled(record_id(FRESH_LESSON), FRESH_LESSON),
    )
    assert cleared.material.restrictions == Restrictions(available_at=LATER, labels={VENDOR})


@pytest.mark.parametrize(
    "initial, amended, current",
    [
        (
            Restrictions(available_at=EARLIER, labels=set(), approved_for=PROVIDERS),
            Restrictions(available_at=LATER, labels={VENDOR}, approved_for={FIRM}),
            Restrictions(available_at=LATER, labels={VENDOR}, approved_for={FIRM}),
        ),
        (
            Restrictions(available_at=LATER, labels={VENDOR}, approved_for={FIRM}),
            Restrictions(available_at=EARLIER, labels=set(), approved_for=PROVIDERS),
            Restrictions(available_at=LATER, labels={VENDOR}, approved_for={FIRM}),
        ),
    ],
    ids=("tightening", "attempted_loosening"),
)
def test_lesson_authority_and_material_replay_from_the_admitting_prefix(
    world, initial, amended, current
):
    evidence = "artifact:open-backtest"
    first = Declaration(evidence, initial, "ingestion", TIME, "Initial restrictions")
    pinned = Ledger((first, *admit(OPEN_LESSON)))
    toolset = tools(
        world, VENDOR, cutoff=LATER, ledger=pinned, memory=pinned.head, providers=PROVIDERS
    )
    call = invoke(toolset, "read_lessons")
    expected = Restrictions(
        available_at=LATER, labels=initial.labels, approved_for=initial.approved_for
    )
    assert call.outcome == Lessons(pinned.head, (Recalled(record_id(OPEN_LESSON), OPEN_LESSON),))
    assert call.restrictions == expected
    grown = pinned.apply(Declaration(evidence, amended, "ingestion", TIME, "Amended restrictions"))
    assert grown.admissible_lesson_restrictions(toolset.context) == (
        (record_id(OPEN_LESSON), initial),
    )
    later = replace(toolset, ledger=grown)
    assert invoke(later, "read_lessons") == call
    replayed = later.replay(call.to_bytes())
    assert replayed == call
    assert replayed.material == call.material
    assert replayed.to_bytes() == call.to_bytes()
    live_tools = tools(
        world, VENDOR, cutoff=LATER, ledger=grown, memory=grown.head, providers=PROVIDERS
    )
    assert grown.admissible_lesson_restrictions(live_tools.context) == (
        (record_id(OPEN_LESSON), current),
    )
    live = invoke(live_tools, "read_lessons")
    assert live.outcome.lessons == call.outcome.lessons
    assert live.restrictions == current
    assert live.material.restrictions == current
    assert toolset.context.send_denials(call.restrictions, SUBSCRIPTION) == (
        () if SUBSCRIPTION in initial.approved_for else ("provider_not_approved",)
    )
    assert toolset.context.send_denials(live.restrictions, SUBSCRIPTION) == (
        "provider_not_approved",
    )


def test_later_declared_lesson_evidence_stays_unreadable_at_the_old_pin(world):
    pinned = Ledger(admit(OPEN_LESSON))
    toolset = tools(world, ledger=pinned, memory=pinned.head)
    call = invoke(toolset, "read_lessons")
    assert call.outcome == Lessons(pinned.head, ())
    grown = pinned.apply(declare("artifact:open-backtest"))
    replayed = replace(toolset, ledger=grown).replay(call.to_bytes())
    assert replayed == call
    assert replayed.material == call.material
    assert replayed.to_bytes() == call.to_bytes()
    current = invoke(tools(world, ledger=grown, memory=grown.head), "read_lessons")
    assert current.outcome == Lessons(grown.head, (Recalled(record_id(OPEN_LESSON), OPEN_LESSON),))


def test_list_datasets_names_what_the_run_may_read_and_renders_no_restriction(world):
    toolset = tools(
        world, providers=PROVIDERS, scope=("vendor.prices", "firm.prices", "zq.missing")
    )
    call = invoke(toolset, "list_datasets")
    assert json.loads(call.material.text) == {
        "datasets": [
            {"dataset": name, "owner": "Data Engineering", "fields": ["close"]}
            for name in ("firm.prices", "vendor.prices")
        ]
    }
    assert json.loads(call.to_bytes())["outcome"] == {
        "listing": {"catalog": CATALOG.digest, "datasets": ["firm.prices", "vendor.prices"]}
    }
    for text in (call.material.text, call.to_bytes().decode()):
        for secret in ("synthetic-vendor", "vendor.restricted", "firm-proxy", "2026-10"):
            assert secret not in text
    assert CATALOG.digest not in call.material.text
    assert toolset.replay(call.to_bytes()) == call


def test_a_listing_goes_only_where_every_listed_dataset_may_go(world, tmp_path):
    subscription, firm = ScriptedProvider(["Listed"]), ScriptedProvider(["Listed"] * 2)
    send = gateway(tmp_path, [(SUBSCRIPTION_POLICY, subscription), (FIRM_POLICY, firm)])
    for scope, approved in (
        (("firm.prices", "shared.prices"), {FIRM}),
        (("shared.prices",), {FIRM, SUBSCRIPTION}),
    ):
        toolset = tools(world, providers=PROVIDERS, scope=scope)
        material = invoke(toolset, "list_datasets").material
        assert material.restrictions.approved_for == approved
        sent = send(toolset.context, SUBSCRIPTION_MODEL, material)
        assert isinstance(sent, Answer) is (SUBSCRIPTION in approved)
        assert isinstance(send(toolset.context, FIRM_MODEL, material), Answer)
    assert [call.texts for call in subscription.calls] == [(material.text,)]


def test_describe_dataset_renders_the_contract_parts_the_cutoff_admits(world):
    parts, corpus = contracted(world)
    october = datetime(2025, 12, 17, tzinfo=UTC)
    for cutoff, visible in ((october - timedelta(microseconds=1), parts[:1]), (october, parts)):
        toolset = tools(world, cutoff=cutoff, corpus=corpus, catalog=CONTRACTED)
        call = invoke(toolset, "describe_dataset", dataset=UNRATE)
        assert json.loads(call.material.text) == {"dataset": UNRATE, "units": units(visible)}
        assert json.loads(call.to_bytes())["outcome"] == {
            "description": {
                "catalog": CONTRACTED.digest,
                "dataset": UNRATE,
                "units": units(visible),
            }
        }
        assert ("2025-10-01" in call.material.text) is (cutoff == october)
        assert toolset.replay(call.to_bytes()) == call


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("read_dataset", {}),
        ("describe_dataset", {}),
        ("run_signal", {"source": "pass"}),
    ],
)
def test_a_hidden_dataset_is_indistinguishable_from_a_missing_one_in_every_tool(
    world, tmp_path, tool, arguments
):
    toolset = tools(
        world,
        providers=PROVIDERS,
        scope=("vendor.restricted", "firm.prices"),
        generated=generated(tmp_path),
    )
    calls = [
        invoke(toolset, tool, **arguments, dataset=name)
        for name in ("vendor.prices", "vendor.restricted", "zq.missing")
    ]
    assert {call.outcome for call in calls} == {Refusal("unknown_dataset")}
    assert all(without_arguments(call) == without_arguments(calls[0]) for call in calls)
    assert all(call.material == calls[0].material for call in calls)
    listed = json.loads(invoke(toolset, "list_datasets").material.text)["datasets"]
    assert [item["dataset"] for item in listed] == ["firm.prices"]


def test_replay_refuses_a_listing_after_the_catalog_revokes_a_dataset(world):
    toolset = tools(world)
    record = invoke(toolset, "list_datasets").to_bytes()
    revoked = Catalog((*CATALOG.events, Revoked("vendor.prices", "lead", TIME, "Leaked")))
    with pytest.raises(ToolError):
        replace(toolset, catalog=revoked).replay(record)
