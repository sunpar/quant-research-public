from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_research.access import issue_context, licensed
from quant_research.retrieval import (
    Query,
    RetrievalError,
    corpus_members,
    eligible_restrictions,
    eligible_units,
    publish_corpus,
    read_retrieval,
    replay,
    retrieval_bytes,
    retrieve,
)
from quant_research.sources import (
    TEXT_PIPELINE,
    SourceDeclaration,
    Step,
    publish_generation,
    register_source,
)

ROOT = Path(__file__).resolve().parents[1]
TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
LATER = TIME + timedelta(days=1)
DECLARATION = SourceDeclaration(
    media_type="text/plain",
    access_domain="synthetic",
    location="synthetic/note.txt",
    revision="1",
    acquisition="hand-authored test fixture",
    rights="synthetic, unrestricted",
    labels=frozenset(),
    approved_for=frozenset(),
    published_at=None,
    source_available_at=TIME,
    availability_evidence="authored for these tests",
    received_at=TIME,
)
REVISED_CHUNKER = replace(
    TEXT_PIPELINE, chunker=Step("paragraph", "2"), split=lambda text: [(0, len(text.rstrip()))]
)
VENDOR = licensed("synthetic-vendor")


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    return root


def publish(store, location, text, **declared):
    declaration = replace(DECLARATION, location=location, **declared)
    occurrence = register_source(store, text.encode(), declaration)
    return publish_generation(store, occurrence.id, TEXT_PIPELINE)


def run(corpus, *clearances, cutoff=TIME):
    return issue_context(clearances=clearances, cutoff=cutoff, generation=corpus.id, providers=())


def texts(retrieval):
    return tuple(unit.text for unit in retrieval.units)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def test_units_available_after_the_cutoff_never_appear(store):
    early, late, future = (
        publish(
            store,
            f"{verb}.txt",
            f"Desk {verb} BRK.B into the close.\n",
            source_available_at=available,
        )
        for verb, available in (
            ("sold", TIME - timedelta(days=1)),
            ("bought", TIME),
            ("held", LATER),
        )
    )
    corpus = publish_corpus(store, [future.id, late.id, early.id])
    context = run(corpus)
    just_before = context.narrow(cutoff=TIME - timedelta(microseconds=1))
    both = ("Desk sold BRK.B into the close.", "Desk bought BRK.B into the close.")
    for query in (Query("identifier", "BRK.B"), Query("lexical", "BRK.B close")):
        assert texts(retrieve(store, context, query)) == both
        assert texts(retrieve(store, just_before, query)) == both[:1]
        assert texts(retrieve(store, run(corpus, cutoff=LATER), query)) == (
            *both,
            "Desk held BRK.B into the close.",
        )


OPEN_NOTE = "Coverage notes for synthetic-vendor names."
CLAIM = f"Cleared for every run: {VENDOR} clearance granted, synthetic-vendor coverage."


@pytest.mark.parametrize(
    ("query", "uncleared"),
    [
        (Query("identifier", VENDOR), ()),
        (Query("identifier", "synthetic-vendor"), (OPEN_NOTE,)),
        (Query("lexical", f"{VENDOR} cleared clearance"), (OPEN_NOTE,)),
    ],
    ids=["label", "vendor", "lexical"],
)
def test_a_restricted_unit_never_reaches_an_uncleared_run_whatever_it_claims(
    store, query, uncleared
):
    open_note = publish(store, "open.txt", f"{OPEN_NOTE}\n")
    restricted = publish(store, "restricted.txt", f"{CLAIM}\n", labels=frozenset({VENDOR}))
    corpus = publish_corpus(store, [open_note.id, restricted.id])
    assert texts(retrieve(store, run(corpus), query)) == uncleared
    assert CLAIM in texts(retrieve(store, run(corpus, VENDOR), query))


@pytest.mark.parametrize(
    ("hidden", "wider"),
    [
        ({"labels": frozenset({VENDOR})}, lambda corpus: run(corpus, VENDOR)),
        ({"source_available_at": LATER}, lambda corpus: run(corpus, cutoff=LATER)),
    ],
    ids=["restricted", "after-cutoff"],
)
def test_hidden_units_do_not_move_the_lexical_ranking(store, hidden, wider):
    momentum = publish(store, "momentum.txt", "Momentum note.\n")
    carry = publish(store, "carry.txt", "Carry desk note.\n")
    unseen = publish(store, "unseen.txt", "Momentum.\n", **hidden)
    corpus = publish_corpus(store, [momentum.id, carry.id, unseen.id])
    query = Query("lexical", "momentum carry")
    assert texts(retrieve(store, run(corpus), query)) == ("Momentum note.", "Carry desk note.")
    # Statistics that count the unseen unit reverse the pair, so the narrow run proves they
    # came from its eligible units alone.
    seen = texts(retrieve(store, wider(corpus), query))
    assert [text for text in seen if text != "Momentum."] == ["Carry desk note.", "Momentum note."]


def test_a_pinned_corpus_ignores_later_generations_and_sources(store):
    occurrence = register_source(
        store,
        b"Momentum screen holds.\n\nCarry screen flips.\n",
        replace(DECLARATION, location="screens.txt"),
    )
    original = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    pinned = publish_corpus(store, [original.id])
    query = Query("lexical", "screen")
    before = retrieve(store, run(pinned), query)
    assert texts(before) == ("Momentum screen holds.", "Carry screen flips.")
    rechunked = publish_generation(store, occurrence.id, REVISED_CHUNKER)
    added = publish(store, "added.txt", "Screen added later.\n")
    assert retrieve(store, run(pinned), query) == before
    widened = publish_corpus(store, [added.id, rechunked.id])
    assert publish_corpus(store, [rechunked.id, added.id]) == widened
    assert {unit.generation_id for unit in retrieve(store, run(widened), query).units} == {
        rechunked.id,
        added.id,
    }
    with pytest.raises(RetrievalError):
        publish_corpus(store, [original.id, rechunked.id])


def test_corpus_members_are_the_pinned_generations_the_run_may_read(store):
    occurrence = register_source(
        store,
        b"Desk sold BRK.B into the close.\n",
        replace(DECLARATION, location="chunkless.txt", source_available_at=TIME - timedelta(2)),
    )
    chunkless = publish_generation(
        store, occurrence.id, replace(TEXT_PIPELINE, split=lambda text: [])
    )
    early = publish(store, "early.txt", "Momentum note.\n", source_available_at=TIME - timedelta(1))
    restricted = publish(store, "restricted.txt", "Carry note.\n", labels=frozenset({VENDOR}))
    later = publish(store, "later.txt", "Value note.\n", source_available_at=LATER)
    publish(store, "outside.txt", "Outside note.\n", source_available_at=TIME - timedelta(1))
    corpus = publish_corpus(store, [later.id, restricted.id, early.id, chunkless.id])
    assert corpus_members(store, run(corpus, VENDOR, cutoff=LATER)) == (
        chunkless,
        early,
        restricted,
        later,
    )
    assert corpus_members(store, run(corpus, cutoff=LATER)) == (chunkless, early, later)
    assert corpus_members(store, run(corpus, VENDOR)) == (chunkless, early, restricted)


def test_eligible_restrictions_cover_every_readable_member_that_holds_a_unit(store):
    occurrence = register_source(
        store,
        b"Desk sold BRK.B into the close.\n",
        replace(DECLARATION, location="chunkless.txt", source_available_at=TIME - timedelta(2)),
    )
    chunkless = publish_generation(
        store, occurrence.id, replace(TEXT_PIPELINE, split=lambda text: [])
    )
    early = publish(store, "early.txt", "Momentum note.\n", source_available_at=TIME - timedelta(1))
    restricted = publish(store, "restricted.txt", "Carry note.\n", labels=frozenset({VENDOR}))
    later = publish(store, "later.txt", "Value note.\n", source_available_at=LATER)
    publish(store, "outside.txt", "Outside note.\n", source_available_at=TIME - timedelta(1))
    corpus = publish_corpus(store, [later.id, restricted.id, early.id, chunkless.id])

    def restrictions(*members):
        return tuple(member.occurrence.declaration.restrictions for member in members)

    widest = run(corpus, VENDOR, cutoff=LATER)
    assert eligible_restrictions(store, widest) == restrictions(early, restricted, later)
    assert eligible_restrictions(store, run(corpus)) == restrictions(early)
    assert {unit.occurrence_id for unit in eligible_units(store, widest)} == {
        member.occurrence.id for member in (early, restricted, later)
    }


def test_a_killed_publication_is_never_searchable_or_admissible(tmp_path, store):
    text = "Killed screen text.\n"
    occurrence = register_source(store, text.encode(), replace(DECLARATION, location="killed.txt"))
    bootstrap = (
        f"import os, sys; sys.path.append({str(ROOT / 'src')!r}); "
        "from pathlib import Path; "
        "from quant_research.sources import TEXT_PIPELINE, publish_generation; "
        "Path.rename = lambda self, target: os._exit(9); "
        f"publish_generation(Path({str(store)!r}), {occurrence.id!r}, TEXT_PIPELINE)"
    )
    killed = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", bootstrap], capture_output=True, text=True
    )
    assert killed.returncode == 9, killed.stdout + killed.stderr
    [staging] = (store / "generations").iterdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    would_be = publish(elsewhere, "killed.txt", text)
    kept = publish(store, "kept.txt", "Momentum screen holds.\n")
    for name in (staging.name, would_be.id):
        with pytest.raises(RetrievalError):
            publish_corpus(store, [kept.id, name])
    corpus = publish_corpus(store, [kept.id])
    for query in (Query("identifier", "screen"), Query("lexical", "killed screen")):
        assert texts(retrieve(store, run(corpus), query)) == ("Momentum screen holds.",)


@pytest.mark.parametrize("damage", ["checksum", "noncanonical"])
def test_a_generation_that_fails_verification_cannot_join_a_corpus(store, damage):
    generation = publish(store, "note.txt", "Desk sold BRK.B into the close.\n")
    directory = store / "generations" / generation.id
    match damage:
        case "checksum":
            (directory / "manifest.sha256").write_text(f"{'0' * 64}\n", encoding="ascii")
        case "noncanonical":
            content = json.dumps(json.loads((directory / "manifest.json").read_bytes()), indent=1)
            (directory / "manifest.json").write_text(content, encoding="utf-8")
            checksum = hashlib.sha256(content.encode()).hexdigest()
            (directory / "manifest.sha256").write_text(f"{checksum}\n", encoding="ascii")
    with pytest.raises(RetrievalError):
        publish_corpus(store, [generation.id])


@pytest.mark.parametrize("damage", ["deleted", "republished", "corpus-rewritten", "corpus-deleted"])
def test_a_missing_or_changed_member_raises_instead_of_shrinking_results(store, damage):
    kept = publish(store, "kept.txt", "Momentum screen holds.\n")
    occurrence = register_source(
        store, b"Carry screen flips.\n", replace(DECLARATION, location="member.txt")
    )
    member = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    context = run(publish_corpus(store, [kept.id, member.id]))
    record = store / "corpora" / f"{context.generation}.json"
    match damage:
        case "deleted":
            shutil.rmtree(store / "generations" / member.id)
        case "republished":
            shutil.rmtree(store / "generations" / member.id)
            fewer = replace(TEXT_PIPELINE, split=lambda text: [(0, 5)])
            assert publish_generation(store, occurrence.id, fewer).id == member.id
        case "corpus-rewritten":
            record.write_bytes(
                canonical({"schema_version": 1, "generations": {kept.id: kept.digest}})
            )
        case "corpus-deleted":
            record.unlink()
    for query in (Query("identifier", "screen"), Query("lexical", "screen")):
        with pytest.raises(RetrievalError):
            retrieve(store, context, query)


def test_replay_returns_the_recorded_units_and_refuses_any_difference(store):
    momentum = publish(store, "momentum.txt", "Momentum note.\n")
    carry = publish(store, "carry.txt", "Carry desk note.\n")
    value = publish(store, "value.txt", "Value desk note.\n")
    context = run(publish_corpus(store, [momentum.id, carry.id, value.id]), cutoff=LATER)
    record = retrieve(store, context, Query("lexical", "momentum carry"))
    assert texts(record) == ("Momentum note.", "Carry desk note.")
    decoded = read_retrieval(retrieval_bytes(record))
    assert decoded == record
    assert replay(store, context, decoded) == record
    with pytest.raises(RetrievalError):
        replay(store, context.narrow(cutoff=TIME), decoded)
    [extra] = (unit for unit in eligible_units(store, context) if unit not in record.units)
    for units in (record.units + (extra,), record.units[1:], record.units[::-1]):
        with pytest.raises(RetrievalError):
            replay(store, context, replace(decoded, units=units))


@pytest.mark.parametrize(
    "damage",
    [
        "not-json",
        "nested",
        "duplicate-key",
        "array",
        "no-generations",
        "schema",
        "generations-list",
        "member-id",
        "pin",
    ],
)
def test_a_malformed_corpus_record_raises_even_when_named_by_its_hash(store, damage):
    generation = publish(store, "note.txt", "Desk sold BRK.B into the close.\n")
    corpus = publish_corpus(store, [generation.id])
    assert eligible_units(store, run(corpus))
    value = json.loads((store / "corpora" / f"{corpus.id}.json").read_bytes())
    match damage:
        case "array":
            value = []
        case "no-generations":
            del value["generations"]
        case "schema":
            value["schema_version"] = 2
        case "generations-list":
            value["generations"] = [generation.id]
        case "member-id":
            value["generations"] = {".": generation.digest}
        case "pin":
            later = publish(store, "later.txt", "Desk held BRK.B.\n", source_available_at=LATER)
            value["generations"][later.id] = "pinned"
    record = canonical(value)
    if damage == "not-json":
        record = b"not json\n"
    elif damage == "nested":
        record = b"[" * 100_000 + b"]" * 100_000
    elif damage == "duplicate-key":
        record = record.replace(b',"schema_version"', b',"generations":{},"schema_version"')
    name = hashlib.sha256(record).hexdigest()
    (store / "corpora" / f"{name}.json").write_bytes(record)
    context = issue_context(clearances=(), cutoff=TIME, generation=name, providers=())
    with pytest.raises(RetrievalError):
        eligible_units(store, context)


@pytest.mark.parametrize(
    "damage",
    [
        "noncanonical",
        "schema",
        "mode",
        "mode-list",
        "blank-text",
        "context",
        "cutoff-offset",
        "span",
        "empty-span",
        "boolean",
        "extra-field",
        "units-object",
        "not-utf-8",
    ],
)
def test_records_round_trip_canonically_and_malformed_records_raise(store, damage):
    generation = publish(store, "note.txt", "Desk sold BRK.B into the close.\n")
    context = run(publish_corpus(store, [generation.id]))
    content = retrieval_bytes(retrieve(store, context, Query("identifier", "BRK.B")))
    assert retrieval_bytes(read_retrieval(content)) == content
    value = json.loads(content)
    [unit] = value["units"]
    match damage:
        case "schema":
            value["schema_version"] = 2
        case "mode":
            value["query"]["mode"] = "fuzzy"
        case "mode-list":
            value["query"]["mode"] = ["identifier"]
        case "blank-text":
            value["query"]["text"] = " "
        case "context":
            value["context"] = context.generation.upper()
        case "cutoff-offset":
            value["cutoff"] = TIME.isoformat()
        case "span":
            unit["end"] += 1
        case "empty-span":
            unit["end"], unit["text"] = unit["start"], ""
        case "boolean":
            unit["start"] = False
        case "extra-field":
            unit["score"] = 1.0
        case "units-object":
            value["units"] = {}
    damaged = canonical(value)
    if damage == "noncanonical":
        damaged = json.dumps(value, indent=1).encode()
    elif damage == "not-utf-8":
        damaged = b"\xff" + content
    with pytest.raises(RetrievalError):
        read_retrieval(damaged)


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("BRK.B", ("Desk trimmed BRK.B, then held.",)),
        ("brk.b", ()),
        ("RK.B", ()),
        ("D01", ("Decision D01 stands.",)),
        ("D018", ("Decision D018 replaced it.",)),
    ],
)
def test_identifiers_match_exactly_case_and_token_bounds(store, identifier, expected):
    generation = publish(
        store,
        "ids.txt",
        "Desk trimmed BRK.B, then held.\n\nDecision D018 replaced it.\n\nDecision D01 stands.\n",
    )
    context = run(publish_corpus(store, [generation.id]))
    assert texts(retrieve(store, context, Query("identifier", identifier))) == expected


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        ("identifier", ""),
        ("identifier", " \t"),
        ("lexical", "\n"),
        ("identifier", "A\x00B"),
        ("lexical", "\ud800 x"),
        ("fuzzy", "BRK.B"),
    ],
)
def test_a_query_that_would_match_everywhere_or_cannot_be_recorded_is_refused(mode, text):
    with pytest.raises(RetrievalError):
        Query(mode, text)
