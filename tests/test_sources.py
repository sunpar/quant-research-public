from __future__ import annotations

import errno
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_research.access import (
    CONFIRMATION,
    INTERNAL_CODE,
    Provider,
    Restrictions,
    issue_context,
    licensed,
)
from quant_research.sources import (
    TEXT_PIPELINE,
    Chunk,
    IdentityConflict,
    SourceDeclaration,
    SourceError,
    Step,
    publish_generation,
    published_generations,
    register_source,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "sources"
NOTE = (FIXTURES / "note.txt").read_bytes()
TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
DECLARATION = SourceDeclaration(
    media_type="text/plain",
    access_domain="synthetic",
    location="tests/fixtures/sources/note.txt",
    revision="1",
    acquisition="hand-authored test fixture",
    rights="synthetic, unrestricted",
    labels=frozenset(),
    approved_for=frozenset(),
    published_at=TIME - timedelta(days=1),
    source_available_at=TIME,
    availability_evidence="authored for these tests",
    received_at=TIME,
)
REVISED_CHUNKER = replace(
    TEXT_PIPELINE, chunker=Step("paragraph", "2"), split=lambda text: [(0, len(text.rstrip()))]
)
PADDED_TEXT = replace(TEXT_PIPELINE, extract=lambda data, media_type: data.decode() + "\n")
VENDOR = licensed("synthetic-vendor")


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    return root


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def run(*clearances, cutoff=TIME):
    return issue_context(
        clearances=clearances, cutoff=cutoff, generation="synthetic/v1", providers=()
    )


def rewrite_manifest(directory: Path, content: bytes) -> None:
    (directory / "manifest.json").write_bytes(content)
    (directory / "manifest.sha256").write_text(hashlib.sha256(content).hexdigest() + "\n")


def test_identical_bytes_under_two_declarations_keep_one_blob_and_two_occurrences(store):
    first = register_source(store, NOTE, DECLARATION)
    mirror = replace(
        DECLARATION,
        access_domain="licensed mirror",
        location="mirror/note.txt",
        labels=frozenset({licensed("mirror")}),
    )
    second = register_source(store, NOTE, mirror)
    shifted = replace(DECLARATION, received_at=TIME.astimezone(timezone(timedelta(hours=2))))
    assert register_source(store, NOTE, DECLARATION) == first
    assert register_source(store, NOTE, shifted) == first
    assert first.blob_digest == second.blob_digest == hashlib.sha256(NOTE).hexdigest()
    assert first.id != second.id
    assert [path.name for path in (store / "blobs").iterdir()] == [first.blob_digest]
    generations = [
        publish_generation(store, occurrence.id, TEXT_PIPELINE) for occurrence in (first, second)
    ]
    assert generations[0].id != generations[1].id
    assert generations[0].chunks == generations[1].chunks
    assert set(published_generations(store, run(licensed("mirror")))) == set(generations)
    assert published_generations(store, run()) == (generations[0],)


def test_same_declaration_with_other_bytes_conflicts_and_keeps_the_original(store):
    occurrence = register_source(store, NOTE, DECLARATION)
    generation = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    with pytest.raises(IdentityConflict):
        register_source(store, NOTE.replace(b"4%", b"5%"), DECLARATION)
    assert [path.name for path in (store / "blobs").iterdir()] == [occurrence.blob_digest]
    assert register_source(store, NOTE, DECLARATION) == occurrence
    assert published_generations(store, run()) == (generation,)


def test_declarations_that_differ_only_in_provider_approvals_are_two_occurrences(store):
    approved = replace(DECLARATION, approved_for=frozenset({Provider("local-model")}))
    first, second = (register_source(store, NOTE, item) for item in (DECLARATION, approved))
    assert first.id != second.id
    assert len(list((store / "occurrences").iterdir())) == 2


@pytest.mark.parametrize(
    ("name", "media_type", "expected"),
    [
        (
            "note.txt",
            "text/plain",
            (
                "Synthetic quarterly note for source registry tests.",
                "The café index rose 4% while the € basket held steady.",
                "Volumes stayed flat across the synthetic venues.",
                "No figures here describe a real market.",
            ),
        ),
        (
            "memo.md",
            "text/markdown",
            (
                "# Desk memo",
                "Résumé: the naïve momentum screen stays synthetic.",
                "- Rebalance the synthetic book on Fridays.",
                "> Nothing here is investment advice.",
            ),
        ),
    ],
)
def test_text_pipeline_locates_fixture_paragraphs_by_code_point(store, name, media_type, expected):
    data = (FIXTURES / name).read_bytes()
    declaration = replace(DECLARATION, media_type=media_type, location=name)
    occurrence = register_source(store, data, declaration)
    generation = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    original = data.decode("utf-8")
    assert tuple(chunk.text for chunk in generation.chunks) == expected
    assert tuple(original[chunk.start : chunk.end] for chunk in generation.chunks) == expected
    last = generation.chunks[-1]
    assert data[last.start : last.end] != last.text.encode()
    assert (generation.text_digest, generation.length) == (occurrence.blob_digest, len(original))


def test_paragraphs_join_adjacent_lines_and_trim_outer_whitespace(store):
    text = "  Café opening\n  continues indented.  \n \t \nCRLF first\r\nCRLF second\r\n\r\nLast"
    occurrence = register_source(store, text.encode(), DECLARATION)
    assert publish_generation(store, occurrence.id, TEXT_PIPELINE).chunks == (
        Chunk(2, 36, "Café opening\n  continues indented."),
        Chunk(43, 66, "CRLF first\r\nCRLF second"),
        Chunk(70, 74, "Last"),
    )


def test_retries_and_races_keep_the_first_recorded_time_and_stores_share_content(tmp_path, store):
    occurrence = register_source(store, NOTE, DECLARATION)
    before = datetime.now(UTC)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            workers.map(lambda _: publish_generation(store, occurrence.id, TEXT_PIPELINE), range(4))
        )
    after = datetime.now(UTC)
    generation = results[0]
    assert results == [generation] * 4
    assert before <= generation.recorded_at <= after
    retry = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    assert retry == generation
    assert retry.recorded_at == generation.recorded_at
    assert published_generations(store, run()) == (generation,)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    fresh = register_source(elsewhere, NOTE, DECLARATION)
    copy = publish_generation(elsewhere, fresh.id, TEXT_PIPELINE)
    assert (fresh.id, copy.id, copy.digest) == (occurrence.id, generation.id, generation.digest)
    assert copy.recorded_at > generation.recorded_at
    assert copy != generation
    assert replace(generation, recorded_at=copy.recorded_at) == copy


def test_failed_processing_publishes_nothing(store):
    occurrence = register_source(store, NOTE, DECLARATION)

    def crashing_split(text):
        yield 0, 1
        raise RuntimeError("chunker crashed")

    with pytest.raises(RuntimeError, match="chunker crashed"):
        publish_generation(store, occurrence.id, replace(TEXT_PIPELINE, split=crashing_split))
    assert published_generations(store, run()) == ()
    generation = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    assert published_generations(store, run()) == (generation,)


def test_a_process_killed_before_publication_leaves_nothing_visible(store):
    occurrence = register_source(store, NOTE, DECLARATION)
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
    assert list((store / "generations").iterdir())
    assert published_generations(store, run()) == ()
    resumed = datetime.now(UTC)
    generation = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    assert generation.recorded_at >= resumed
    assert published_generations(store, run()) == (generation,)


def test_failed_rename_publishes_nothing_and_a_retry_succeeds(store, monkeypatch):
    occurrence = register_source(store, NOTE, DECLARATION)

    def failing_rename(self, target):
        raise OSError(errno.EIO, "injected rename failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", failing_rename)
        with pytest.raises(SourceError, match="injected rename failure"):
            publish_generation(store, occurrence.id, TEXT_PIPELINE)
    assert published_generations(store, run()) == ()
    generation = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    assert published_generations(store, run()) == (generation,)


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "noncanonical",
        "chunker",
        "length",
        "text",
        "occurrence",
        "blob",
        "binding-deleted",
        "binding-changed",
        "name",
        "labels-string",
        "label-deleted",
        "publication-offset",
        "publication-deleted",
    ],
)
def test_published_generations_reject_damaged_publications(store, damage):
    occurrence = register_source(store, NOTE, replace(DECLARATION, labels=frozenset({VENDOR})))
    directory = store / "generations" / publish_generation(store, occurrence.id, TEXT_PIPELINE).id
    manifest = json.loads((directory / "manifest.json").read_bytes())
    match damage:
        case "checksum":
            (directory / "manifest.sha256").write_text("0" * 64 + "\n")
        case "noncanonical":
            rewrite_manifest(directory, json.dumps(manifest, indent=1).encode())
        case "chunker":
            manifest["chunker"]["version"] = "2"
            rewrite_manifest(directory, canonical(manifest))
        case "length":
            manifest["extraction"]["length"] = manifest["chunks"][-1]["end"] - 1
            rewrite_manifest(directory, canonical(manifest))
        case "text":
            manifest["chunks"][0]["text"] += "!"
            rewrite_manifest(directory, canonical(manifest))
        case "occurrence":
            path = store / "occurrences" / f"{occurrence.id}.json"
            record = json.loads(path.read_bytes())
            record["declaration"]["rights"] = "restricted"
            path.write_bytes(canonical(record))
        case "blob":
            path = store / "occurrences" / f"{occurrence.id}.json"
            record = json.loads(path.read_bytes())
            record["blob_digest"] = hashlib.sha256(b"other bytes").hexdigest()
            path.write_bytes(canonical(record))
        case "binding-deleted":
            next((store / "extractions").iterdir()).unlink()
        case "binding-changed":
            binding = next((store / "extractions").iterdir())
            binding.write_bytes(canonical({"length": 0, "text_digest": "0" * 64}))
        case "name":
            (store / "generations" / "unpublished").mkdir()
        case "labels-string" | "label-deleted":
            path = store / "occurrences" / f"{occurrence.id}.json"
            record = json.loads(path.read_bytes())
            record["declaration"]["labels"] = VENDOR if damage == "labels-string" else []
            path.write_bytes(canonical(record))
        case "publication-offset":
            publication = json.loads((directory / "publication.json").read_bytes())
            offset = datetime.fromisoformat(publication["recorded_at"]).isoformat()
            (directory / "publication.json").write_bytes(canonical({"recorded_at": offset}))
        case "publication-deleted":
            (directory / "publication.json").unlink()
    with pytest.raises(SourceError):
        published_generations(store, run())


class MisencodingText(str):
    """Report an encoding for text that has none."""

    def encode(self, *args, **kwargs):
        return b"text"


PUBLICATION_DAMAGE = {
    "extracted-surrogate": {"extract": lambda data, media_type: f"{data.decode()}\ud800"},
    "extracted-bytes": {"extract": lambda data, media_type: data},
    "misencoding-text": {
        "extract": lambda data, media_type: MisencodingText(f"{data.decode()}\ud800")
    },
    "extracted-named-type": {"extract": lambda data, media_type: type(data.decode(), (), {})()},
    "chunk-text": {"split": lambda text: [text]},
    "chunk-pair-text": {"split": lambda text: [(0, text)]},
}


@pytest.mark.parametrize(
    "damage", ["truncated", "non-utf-8", "surrogate", "source", *PUBLICATION_DAMAGE]
)
def test_errors_keep_restricted_text_out_of_their_exception_chains(store, damage, reachable_text):
    text = b"Licensed zebra-vault paragraph.\n"
    declaration = replace(DECLARATION, labels=frozenset({VENDOR}))
    if damage == "source":
        occurrence = register_source(store, text + b"\xff", declaration)
        attempt = partial(publish_generation, store, occurrence.id, TEXT_PIPELINE)
    elif damage in PUBLICATION_DAMAGE:
        occurrence = register_source(store, text, declaration)
        pipeline = replace(TEXT_PIPELINE, **PUBLICATION_DAMAGE[damage])
        attempt = partial(publish_generation, store, occurrence.id, pipeline)
    else:
        occurrence = register_source(store, text, declaration)
        generation = publish_generation(store, occurrence.id, TEXT_PIPELINE)
        directory = store / "generations" / generation.id
        content = (directory / "manifest.json").read_bytes()
        damaged = {
            "truncated": content[:-3],
            "non-utf-8": content + b"\xff",
            "surrogate": content.replace(b"paragraph.", b"paragraph.\\ud800"),
        }
        rewrite_manifest(directory, damaged[damage])
        attempt = partial(published_generations, store, run())
    with pytest.raises(SourceError) as caught:
        attempt()
    assert "zebra-vault" not in reachable_text(caught.value)


@pytest.mark.parametrize("record", ["blob", "occurrence"])
def test_changed_stored_records_are_rejected_on_reuse(store, record):
    occurrence = register_source(store, NOTE, DECLARATION)
    if record == "blob":
        (store / "blobs" / occurrence.blob_digest).write_bytes(NOTE.replace(b"4%", b"5%"))
    else:
        path = store / "occurrences" / f"{occurrence.id}.json"
        value = json.loads(path.read_bytes())
        value["declaration"]["rights"] = "restricted"
        path.write_bytes(canonical(value))
    with pytest.raises(SourceError):
        register_source(store, NOTE, DECLARATION)
    with pytest.raises(SourceError):
        publish_generation(store, occurrence.id, TEXT_PIPELINE)
    assert published_generations(store, run()) == ()


@pytest.mark.parametrize(
    "altered",
    [
        replace(TEXT_PIPELINE, split=REVISED_CHUNKER.split),
        PADDED_TEXT,
        replace(REVISED_CHUNKER, extract=PADDED_TEXT.extract),
    ],
    ids=["split", "extract", "extract-with-new-chunker"],
)
def test_changed_output_under_unchanged_steps_conflicts_and_keeps_the_original(store, altered):
    occurrence = register_source(store, NOTE, DECLARATION)
    original = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    with pytest.raises(IdentityConflict):
        publish_generation(store, occurrence.id, altered)
    assert published_generations(store, run()) == (original,)


def test_concurrent_conflicting_extractions_publish_one_text(store):
    occurrence = register_source(store, NOTE, DECLARATION)

    def publish(pipeline):
        try:
            return publish_generation(store, occurrence.id, pipeline)
        except IdentityConflict:
            return None

    pipelines = [TEXT_PIPELINE, replace(REVISED_CHUNKER, extract=PADDED_TEXT.extract)] * 4
    with ThreadPoolExecutor(max_workers=len(pipelines)) as workers:
        results = list(workers.map(publish, pipelines))
    winners = {result for result in results if result is not None}
    assert len(winners) == 1
    assert results.count(None) == 4
    assert published_generations(store, run()) == tuple(winners)


@pytest.mark.smoke
def test_new_step_versions_publish_beside_the_unchanged_original(store):
    data = (FIXTURES / "memo.md").read_bytes()
    declaration = replace(DECLARATION, media_type="text/markdown", location="memo.md")
    occurrence = register_source(store, data, declaration)
    original = publish_generation(store, occurrence.id, TEXT_PIPELINE)
    rechunked = publish_generation(store, occurrence.id, REVISED_CHUNKER)
    reextracted = publish_generation(
        store, occurrence.id, replace(TEXT_PIPELINE, extractor=Step("utf-8-text", "2"))
    )
    assert len({original.id, rechunked.id, reextracted.id}) == 3
    assert original.extraction_id == rechunked.extraction_id != reextracted.extraction_id
    trimmed = data.decode("utf-8").rstrip()
    assert rechunked.chunks == (Chunk(0, len(trimmed), trimmed),)
    assert reextracted.chunks == original.chunks
    listed = published_generations(store, run())
    assert set(listed) == {original, rechunked, reextracted}
    assert listed == tuple(sorted(listed, key=lambda generation: generation.id))


def test_cutoff_visibility_follows_source_availability_alone(store):
    late_available = datetime(2026, 6, 1, 9, 30, tzinfo=UTC)
    late = register_source(
        store,
        b"Old publication claim, late arrival.\n",
        replace(
            DECLARATION,
            published_at=datetime(2020, 1, 2, tzinfo=UTC),
            source_available_at=late_available,
            received_at=late_available + timedelta(minutes=1),
        ),
    )
    backfill = register_source(
        store,
        b"Archived long ago, ingested today.\n",
        replace(
            DECLARATION,
            published_at=None,
            source_available_at=datetime(2018, 5, 1, tzinfo=UTC),
            received_at=TIME,
        ),
    )
    late_generation, backfill_generation = (
        publish_generation(store, occurrence.id, TEXT_PIPELINE) for occurrence in (late, backfill)
    )
    both = (backfill_generation, late_generation)
    assert published_generations(store, run(cutoff=datetime(2021, 1, 1, tzinfo=UTC))) == (
        backfill_generation,
    )
    just_before = run().narrow(cutoff=late_available - timedelta(microseconds=1))
    assert published_generations(store, just_before) == (backfill_generation,)
    assert published_generations(store, run(cutoff=late_available)) == both
    eastern = late_available.astimezone(timezone(timedelta(hours=-5)))
    assert published_generations(store, run(cutoff=eastern)) == both
    with pytest.raises(SourceError):
        published_generations(store, late_available.replace(tzinfo=None))
    naive = replace(DECLARATION, received_at=TIME.replace(tzinfo=None))
    with pytest.raises(SourceError, match="timezone"):
        register_source(store, b"Naive arrival time.\n", naive)


def test_a_context_reads_only_the_generations_it_is_cleared_for(store):
    local = frozenset({Provider("local-model")})
    generations = []
    for index, labels in enumerate(((), (VENDOR,), (VENDOR, INTERNAL_CODE), (CONFIRMATION,))):
        declaration = replace(
            DECLARATION,
            labels=frozenset(labels),
            approved_for=local,
            source_available_at=TIME - timedelta(hours=4 - index),
        )
        occurrence = register_source(store, f"Source {index}.\n".encode(), declaration)
        generations.append(publish_generation(store, occurrence.id, TEXT_PIPELINE))
    unrestricted, vendor, internal, _confirmation = generations
    assert published_generations(store, run()) == (unrestricted,)
    assert published_generations(store, run(VENDOR)) == (unrestricted, vendor)
    widest = published_generations(store, run(VENDOR, INTERNAL_CODE))
    assert widest == (unrestricted, vendor, internal)
    assert widest[-1].occurrence.declaration.restrictions == Restrictions(
        available_at=TIME - timedelta(hours=2),
        labels=frozenset({VENDOR, INTERNAL_CODE}),
        approved_for=local,
    )


@pytest.mark.parametrize(
    "attempt",
    [
        lambda store: published_generations(store, TIME),
        lambda store: published_generations(store, SimpleNamespace(read_denials=lambda item: ())),
        lambda store: register_source(
            store, b"Other.\n", replace(DECLARATION, labels="internal_code")
        ),
        lambda store: register_source(
            store, b"Other.\n", replace(DECLARATION, labels=frozenset({""}))
        ),
        lambda store: register_source(
            store, b"Other.\n", replace(DECLARATION, approved_for="local-model")
        ),
    ],
    ids=["datetime", "stand-in", "labels-string", "blank-label", "approved-for-string"],
)
def test_malformed_policy_input_raises_and_stores_nothing(store, attempt):
    occurrence = register_source(store, NOTE, replace(DECLARATION, labels=frozenset({VENDOR})))
    publish_generation(store, occurrence.id, TEXT_PIPELINE)
    stored = sorted(store.rglob("*"))
    with pytest.raises(SourceError):
        attempt(store)
    assert sorted(store.rglob("*")) == stored


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("negative", "span"),
        ("beyond", "span"),
        ("empty", "span"),
        ("unsorted", "span"),
        ("duplicate", "span"),
        ("boolean", "span"),
        ("utf-8", "UTF-8"),
        ("media-type", "media type"),
        ("surrogate", "encode"),
        ("not-text", "did not return str"),
    ],
)
def test_invalid_processing_output_publishes_nothing(store, damage, message):
    data, media_type = b"First paragraph.\n\nSecond paragraph.\n", "text/plain"
    changes = {}
    match damage:
        case "negative":
            changes["split"] = lambda text: [(-1, 5)]
        case "beyond":
            changes["split"] = lambda text: [(18, 37)]
        case "empty":
            changes["split"] = lambda text: [(3, 3)]
        case "unsorted":
            changes["split"] = lambda text: [(18, 35), (0, 16)]
        case "duplicate":
            changes["split"] = lambda text: [(0, 16), (0, 16)]
        case "boolean":
            changes["split"] = lambda text: [(False, True)]
        case "utf-8":
            data = b"Caf\xe9 au lait.\n"
        case "media-type":
            media_type = "application/pdf"
        case "surrogate":
            changes["extract"] = lambda data, media_type: "Lone \ud800 surrogate."
        case "not-text":
            changes["extract"] = lambda data, media_type: None
    pipeline = replace(TEXT_PIPELINE, **changes)
    occurrence = register_source(store, data, replace(DECLARATION, media_type=media_type))
    with pytest.raises(SourceError, match=message):
        publish_generation(store, occurrence.id, pipeline)
    assert published_generations(store, run()) == ()


def test_store_root_must_exist_and_blank_sources_publish_empty_generations(tmp_path, store):
    missing = tmp_path / "missing"
    with pytest.raises(SourceError):
        register_source(missing, NOTE, DECLARATION)
    with pytest.raises(SourceError):
        publish_generation(missing, "0" * 64, TEXT_PIPELINE)
    with pytest.raises(SourceError):
        published_generations(missing, run())
    with pytest.raises(SourceError):
        publish_generation(tmp_path / "missing\ud800", "0" * 64, TEXT_PIPELINE)
    assert not missing.exists()
    assert published_generations(store, run()) == ()
    for unregistered in ("0" * 64, "../outside"):
        with pytest.raises(SourceError):
            publish_generation(store, unregistered, TEXT_PIPELINE)
    blank = register_source(store, b" \n\t\r\n", DECLARATION)
    generation = publish_generation(store, blank.id, TEXT_PIPELINE)
    assert (generation.chunks, generation.length) == ((), 5)
    assert published_generations(store, run()) == (generation,)
