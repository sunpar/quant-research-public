from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_research.access import issue_context, licensed
from quant_research.locators import (
    MARKDOWN_LOCATOR,
    MARKDOWN_LOCATOR_V2,
    Location,
    publish_locations,
    published_locations,
)
from quant_research.sources import (
    TEXT_PIPELINE,
    IdentityConflict,
    SourceDeclaration,
    SourceError,
    Step,
    publish_generation,
    published_generations,
    register_source,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "locators"
PAPER = (FIXTURES / "paper.md").read_bytes()
TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
DECLARATION = SourceDeclaration(
    media_type="text/markdown",
    access_domain="synthetic",
    location="tests/fixtures/locators/paper.md",
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
PAPER_ELEMENTS = (
    ("claim", "We find that the top decile earns 1.2% per month, e.g. in calm regimes."),
    (
        "claim",
        r"The model is $r_t = \alpha + \beta m_t$ and we estimate \(\beta = 0.8\) on 40 venues.",
    ),
    ("equation", r"$r_t = \alpha + \beta m_t$"),
    ("equation", r"\(\beta = 0.8\)"),
    ("equation", "$q$"),
    ("equation", "\n".join(["$$", r"r_t = \alpha + \beta m_t + \varepsilon_t", "$$"])),
    ("equation", r"\[ \hat\beta = (X'X)^{-1} X'y \]"),
    ("equation", "\n".join([r"\begin{equation}", r"SR = \mu / \sigma", r"\end{equation}"])),
    (
        "table",
        "\n".join(
            [
                "| Decile | Return | t-stat |",
                "| :----- | -----: | -----: |",
                "| 1      | 0.4%   | 1.1    |",
                "| 10     | 1.6%   | 3.2    |",
            ]
        ),
    ),
    ("figure", '![Cumulative returns](figures/cumulative.png "Synthetic signal")'),
    ("figure", "Figure 1: Cumulative returns of the top decile, which outperforms by 3%."),
    ("claim", "The spread is statistically significant (see Fig. 2 and Eq. 1)."),
    ("claim", r"The premium is $2\%$ per month."),
    ("equation", r"$2\%$"),
    ("claim", "Results suggest the signal survives costs."),
    ("claim", "Turnover averages 35% a year."),
    ("claim", "Costs take 150 basis points."),
    ("claim", "We document a 2% drift in the synthetic book."),
    ("claim", "We report a t-statistic of 2.4!"),
)
PADDED_TEXT = replace(MARKDOWN_LOCATOR, extract=lambda data, media_type: data.decode() + "\n")
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


def extract(store, data=PAPER, declaration=DECLARATION):
    """Register data and publish its paragraph generation, which publishes the extraction."""
    occurrence = register_source(store, data, declaration)
    return occurrence, publish_generation(store, occurrence.id, TEXT_PIPELINE)


def crashing_locate(text):
    yield "claim", 0, 5
    raise RuntimeError("locator crashed")


@pytest.mark.smoke
def test_markdown_locator_names_every_element_outside_code_in_order(store):
    occurrence, _ = extract(store)
    generation = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    original = PAPER.decode("utf-8")
    assert tuple((item.kind, item.text) for item in generation.locations) == PAPER_ELEMENTS
    assert all(original[item.start : item.end] == item.text for item in generation.locations)
    assert (generation.locator, generation.length) == (
        Step("markdown-elements", "1"),
        len(original),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Intro.\n\n```\nWe find 9% here.\n```\n\nWe find 5% gains.\n\n| a | b |\n|---|---|\n"
            "| 1 | 2 |\n".replace("\n", "\r\n"),
            (("claim", "We find 5% gains."), ("table", "| a | b |\r\n|---|---|\r\n| 1 | 2 |")),
        ),
        ("```x``` is inline code.\n\nWe find 5% gains.\n", (("claim", "We find 5% gains."),)),
        ("﻿# Returns rose 5%\n\nWe find 5% gains.\n", (("claim", "We find 5% gains."),)),
        (
            "![a](fig_(1).png) and ![Returns rose 5%](fig.png 'T') and ![a](<my fig.png>) "
            "and ![a [b] c](fig.png (T)).\n",
            (
                ("figure", "![a](fig_(1).png)"),
                ("figure", "![Returns rose 5%](fig.png 'T')"),
                ("figure", "![a](<my fig.png>)"),
                ("figure", "![a [b] c](fig.png (T))"),
            ),
        ),
    ],
    ids=["crlf", "line-leading-code-span", "byte-order-mark", "image-forms"],
)
def test_markdown_locator_reads_line_endings_and_line_leading_backticks(store, text, expected):
    occurrence, _ = extract(store, text.encode())
    generation = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    assert tuple((item.kind, item.text) for item in generation.locations) == expected
    assert all(text[item.start : item.end] == item.text for item in generation.locations)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "![Full][f1] and ![Collapsed][] and ![Short] but not ![Missing][nope].\n\n"
            "[f1]: figures/a.png\n"
            "[collapsed]: <figures/b c.png> 'B'\n"
            '[  SHORT ]: figures/c.png "We find 5% gains."\n'
            "[ ]: figures/d.png\n\n"
            "[^1]: Significant.\n\n"
            "![] is no figure.\n",
            (
                ("figure", "![Full][f1]"),
                ("figure", "![Collapsed][]"),
                ("figure", "![Short]"),
                ("claim", "[^1]: Significant."),
            ),
        ),
        (
            "> | a | b |\n> |---|---|\n> | 1 | 2% |\n>\n> We find 5% gains.\n",
            (("table", "| a | b |\n> |---|---|\n> | 1 | 2% |"), ("claim", "We find 5% gains.")),
        ),
        (
            "Returns rose 5%\n===============\n\nWe find 5% gains.\n\n---\n\nCosts fell 2%\n---\n\n"
            "- Item.\nWe report 3% drift.\n---\n",
            (("claim", "We find 5% gains."), ("claim", "We report 3% drift.")),
        ),
        (
            "<!-- We find 9% here.\n\nStill 3% hidden. -->\n\nWe find 5% gains.\n",
            (("claim", "We find 5% gains."),),
        ),
        (
            "1.  Item.\n\n    We find 5% gains.\n\nCode follows.\n\n    x = 9%  # we find\n",
            (("claim", "We find 5% gains."),),
        ),
        ("# Returns rose 5%\r\rWe find 5% gains.\r", (("claim", "We find 5% gains."),)),
        (
            "Returns rose 5%\r\n===============\r\n\r\nWe find 5% gains.\r\n",
            (("claim", "We find 5% gains."),),
        ),
    ],
    ids=[
        "reference-images",
        "quoted-table",
        "setext-heading",
        "html-comment",
        "indented-code",
        "carriage-returns",
        "crlf-setext-heading",
    ],
)
def test_markdown_locator_version_2_reads_forms_version_1_leaves_unread(store, text, expected):
    occurrence, _ = extract(store, text.encode())
    revised = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR_V2)
    original = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    assert tuple((item.kind, item.text) for item in revised.locations) == expected
    assert all(text[item.start : item.end] == item.text for item in revised.locations)
    assert tuple((item.kind, item.text) for item in original.locations) != expected


@pytest.mark.parametrize(
    "text",
    [
        "> " * 26 + "| a |\n",
        "<!-- open\n" * 16_000,
        "[" + "a" * 40_000 + "\n",
        "> | a |\n>" + " " * 16_000 + "-|x\n",
        "| a |\n" + " " * 16_000 + "-|x\n",
        "|" * 8_000 + "\n" + "-|" * 8_000 + "x\n",
        "> " + "|" * 8_000 + "\n> " + "-|" * 8_000 + "x\n",
    ],
    ids=[
        "nested-quotes",
        "unclosed-comments",
        "unclosed-label",
        "quoted-delimiter-indent",
        "delimiter-indent",
        "header-pipes",
        "quoted-header-pipes",
    ],
)
def test_markdown_locator_version_2_reads_adversarial_markdown_in_linear_time(text):
    start = time.perf_counter()
    MARKDOWN_LOCATOR_V2.locate(text)
    assert time.perf_counter() - start < 1


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "beyond",
        "text",
        "kind",
        "unsorted",
        "duplicate",
        "binding",
        "occurrence",
        "publication-deleted",
    ],
)
def test_published_locations_reject_damaged_publications(store, damage):
    occurrence, _ = extract(store)
    directory = store / "locations" / publish_locations(store, occurrence.id, MARKDOWN_LOCATOR).id
    manifest = json.loads((directory / "manifest.json").read_bytes())
    locations = manifest["locations"]
    match damage:
        case "checksum":
            (directory / "manifest.sha256").write_text("0" * 64 + "\n")
        case "beyond":
            beyond = manifest["extraction"]["length"] + 1
            locations[-1]["text"] += "!" * (beyond - locations[-1]["end"])
            locations[-1]["end"] = beyond
            rewrite_manifest(directory, canonical(manifest))
        case "text":
            locations[0]["text"] += "!"
            rewrite_manifest(directory, canonical(manifest))
        case "kind":
            locations[0]["kind"] = "theorem"
            rewrite_manifest(directory, canonical(manifest))
        case "unsorted":
            locations[0], locations[1] = locations[1], locations[0]
            rewrite_manifest(directory, canonical(manifest))
        case "duplicate":
            locations.insert(0, locations[0])
            rewrite_manifest(directory, canonical(manifest))
        case "binding":
            binding = next((store / "extractions").iterdir())
            binding.write_bytes(canonical({"length": 0, "text_digest": "0" * 64}))
        case "occurrence":
            path = store / "occurrences" / f"{occurrence.id}.json"
            record = json.loads(path.read_bytes())
            record["declaration"]["rights"] = "restricted"
            path.write_bytes(canonical(record))
        case "publication-deleted":
            (directory / "publication.json").unlink()
    with pytest.raises(SourceError):
        published_locations(store, run())


def test_ref_lookup_compares_at_most_one_location_however_large_the_generation(store, monkeypatch):
    data = b"x" * 10_000 + b"\n"
    occurrence, _ = extract(store, data, replace(DECLARATION, media_type="text/plain"))
    claims = replace(
        MARKDOWN_LOCATOR, locate=lambda text: [("claim", i, i + 1) for i in range(10_000)]
    )
    generation = publish_locations(store, occurrence.id, claims)
    last = replace(generation.locations[-1])
    compared = 0
    equal = Location.__eq__

    def counted(self, other):
        nonlocal compared
        compared += 1
        return equal(self, other)

    with monkeypatch.context() as patch:
        patch.setattr(Location, "__eq__", counted)
        ref = generation.ref(last)
        assert compared <= 1
        found = generation.lookup(ref)
    assert (ref, found) == (f"{generation.id}:claim:9999-10000", last)
    assert generation.lookup(f"{generation.id}:claim:9999-10001") is None


def test_republishing_returns_the_first_generation_and_reads_keep_kinds_apart(store):
    occurrence, chunks = extract(store)
    first = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    retry = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    assert retry == first
    assert retry.recorded_at == first.recorded_at
    assert published_locations(store, run()) == (first,)
    assert published_generations(store, run()) == (chunks,)


@pytest.mark.parametrize(
    "altered",
    [
        replace(MARKDOWN_LOCATOR, locate=lambda text: [("claim", 0, 5)]),
        PADDED_TEXT,
        replace(MARKDOWN_LOCATOR_V2, extract=PADDED_TEXT.extract),
    ],
    ids=["locate", "extract", "extract-with-new-locator"],
)
def test_changed_output_under_unchanged_steps_conflicts_and_keeps_the_original(store, altered):
    occurrence, _ = extract(store)
    original = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    with pytest.raises(IdentityConflict):
        publish_locations(store, occurrence.id, altered)
    assert published_locations(store, run()) == (original,)


def test_a_new_locator_version_publishes_beside_the_original_under_other_refs(store):
    occurrence, _ = extract(store)
    original = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    revised = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR_V2)
    assert original.id != revised.id
    assert revised.locations == original.locations
    assert set(published_locations(store, run())) == {original, revised}
    claim = original.locations[0]
    assert original.ref(claim) == f"{original.id}:claim:{claim.start}-{claim.end}"
    assert original.ref(claim) != revised.ref(claim)
    assert len({original.ref(item) for item in original.locations}) == len(original.locations)
    for foreign in (replace(claim, kind="figure"), Location("claim", 0, 1, "#")):
        with pytest.raises(SourceError):
            original.ref(foreign)


def test_locations_require_a_published_extraction(store):
    occurrence = register_source(store, PAPER, DECLARATION)
    stored = sorted(store.rglob("*"))
    with pytest.raises(SourceError) as raised:
        publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    assert not isinstance(raised.value, IdentityConflict)
    assert sorted(store.rglob("*")) == stored
    publish_generation(store, occurrence.id, TEXT_PIPELINE)
    generation = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    assert published_locations(store, run()) == (generation,)


@pytest.mark.parametrize(
    ("locate", "error"),
    [
        (lambda text: [("theorem", 0, 5)], SourceError),
        (lambda text: [("claim", 0, len(text) + 1)], SourceError),
        (lambda text: [("claim", 3, 3)], SourceError),
        (lambda text: [("claim", 0, 5), ("claim", 0, 5)], SourceError),
        (lambda text: [("claim", 6, 9), ("claim", 0, 5)], SourceError),
        (lambda text: [("claim", False, True)], SourceError),
        (crashing_locate, RuntimeError),
    ],
    ids=["kind", "beyond", "empty", "duplicate", "unsorted", "boolean", "crash"],
)
def test_invalid_locate_output_publishes_nothing(store, locate, error):
    occurrence, _ = extract(store)
    stored = sorted(store.rglob("*"))
    with pytest.raises(error):
        publish_locations(store, occurrence.id, replace(MARKDOWN_LOCATOR, locate=locate))
    assert sorted(store.rglob("*")) == stored
    assert published_locations(store, run()) == ()


class EqualToAnything(str):
    __hash__ = str.__hash__

    def __eq__(self, other):
        return True


@pytest.mark.parametrize(
    "locate",
    [
        lambda text: [text],
        lambda text: [("claim", 0, text)],
        lambda text: [(EqualToAnything("claim\ud800"), 0, len(text))],
        lambda text: [(EqualToAnything(text), 0, len(text))],
    ],
    ids=["text", "triple-with-text", "unencodable-kind", "text-kind"],
)
def test_invalid_locate_output_keeps_the_text_out_of_the_error(store, locate, reachable_text):
    occurrence, _ = extract(store)
    with pytest.raises(SourceError) as raised:
        publish_locations(store, occurrence.id, replace(MARKDOWN_LOCATOR, locate=locate))
    assert "top decile" not in reachable_text(raised.value)
    assert published_locations(store, run()) == ()


def test_unencodable_store_paths_raise_source_errors(tmp_path):
    with pytest.raises(SourceError):
        publish_locations(tmp_path / "missing\ud800", "0" * 64, MARKDOWN_LOCATOR)


def test_a_source_without_elements_publishes_an_empty_generation(store):
    note = (FIXTURES / "note.txt").read_bytes()
    plain = replace(DECLARATION, media_type="text/plain", location="note.txt")
    occurrence, _ = extract(store, note, plain)
    generation = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    assert (generation.locations, generation.length) == ((), len(note.decode("utf-8")))
    assert published_locations(store, run()) == (generation,)


def test_reads_show_only_generations_the_run_context_may_read(store):
    occurrence, _ = extract(store, PAPER, replace(DECLARATION, labels=frozenset({VENDOR})))
    generation = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    assert published_locations(store, run()) == ()
    assert published_locations(store, run(VENDOR)) == (generation,)
    early = run(VENDOR).narrow(cutoff=TIME - timedelta(microseconds=1))
    assert published_locations(store, early) == ()
    for stand_in in (TIME, SimpleNamespace(read_denials=lambda item: ())):
        with pytest.raises(SourceError):
            published_locations(store, stand_in)


def test_the_manifest_is_canonical_json_named_by_its_digest(store):
    data = b"We find $x$ is 5%.\n"
    occurrence, _ = extract(store, data, replace(DECLARATION, media_type="text/plain"))
    generation = publish_locations(store, occurrence.id, MARKDOWN_LOCATOR)
    directory = store / "locations" / generation.id
    content = (directory / "manifest.json").read_bytes()
    assert json.loads(content) == {
        "schema_version": 1,
        "occurrence_id": occurrence.id,
        "extraction": {
            "extractor": {"name": "utf-8-text", "version": "1"},
            "text_digest": hashlib.sha256(data).hexdigest(),
            "length": 19,
        },
        "locator": {"name": "markdown-elements", "version": "1"},
        "locations": [
            {"kind": "claim", "start": 0, "end": 18, "text": "We find $x$ is 5%."},
            {"kind": "equation", "start": 8, "end": 11, "text": "$x$"},
        ],
    }
    assert canonical(json.loads(content)) == content
    assert hashlib.sha256(content).hexdigest() == generation.digest
    assert (directory / "manifest.sha256").read_text() == f"{generation.digest}\n"
