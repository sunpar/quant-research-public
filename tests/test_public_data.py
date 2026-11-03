from __future__ import annotations

import hashlib
import io
import json
import socket
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_research.access import issue_context
from quant_research.catalog import (
    Approved,
    Catalog,
    CatalogError,
    Documented,
    Evidence,
    Revoked,
    Validated,
    Vintage,
)
from quant_research.datasets import ReadRecord, Refused, Supplied, read
from quant_research.locators import MARKDOWN_LOCATOR_V2, published_locations
from quant_research.public_data import (
    ESTRELLA_MISHKIN,
    PAPER_CASE,
    PILOT,
    SEARCH_UNIVERSE,
    Paper,
    PublicDataError,
    approve,
    ingest,
    load_catalog,
    pull,
    register_paper,
    source,
    validate,
)
from quant_research.sources import (
    TEXT_PIPELINE,
    SourceDeclaration,
    published_generations,
    register_source,
)

FIXTURES = Path(__file__).parent / "fixtures" / "public_data"
# Each ALFRED response is an excerpt of a real download, with count set to the excerpt's length.
RESPONSES = {
    "alfred.unrate": (FIXTURES / "alfred-unrate.json").read_bytes(),
    "alfred.sahmrealtime": (FIXTURES / "alfred-sahmrealtime.json").read_bytes(),
    "french.ff3_monthly": (FIXTURES / "french-ff3.zip").read_bytes(),
    "alfred.gs10": (FIXTURES / "alfred-gs10.json").read_bytes(),
    "alfred.tb3ms": (FIXTURES / "alfred-tb3ms.json").read_bytes(),
    "alfred.usrecq": (FIXTURES / "alfred-usrecq.json").read_bytes(),
}
KEY = "0123456789abcdef0123456789abcdef"
NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
LATER = NOW + timedelta(days=1)
ALFRED_GAPS = (
    "ALFRED real-time dates show when ALFRED published a value, not when a firm got it",
    "Each value is available at the end of its realtime_start date in UTC (D052)",
    "A missing observation has a null value",
)
PAPER = (FIXTURES.parent / "locators" / "paper.md").read_bytes()
PAPER_AVAILABLE = datetime(2026, 10, 21, tzinfo=UTC)
SYNTHETIC_PAPER = Paper(
    SourceDeclaration(
        media_type="text/markdown",
        access_domain="synthetic",
        location="tests/fixtures/locators/paper.md",
        revision="1",
        acquisition="hand-authored test fixture",
        rights="synthetic, unrestricted",
        labels=frozenset(),
        approved_for=frozenset(),
        published_at=None,
        source_available_at=PAPER_AVAILABLE,
        availability_evidence="authored for these tests",
        received_at=PAPER_AVAILABLE,
    ),
    hashlib.sha256(PAPER).hexdigest(),
)
PAPER_TABLE = "\n".join(
    [
        "| Decile | Return | t-stat |",
        "| :----- | -----: | -----: |",
        "| 1      | 0.4%   | 1.1    |",
        "| 10     | 1.6%   | 3.2    |",
    ]
)


def fetcher(responses=RESPONSES):
    def fetch(url):
        fetch.requested.append(url)
        base = url.removesuffix(f"&api_key={KEY}")
        return next(data for name, data in responses.items() if source(name).url == base)

    fetch.requested = []
    return fetch


def pull_all(root, now=NOW, responses=RESPONSES, names=PILOT):
    fetch = fetcher(responses)
    return [pull(root, name, now=now, api_key=KEY, fetch=fetch) for name in names]


def context(cutoff):
    return issue_context(clearances=(), cutoff=cutoff, generation="pilot/v1", providers=())


def read_root(root, dataset, cutoff=NOW):
    return read(load_catalog(root), root / "datasets", dataset, context(cutoff))


def supplied(result) -> Supplied:
    assert isinstance(result, Supplied), result
    return result


def approve_all(root, names=PILOT):
    pull_all(root, names=names)
    for name in names:
        ingest(root, name, now=NOW)
        validate(
            root,
            name,
            actor="researcher",
            evidence=b"Compared every row with the FRED and Ken French web pages",
            summary="Manual comparison with the publishers' pages",
            now=NOW,
        )
        assert approve(root, name, actor="researcher", now=NOW) == "approved"
    return root


@pytest.fixture
def approved_root(tmp_path):
    return approve_all(tmp_path)


def alfred(change) -> bytes:
    value = json.loads(RESPONSES["alfred.unrate"])
    change(value)
    return json.dumps(value).encode()


def french(change) -> bytes:
    with zipfile.ZipFile(io.BytesIO(RESPONSES["french.ff3_monthly"])) as archive:
        (member,) = archive.namelist()
        text = archive.read(member).decode("ascii")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in change(member, text):
            archive.writestr(name, content)
    return buffer.getvalue()


def test_recorded_responses_convert_to_adapter_rows():
    unrate = source("alfred.unrate").convert(RESPONSES["alfred.unrate"])
    versions = {(row["key"], row["revision"]): row for row in unrate}
    assert [revision for key, revision in versions if key == "1989-10-01"] == [1, 2, 3]
    assert versions[("1989-10-01", 3)] == {
        "arrived_at": "1994-01-08T00:00:00Z",
        "key": "1989-10-01",
        "published_at": "1994-01-08T00:00:00Z",
        "revision": 3,
        "values": {"value": "5.3"},
    }
    assert versions[("2025-10-01", 1)]["values"] == {"value": None}
    months = source("french.ff3_monthly").convert(RESPONSES["french.ff3_monthly"])
    assert months[0] == {
        "arrived_at": "1989-08-01T00:00:00Z",
        "key": "1989-07-01",
        "published_at": "1989-08-01T00:00:00Z",
        "revision": 1,
        "values": {"hml": "-2.80", "mkt_rf": "7.18", "rf": "0.70", "smb": "-4.11"},
    }
    assert [row["key"] for row in months] == [
        *(f"1989-{month:02d}-01" for month in range(7, 13)),
        *(f"1990-{month:02d}-01" for month in range(1, 7)),
    ]
    assert months[5]["published_at"] == "1990-01-01T00:00:00Z"


def gap(value):
    value["observations"][0]["realtime_end"] = "1990-01-31"


def closed(value):
    value["observations"][-1]["realtime_end"] = "2026-01-01"


def duplicate(value):
    value["observations"].append(value["observations"][0])
    value["count"] += 1


@pytest.mark.parametrize(
    ("dataset", "data"),
    [
        ("alfred.unrate", alfred(lambda value: value.update(count=value["count"] + 1))),
        ("alfred.unrate", alfred(lambda value: value.update(realtime_start="2026-10-22"))),
        ("alfred.unrate", alfred(lambda value: value.pop("observations"))),
        ("alfred.unrate", alfred(lambda value: value["observations"][0].update(value="five"))),
        ("alfred.unrate", alfred(lambda value: value["observations"][0].update(value=5.3))),
        ("alfred.unrate", alfred(lambda value: value["observations"][0].update(date="1989-13-01"))),
        ("alfred.unrate", alfred(lambda value: value["observations"][0].update(date="1989-10-1"))),
        ("alfred.unrate", alfred(gap)),
        ("alfred.unrate", alfred(closed)),
        ("alfred.unrate", alfred(duplicate)),
        ("alfred.unrate", b'{"error_code":400,"error_message":"Bad Request."}'),
        ("alfred.unrate", b"<html>Service unavailable</html>"),
        ("french.ff3_monthly", b"garbage"),
        ("french.ff3_monthly", french(lambda name, text: [(name, text), ("extra.csv", text)])),
        ("french.ff3_monthly", french(lambda name, text: [(name, text.replace(",Mkt-RF", ","))])),
        ("french.ff3_monthly", french(lambda name, text: [(name, text.replace(",   0.70", ""))])),
        ("french.ff3_monthly", french(lambda name, text: [(name, text.replace("198908,", "x"))])),
        (
            "french.ff3_monthly",
            french(lambda name, text: [(name, text.replace("198907", "198913"))]),
        ),
        ("french.ff3_monthly", french(lambda name, text: [(name, text.replace("7.18", "n/a"))])),
        (
            "french.ff3_monthly",
            french(lambda name, text: [(name, text.replace("RF\r\n198907", "RF\r\n\r\n198907"))]),
        ),
    ],
)
def test_a_malformed_download_is_refused_and_its_pull_stores_nothing(tmp_path, dataset, data):
    with pytest.raises(PublicDataError):
        source(dataset).convert(data)
    with pytest.raises(PublicDataError):
        pull(tmp_path, dataset, now=NOW, api_key=KEY, fetch=lambda url: data)
    assert list(tmp_path.iterdir()) == []


def test_pull_keeps_the_original_bytes_and_retrieval_time_and_never_the_key(tmp_path):
    fetch = fetcher()
    pulls = [pull(tmp_path, name, now=NOW, api_key=KEY, fetch=fetch) for name in PILOT]
    assert [KEY in url for url in fetch.requested] == [True, True, False]
    for record in pulls:
        assert (tmp_path / "objects" / record.download).read_bytes() == RESPONSES[record.dataset]
        assert record.retrieved_at == NOW
        assert KEY not in record.url
    saved = json.loads((tmp_path / "pulls" / "alfred.unrate" / "20261022T120000Z.json").read_text())
    assert saved == {
        "dataset": "alfred.unrate",
        "download": hashlib.sha256(RESPONSES["alfred.unrate"]).hexdigest(),
        "retrieved_at": "2026-10-22T12:00:00Z",
        "url": source("alfred.unrate").url,
    }
    stored = b"".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    assert KEY.encode() not in stored


def test_repeated_pull_refuses_a_corrupt_object_without_replacing_it(tmp_path):
    root = tmp_path / "new" / "store"
    first = pull(root, "alfred.unrate", now=NOW, api_key=KEY, fetch=fetcher())
    assert pull(root, "alfred.unrate", now=NOW, api_key=KEY, fetch=fetcher()) == first
    path = root / "objects" / first.download
    path.write_bytes(b"corrupt")
    with pytest.raises(PublicDataError, match="does not match its digest"):
        pull(root, "alfred.unrate", now=LATER, api_key=KEY, fetch=fetcher())
    assert path.read_bytes() == b"corrupt"
    assert len(list((root / "pulls" / "alfred.unrate").iterdir())) == 1


def test_an_alfred_pull_without_a_key_is_refused_before_any_request(tmp_path):
    fetch = fetcher()
    for key in (None, ""):
        with pytest.raises(PublicDataError, match="FRED_API_KEY"):
            pull(tmp_path, "alfred.unrate", now=NOW, api_key=key, fetch=fetch)
    assert fetch.requested == []
    assert (
        pull(tmp_path, "french.ff3_monthly", now=NOW, api_key=None, fetch=fetch).url
        == (fetch.requested[0])
    )


def test_a_failed_download_names_no_key(tmp_path):
    def refuse(url):
        raise urllib.error.HTTPError(url, 400, "Bad Request", None, None)

    with pytest.raises(PublicDataError) as raised:
        pull(tmp_path, "alfred.unrate", now=NOW, api_key=KEY, fetch=refuse)
    assert KEY not in str(raised.value)
    assert "400" in str(raised.value)


@pytest.mark.parametrize(
    "dataset", ["fred.unrate", "alfred.UNRATE", "alfred.", "alfred.un rate", "french.ff5", ""]
)
def test_a_name_without_a_public_source_is_refused(tmp_path, dataset):
    with pytest.raises(PublicDataError):
        source(dataset)
    with pytest.raises(PublicDataError):
        ingest(tmp_path, dataset, now=NOW)


def test_ingest_without_a_pull_is_refused(tmp_path):
    with pytest.raises(PublicDataError, match="data pull"):
        ingest(tmp_path, "alfred.unrate", now=NOW)


@pytest.mark.smoke
def test_ingest_leaves_each_dataset_documented_and_a_repeat_changes_no_digest(tmp_path):
    first_pulls = pull_all(tmp_path)
    first = [ingest(tmp_path, name, now=NOW) for name in PILOT]
    catalog = (tmp_path / "catalog.json").read_bytes()
    second_pulls = pull_all(tmp_path, now=LATER)
    assert [ingest(tmp_path, name, now=LATER) for name in PILOT] == first
    assert [item.download for item in second_pulls] == [item.download for item in first_pulls]
    assert (tmp_path / "catalog.json").read_bytes() == catalog
    assert [(item.rows, item.status) for item in first] == [
        (14, "documented"),
        (3, "documented"),
        (12, "documented"),
    ]
    for item, pulled in zip(first, first_pulls, strict=True):
        documentation = load_catalog(tmp_path).entry(item.dataset).documentation
        assert documentation.vintage == item.vintage
        assert {vintage.evidence.digest for vintage in documentation.vintages} == {pulled.download}
        assert documentation.access == documentation.approved_for == ()
        assert load_catalog(tmp_path).entry(item.dataset).gaps == documentation.known_gaps
        assert read_root(tmp_path, item.dataset) == Refused("documented")


@pytest.mark.parametrize(
    ("cutoff", "unrate", "month"),
    [
        (datetime(1990, 1, 1, tzinfo=UTC), [("1989-10-01", 1), ("1989-11-01", 1)], "1989-12-01"),
        (
            datetime(1990, 1, 5, 23, 59, 59, tzinfo=UTC),
            [("1989-10-01", 1), ("1989-11-01", 1)],
            "1989-12-01",
        ),
        (
            datetime(1990, 1, 6, tzinfo=UTC),
            [("1989-10-01", 1), ("1989-11-01", 2), ("1989-12-01", 1)],
            "1989-12-01",
        ),
        (
            datetime(1990, 2, 3, tzinfo=UTC),
            [("1989-10-01", 1), ("1989-11-01", 2), ("1989-12-01", 1), ("1990-01-01", 1)],
            "1990-01-01",
        ),
    ],
)
def test_an_approved_read_at_a_1990_cutoff_sees_only_what_was_released_by_then(
    approved_root, cutoff, unrate, month
):
    result = supplied(read_root(approved_root, "alfred.unrate", cutoff))
    assert [(row.key, row.revision) for row in result.rows] == unrate
    assert all(row.available_at <= cutoff for row in result.rows)
    assert supplied(read_root(approved_root, "alfred.sahmrealtime", cutoff)).rows == ()
    returns = supplied(read_root(approved_root, "french.ff3_monthly", cutoff)).rows
    assert returns[0].key == "1989-07-01"
    assert returns[-1].key == month


def test_validation_keeps_its_evidence_and_approval_names_the_adapter(approved_root):
    entry = load_catalog(approved_root).entry("alfred.unrate")
    assert [event.kind for event in entry.history] == ["documented", "validated", "approved"]
    evidence = entry.history[1].evidence
    assert (approved_root / "objects" / evidence.digest).read_bytes().startswith(b"Compared")
    assert entry.adapter == "quant_research.datasets/v1"
    with pytest.raises(CatalogError):
        approve(approved_root, "alfred.unrate", actor="researcher", now=NOW)


def newer_unrate(value):
    value["observations"][-1]["realtime_end"] = "2026-02-05"
    value["observations"].extend(
        [
            {"realtime_start": "2026-02-06", "realtime_end": "9999-12-31", "date": "2025-11-01"}
            | {"value": "4.4"},
            {"realtime_start": "2026-02-06", "realtime_end": "9999-12-31", "date": "2025-12-01"}
            | {"value": "4.4"},
        ]
    )
    value["count"] += 2


def test_an_old_read_record_replays_after_a_newer_vintage_is_stored(approved_root):
    old = supplied(read_root(approved_root, "alfred.unrate"))
    newer = {**RESPONSES, "alfred.unrate": alfred(newer_unrate)}
    pull(approved_root, "alfred.unrate", now=LATER, api_key=KEY, fetch=fetcher(newer))
    result = ingest(approved_root, "alfred.unrate", now=LATER)
    assert (result.rows, result.status) == (16, "documented")
    assert result.vintage != old.record.vintage
    history = load_catalog(approved_root).entry("alfred.unrate").history
    assert [event.kind for event in history][-3:] == ["approved", "revoked", "documented"]
    assert read_root(approved_root, "alfred.unrate") == Refused("documented")
    stored = approved_root / "datasets" / "alfred.unrate" / f"{old.record.vintage}.json"
    assert ReadRecord.from_bytes(old.record.to_bytes()).replay(stored.read_bytes()) == old.rows
    validate(
        approved_root,
        "alfred.unrate",
        actor="researcher",
        evidence=b"Rechecked",
        summary="Rechecked",
        now=LATER,
    )
    approve(approved_root, "alfred.unrate", actor="researcher", now=LATER)
    latest = supplied(read_root(approved_root, "alfred.unrate", datetime(2026, 3, 1, tzinfo=UTC)))
    assert [(row.key, row.revision) for row in latest.rows][-2:] == [
        ("2025-11-01", 3),
        ("2025-12-01", 1),
    ]


def test_changed_provider_approvals_revoke_approval_and_keep_the_vintage(approved_root):
    vintage = load_catalog(approved_root).entry("french.ff3_monthly").documentation.vintage
    providers = ["codex-subscription", "claude-subscription", "codex-subscription"]
    result = ingest(approved_root, "french.ff3_monthly", now=LATER, approved_for=providers)
    entry = load_catalog(approved_root).entry("french.ff3_monthly")
    assert (result.vintage, result.status) == (vintage, "documented")
    assert entry.documentation.approved_for == ("claude-subscription", "codex-subscription")
    assert isinstance(entry.history[-2], Revoked)
    catalog = (approved_root / "catalog.json").read_bytes()
    ingest(approved_root, "french.ff3_monthly", now=LATER, approved_for=providers)
    assert (approved_root / "catalog.json").read_bytes() == catalog


def test_unchanged_ingest_documents_a_revoked_dataset_again(approved_root):
    revoked = load_catalog(approved_root).append(
        Revoked("alfred.sahmrealtime", "researcher", LATER, "Rows looked wrong")
    )
    (approved_root / "catalog.json").write_bytes(revoked.to_bytes())
    assert ingest(approved_root, "alfred.sahmrealtime", now=LATER).status == "documented"


def test_the_search_universe_names_only_alfred_series_with_a_value_field():
    assert "alfred.unrate" in SEARCH_UNIVERSE
    for name in SEARCH_UNIVERSE:
        origin = source(name)
        assert origin.owner == "Federal Reserve Bank of St. Louis ALFRED"
        assert origin.fields == ("value",)
        assert f"series_id={name.removeprefix('alfred.').upper()}&" in origin.url


def test_a_new_universe_series_documents_its_gap_and_unrate_keeps_its_approval(approved_root):
    units = {
        "alfred.payems": "thousands of persons",
        "alfred.indpro": "base year changes across vintages",
        "alfred.awhman": "average weekly hours",
    }
    for name, unit in units.items():
        pull(
            approved_root, name, now=NOW, api_key=KEY, fetch=lambda url: RESPONSES["alfred.unrate"]
        )
        assert ingest(approved_root, name, now=NOW).status == "documented"
        gaps = load_catalog(approved_root).entry(name).documentation.known_gaps
        assert gaps[:3] == ALFRED_GAPS
        assert [unit in gap for gap in gaps[3:]] == [True]
    pull_all(approved_root, now=LATER)
    assert ingest(approved_root, "alfred.unrate", now=LATER).status == "approved"
    assert load_catalog(approved_root).entry("alfred.unrate").documentation.known_gaps == (
        ALFRED_GAPS
    )


def test_the_paper_case_series_convert_and_document_their_units_and_first_vintage(tmp_path):
    pull_all(tmp_path, names=PAPER_CASE)
    ingested = [ingest(tmp_path, name, now=NOW) for name in PAPER_CASE]
    assert [(item.rows, item.status) for item in ingested] == [
        (4, "documented"),
        (5, "documented"),
        (7, "documented"),
    ]
    phrases = {
        "alfred.gs10": ("10-year constant-maturity", "3 December 1996"),
        "alfred.tb3ms": ("discount basis", "3 December 1996"),
        "alfred.usrecq": (
            "NBER peak",
            "first day of each quarter",
            "NBER announces",
            "3 November 2011",
        ),
    }
    for name, expected in phrases.items():
        gaps = load_catalog(tmp_path).entry(name).documentation.known_gaps
        assert gaps[:3] == ALFRED_GAPS
        assert all(phrase in gap for gap, phrase in zip(gaps[3:], expected, strict=True)), (
            f"{name} documents its units and first vintage"
        )


@pytest.mark.parametrize(
    ("dataset", "first_available", "keys"),
    [
        ("alfred.gs10", datetime(1996, 12, 4, tzinfo=UTC), ["1973-10-01", "1996-11-01"]),
        ("alfred.tb3ms", datetime(1996, 12, 4, tzinfo=UTC), ["1973-10-01", "1996-11-01"]),
        ("alfred.usrecq", datetime(2011, 11, 4, tzinfo=UTC), ["1973-10-01", "1974-01-01"]),
    ],
)
def test_a_paper_case_read_before_the_first_vintage_supplies_no_row(
    tmp_path, dataset, first_available, keys
):
    approve_all(tmp_path, (dataset,))
    before = supplied(read_root(tmp_path, dataset, first_available - timedelta(seconds=1)))
    assert before.rows == ()
    after = supplied(read_root(tmp_path, dataset, first_available))
    assert [row.key for row in after.rows] == keys


def test_register_paper_publishes_chunks_and_locations_of_one_occurrence_once(tmp_path):
    generation, locations = register_paper(tmp_path, SYNTHETIC_PAPER, PAPER)
    assert generation.occurrence == locations.occurrence
    assert generation.occurrence.declaration == SYNTHETIC_PAPER.declaration
    assert generation.extraction_id == locations.extraction_id
    assert (generation.chunker, locations.locator) == (
        TEXT_PIPELINE.chunker,
        MARKDOWN_LOCATOR_V2.locator,
    )
    (table,) = [location for location in locations.locations if location.kind == "table"]
    assert locations.lookup(locations.ref(table)).text == PAPER_TABLE
    again = register_paper(tmp_path, SYNTHETIC_PAPER, PAPER)
    assert [(item.id, item.digest, item.recorded_at) for item in again] == [
        (item.id, item.digest, item.recorded_at) for item in (generation, locations)
    ]


def test_a_registered_paper_is_read_only_from_its_availability(tmp_path):
    generation, locations = register_paper(tmp_path, SYNTHETIC_PAPER, PAPER)
    store = tmp_path / "sources"
    earlier = context(PAPER_AVAILABLE - timedelta(seconds=1))
    assert published_generations(store, earlier) == published_locations(store, earlier) == ()
    assert published_generations(store, context(PAPER_AVAILABLE)) == (generation,)
    assert published_locations(store, context(PAPER_AVAILABLE)) == (locations,)


def test_estrella_mishkin_refuses_other_markdown_and_reads_from_june_1996(tmp_path):
    with pytest.raises(PublicDataError):
        register_paper(tmp_path, ESTRELLA_MISHKIN, PAPER)
    assert list(tmp_path.iterdir()) == []
    declaration = register_source(tmp_path, PAPER, ESTRELLA_MISHKIN.declaration).declaration
    assert declaration == ESTRELLA_MISHKIN.declaration
    end_of_may = datetime(1996, 6, 1, tzinfo=UTC)
    may = context(end_of_may - timedelta(seconds=1))
    assert may.read_denials(declaration.restrictions) == ("after_cutoff",)
    assert context(end_of_may).read_denials(declaration.restrictions) == ()


def root_state(root):
    return {
        str(path.relative_to(root)): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
            path.stat().st_ino,
        )
        for path in (root, *root.rglob("*"))
        if path.exists()
    }


@pytest.fixture
def status_root(tmp_path):
    original = Documented(
        "a.documented", "writer", NOW, "Former owner", ("old_field",), ("public",), "0" * 64
    )
    latest = replace(
        original,
        actor="catalog editor",
        at=LATER,
        owner="Research Ops",
        fields=("Signal Value", "release_month"),
        access=("internal", "public"),
        vintage="a" * 64,
        known_gaps=("Unresolved splice", "Gap-only-token"),
        vintages=(Vintage("revision", Evidence("2" * 64, "Evidence-only-token")),),
        approved_for=("claude-subscription", "codex-subscription"),
    )
    approved = replace(
        latest,
        dataset="b.approved",
        owner="Markets",
        fields=("return",),
        known_gaps=("Incomplete historical coverage",),
        vintages=(
            Vintage("publication", Evidence("1" * 64, "Publisher archive")),
            Vintage("revision", Evidence("2" * 64, "Revision archive")),
            Vintage("arrival", Evidence("3" * 64, "Receipt archive")),
        ),
        approved_for=("firm-proxy",),
    )
    validated = replace(
        original, dataset="c.validated", owner="Research Ops", fields=("Signal Value",)
    )
    revoked = replace(approved, dataset="d.revoked", approved_for=("retained-route",))
    attestation = Evidence("4" * 64, "Manual attestation")
    catalog = Catalog(
        (
            revoked,
            Validated("d.revoked", "reviewer", LATER, attestation),
            Approved("d.revoked", "lead", LATER, "old.adapter"),
            Revoked("d.revoked", "lead", LATER, "Withdrawn"),
            validated,
            Validated("c.validated", "reviewer", LATER, attestation),
            approved,
            Validated("b.approved", "reviewer", LATER, attestation),
            Approved("b.approved", "lead", LATER, "nonstandard.adapter/v7"),
            original,
            latest,
        )
    )
    (tmp_path / "catalog.json").write_bytes(catalog.to_bytes())
    return tmp_path


def test_cli_data_status_projects_recorded_state_and_latest_documentation(status_root, cli):
    before = root_state(status_root)
    result = cli(status_root, "data", "status", "--root", str(status_root))
    rows = json.loads(result.stdout)
    assert result.stderr == ""
    assert all(
        set(row) == {"dataset", "status", "adapter", "gaps", "documentation"} for row in rows
    )
    assert [(row["dataset"], row["status"], row["adapter"], row["gaps"]) for row in rows] == [
        (
            "a.documented",
            "documented",
            None,
            [
                "Unresolved splice",
                "Gap-only-token",
                "No publication vintage evidence",
                "No arrival vintage evidence",
            ],
        ),
        ("b.approved", "approved", "nonstandard.adapter/v7", ["Incomplete historical coverage"]),
        (
            "c.validated",
            "validated",
            None,
            [
                "No publication vintage evidence",
                "No revision vintage evidence",
                "No arrival vintage evidence",
            ],
        ),
        ("d.revoked", "revoked", None, ["Incomplete historical coverage"]),
    ]
    assert rows[0]["documentation"] == {
        "dataset": "a.documented",
        "actor": "catalog editor",
        "at": "2026-10-23T12:00:00Z",
        "owner": "Research Ops",
        "fields": ["Signal Value", "release_month"],
        "access": ["internal", "public"],
        "vintage": "a" * 64,
        "known_gaps": ["Unresolved splice", "Gap-only-token"],
        "vintages": [
            {
                "aspect": "revision",
                "evidence": {"digest": "2" * 64, "summary": "Evidence-only-token"},
            }
        ],
        "approved_for": ["claude-subscription", "codex-subscription"],
    }
    assert rows[1]["documentation"]["vintages"] == [
        {"aspect": "publication", "evidence": {"digest": "1" * 64, "summary": "Publisher archive"}},
        {"aspect": "revision", "evidence": {"digest": "2" * 64, "summary": "Revision archive"}},
        {"aspect": "arrival", "evidence": {"digest": "3" * 64, "summary": "Receipt archive"}},
    ]
    assert rows[3]["documentation"]["approved_for"] == ["retained-route"]
    assert cli(status_root, "data", "status", "--root", str(status_root)).stdout == result.stdout
    assert root_state(status_root) == before
    assert set(path.name for path in status_root.iterdir()) == {"catalog.json"}


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (("--query", "oPs SiGnAl"), ["a.documented", "c.validated"]),
        (("--query", "OPS", "--status", "validated"), ["c.validated"]),
        (("--status", "approved"), ["b.approved"]),
        (("--query", "no-match"), []),
        (("--query", "Gap-only-token"), []),
        (("--query", "Evidence-only-token"), []),
        (("--query", "old_field"), []),
    ],
)
def test_cli_data_status_wires_catalog_filters(status_root, cli, filters, expected):
    before = root_state(status_root)
    result = cli(status_root, "data", "status", "--root", str(status_root), *filters)
    assert [row["dataset"] for row in json.loads(result.stdout)] == expected
    assert root_state(status_root) == before


@pytest.mark.parametrize("kind", ["missing-root", "empty-root", "empty-catalog"])
def test_cli_data_status_returns_empty_without_creating_state(tmp_path, cli, kind):
    root = tmp_path / "data"
    if kind != "missing-root":
        root.mkdir()
    if kind == "empty-catalog":
        (root / "catalog.json").write_bytes(Catalog().to_bytes())
    before = root_state(root)
    result = cli(tmp_path, "data", "status", "--root", str(root))
    assert result.stdout == "[]\n"
    assert result.stderr == ""
    assert root_state(root) == before


@pytest.mark.parametrize("kind", ["malformed", "noncanonical", "directory"])
def test_cli_data_status_preserves_named_catalog_failures(tmp_path, cli, kind):
    path = tmp_path / "catalog.json"
    if kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"{" if kind == "malformed" else b'{ "schema_version": 1, "events": [] }')
    before = root_state(tmp_path)
    result = cli(
        tmp_path, "data", "status", "--root", str(tmp_path), "--query", "no-match", success=False
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("Data status failed: ")
    if kind == "noncanonical":
        assert "not canonical UTF-8 JSON" in result.stderr
    assert root_state(tmp_path) == before


@pytest.mark.parametrize("arguments", [("--status", "ready"), ()])
def test_cli_data_status_rejects_invalid_arguments(tmp_path, cli, arguments):
    root = ("--root", str(tmp_path)) if arguments else ()
    result = cli(tmp_path, "data", "status", *root, *arguments, success=False)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "invalid choice" in result.stderr if arguments else "--root" in result.stderr


def test_data_status_main_needs_no_key_or_network(status_root, monkeypatch, capsys):
    from quant_research.__main__ import main

    attempted = []

    def denied(*args, **kwargs):
        attempted.append((args, kwargs))
        raise AssertionError("Status attempted network access")

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    with pytest.raises(AssertionError, match="Status attempted network access"):
        urllib.request.urlopen("https://example.invalid")
    attempted.clear()
    before = root_state(status_root)
    monkeypatch.setattr(
        sys, "argv", ["quant-research", "data", "status", "--root", str(status_root)]
    )
    main()
    output = capsys.readouterr()
    assert output.err == ""
    assert [(row["dataset"], row["status"]) for row in json.loads(output.out)] == [
        ("a.documented", "documented"),
        ("b.approved", "approved"),
        ("c.validated", "validated"),
        ("d.revoked", "revoked"),
    ]
    assert attempted == []
    assert root_state(status_root) == before


def test_cli_ingests_validates_and_approves_and_refuses_a_pull_without_a_key(
    tmp_path, cli, monkeypatch
):
    pull_all(tmp_path)
    root = str(tmp_path)
    printed = json.loads(cli(tmp_path, "data", "ingest", "--root", root).stdout)
    catalog = load_catalog(tmp_path)
    assert [(item["dataset"], item["rows"], item["status"]) for item in printed] == [
        ("alfred.unrate", 14, "documented"),
        ("alfred.sahmrealtime", 3, "documented"),
        ("french.ff3_monthly", 12, "documented"),
    ]
    assert all(
        item["vintage"] == catalog.entry(item["dataset"]).documentation.vintage for item in printed
    )
    before = root_state(tmp_path)
    status = json.loads(cli(tmp_path, "data", "status", "--root", root).stdout)
    assert [(item["dataset"], item["status"], item["adapter"]) for item in status] == [
        ("alfred.sahmrealtime", "documented", None),
        ("alfred.unrate", "documented", None),
        ("french.ff3_monthly", "documented", None),
    ]
    assert root_state(tmp_path) == before
    notes = tmp_path / "notes.txt"
    notes.write_text("Compared ten rows of each dataset with the publisher's page\n")
    for name in PILOT:
        arguments = ("--root", root, "--actor", "researcher")
        cli(
            tmp_path,
            "data",
            "validate",
            name,
            *arguments,
            "--evidence",
            str(notes),
            "--summary",
            "Spot check",
        )
        before = root_state(tmp_path)
        status = json.loads(cli(tmp_path, "data", "status", "--root", root, "--query", name).stdout)
        assert [(item["dataset"], item["status"], item["adapter"]) for item in status] == [
            (name, "validated", None)
        ]
        assert root_state(tmp_path) == before
        approved = json.loads(cli(tmp_path, "data", "approve", name, *arguments).stdout)
        assert approved == [{"dataset": name, "status": "approved"}]
        before = root_state(tmp_path)
        status = json.loads(cli(tmp_path, "data", "status", "--root", root, "--query", name).stdout)
        assert [(item["dataset"], item["status"], item["adapter"]) for item in status] == [
            (name, "approved", "quant_research.datasets/v1")
        ]
        assert root_state(tmp_path) == before
        assert isinstance(read_root(tmp_path, name), Supplied)
    refused = cli(
        tmp_path, "data", "approve", "alfred.unrate", "--root", root, "--actor", "x", success=False
    )
    assert refused.stderr.startswith("Data approve failed")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    missing = cli(tmp_path, "data", "pull", "alfred.unrate", "--root", root, success=False)
    assert "FRED_API_KEY" in missing.stderr
