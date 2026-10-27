from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from uuid import uuid4

from quant_research import datasets
from quant_research.catalog import (
    Approved,
    Aspect,
    Catalog,
    CatalogError,
    Documented,
    Evidence,
    Revoked,
    Status,
    Validated,
    Vintage,
)
from quant_research.locators import MARKDOWN_LOCATOR_V2, LocatorGeneration, publish_locations
from quant_research.records import (
    ObjectConflict,
    PackageError,
    canonical_bytes,
    canonical_json,
    format_time,
    keep_object,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
    write_once,
)
from quant_research.sources import (
    TEXT_PIPELINE,
    Generation,
    SourceDeclaration,
    publish_generation,
    register_source,
)

PILOT = ("alfred.unrate", "alfred.sahmrealtime", "french.ff3_monthly")
SEARCH_UNIVERSE = ("alfred.unrate", "alfred.payems", "alfred.indpro", "alfred.awhman")
PAPER_CASE = ("alfred.gs10", "alfred.tb3ms", "alfred.usrecq")
ACTOR = "quant_research.public_data/v1"
FRED = "https://api.stlouisfed.org/fred/series/observations"
FRENCH = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
)
_ALL_PERIODS = ("1776-07-04", "9999-12-31")
_DECIMAL = re.compile(r"-?[0-9]+(\.[0-9]+)?")
_FRENCH_HEADER = ",Mkt-RF,SMB,HML,RF"
_FRENCH_FIELDS = ("mkt_rf", "smb", "hml", "rf")
_FRENCH_ROW = re.compile(r"([0-9]{4})(0[1-9]|1[0-2]),(.*)")
# A series absent here keeps the shared gaps, since changed documentation revokes its approval.
_SERIES_GAPS = {
    "payems": ("Values are thousands of persons",),
    "indpro": (
        "Values are an index whose base year changes across vintages (January 1990 reads "
        "140.9, then 107.2 from April 1990, and 61.6440 in 2023), so compare levels only within "
        "one vintage",
    ),
    "awhman": ("Values are average weekly hours",),
    "gs10": (
        "Values are percent per year, monthly averages of daily 10-year constant-maturity yields",
        "ALFRED vintages start on 3 December 1996, so an earlier cutoff sees no value",
    ),
    "tb3ms": (
        "Values are percent per year on a discount basis, monthly averages of daily 3-month "
        "secondary-market rates",
        "ALFRED vintages start on 3 December 1996, so an earlier cutoff sees no value",
    ),
    "usrecq": (
        "Values are 1 from the quarter after an NBER peak through the trough and 0 otherwise, "
        "FRED's interpretation of NBER's quarterly turning points",
        "Keys are the first day of each quarter",
        "Values change when NBER announces a peak or a trough, so a vintage before an "
        "announcement can differ from the final values",
        "ALFRED vintages start on 3 November 2011, so an earlier cutoff sees no value",
    ),
}

type Rows = list[dict]


class PublicDataError(ValueError):
    """Report an unknown dataset, a failed or malformed download, or a missing pull."""


@dataclass(frozen=True)
class Source:
    """Where one public dataset comes from and what its catalog documentation says.

    url never carries a key. keyed sources append the FRED API key to it at request time.
    """

    url: str
    keyed: bool
    owner: str
    fields: tuple[str, ...]
    known_gaps: tuple[str, ...]
    evidence: Mapping[Aspect, str]
    convert: Callable[[bytes], Rows]


@dataclass(frozen=True)
class Pull:
    """One download: its dataset, the SHA-256 of its original bytes, and when and where."""

    dataset: str
    download: str
    retrieved_at: datetime
    url: str


@dataclass(frozen=True)
class Ingested:
    dataset: str
    rows: int
    vintage: str
    status: Status


@dataclass(frozen=True)
class Paper:
    """A paper's source declaration and the SHA-256 of the converted Markdown it declares."""

    declaration: SourceDeclaration
    markdown: str


ESTRELLA_MISHKIN = Paper(
    SourceDeclaration(
        media_type="text/markdown",
        access_domain="public web",
        location=(
            "https://www.newyorkfed.org/medialibrary/media/research/staff_reports/"
            "research_papers/9609.pdf"
        ),
        revision=(
            "Arturo Estrella and Frederic S. Mishkin, Predicting U.S. Recessions: Financial "
            "Variables as Leading Indicators, Federal Reserve Bank of New York Research Paper "
            "9609, May 1996, the working paper of Review of Economics and Statistics "
            "80(1):45-61, February 1998; Markdown conversion 1"
        ),
        acquisition=(
            "Scanned PDF with SHA-256 "
            "529284d889247958197453552b01561f6f0deb738793362efe743e554fbcc02e downloaded at "
            "2026-10-23T02:17:42Z, converted to Markdown by OCR with corrections checked "
            "against the page images (D070)"
        ),
        rights=(
            "Copyright the authors; the Federal Reserve Bank of New York distributes it free, "
            "and its cover asks that it not be cited or quoted without the authors' permission"
        ),
        labels=frozenset(),
        approved_for=frozenset(),
        published_at=None,
        source_available_at=datetime(1996, 6, 1, tzinfo=UTC),
        availability_evidence=(
            "The cover dates the paper May 1996 and its title page April 1996, so it is "
            "available at the end of May 1996 (D070)"
        ),
        received_at=datetime(2026, 10, 23, 2, 17, 42, tzinfo=UTC),
    ),
    "317a49b5bf0ac60d1fb3663ff090efa0c33307c639d2a07900b1ae2b3d3943a5",
)


def source(dataset: str) -> Source:
    """Return the source of a public dataset.

    `alfred.<series>` names every real-time period of a FRED series, and `french.ff3_monthly`
    names the monthly Fama/French 3 factors.

    Args:
        dataset (str): Dataset name.

    Returns:
        (Source) Where the dataset comes from and how it converts.

    Raises:
        PublicDataError: If no source has that name.
    """
    if dataset == "french.ff3_monthly":
        return Source(
            FRENCH,
            False,
            "Kenneth R. French Data Library",
            _FRENCH_FIELDS,
            (
                "Values are percent, as the file publishes them",
                "The file has no per-row dates, so each month is available at its end (D052)",
                "The library rebuilds its whole history each month and only this cut is kept",
                "RF comes from Ibbotson through May 2024 and from ICE BofA from June 2024",
            ),
            {
                "publication": "Ken French download; each month is published at its end (D052)",
                "revision": "Ken French download; one cut, so every month is revision 1",
                "arrival": "Ken French download; each month arrives at its end (D052)",
            },
            _french_rows,
        )
    prefix, _, series = dataset.partition(".")
    if prefix != "alfred" or not re.fullmatch(r"[a-z0-9_]+", series):
        raise PublicDataError(f"No public source is named {dataset!r}")
    start, end = _ALL_PERIODS
    return Source(
        f"{FRED}?series_id={series.upper()}&realtime_start={start}&realtime_end={end}"
        "&file_type=json",
        True,
        "Federal Reserve Bank of St. Louis ALFRED",
        ("value",),
        (
            "ALFRED real-time dates show when ALFRED published a value, not when a firm got it",
            "Each value is available at the end of its realtime_start date in UTC (D052)",
            "A missing observation has a null value",
            *_SERIES_GAPS.get(series, ()),
        ),
        {
            "publication": "ALFRED download; realtime_start of each value",
            "revision": "ALFRED download; real-time periods of each observation",
            "arrival": "ALFRED download; arrival is set from realtime_start (D052)",
        },
        _alfred_rows,
    )


def pull(
    root: Path,
    dataset: str,
    *,
    now: datetime,
    api_key: str | None,
    fetch: Callable[[str], bytes] | None = None,
) -> Pull:
    """Download a dataset, check that it converts, and keep its original bytes.

    The bytes are stored under their SHA-256 and the pull record under the retrieval time, so a
    repeated pull of unchanged data adds a record and changes no digest. The key is sent and
    never stored.

    Args:
        root (Path): Public data root.
        dataset (str): Dataset name.
        now (datetime): Retrieval time.
        api_key (str | None): FRED API key, needed for `alfred.` datasets.
        fetch (Callable[[str], bytes] | None, optional): Function that returns a URL's bytes.
            None downloads over HTTPS.

    Returns:
        (Pull) The stored pull record.

    Raises:
        PublicDataError: If the source needs a missing key, the download fails, or it does not
            convert.
    """
    origin = source(dataset)
    if origin.keyed and not api_key:
        raise PublicDataError(f"Set FRED_API_KEY to pull {dataset!r}")
    try:
        data = (fetch or _download)(
            f"{origin.url}&api_key={api_key}" if origin.keyed else origin.url
        )
    except OSError as error:
        raise PublicDataError(f"Cannot download {dataset!r}: {error}") from error
    origin.convert(data)
    record = Pull(dataset, _keep(root, data), now, origin.url)
    directory = root / "pulls" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    content = canonical_bytes(
        {
            "dataset": dataset,
            "download": record.download,
            "retrieved_at": format_time(now),
            "url": origin.url,
        }
    )
    if not write_once(directory / f"{now.astimezone(UTC):%Y%m%dT%H%M%SZ}.json", content):
        raise PublicDataError(f"Another pull of {dataset!r} has the same retrieval time")
    return record


def ingest(
    root: Path, dataset: str, *, now: datetime, approved_for: Iterable[str] = ()
) -> Ingested:
    """Convert a dataset's latest pull into a stored vintage and document it in the catalog.

    Unchanged documentation keeps the dataset's status, so a repeated ingest changes no digest.
    Changed documentation revokes any validation or approval in the same catalog revision and
    leaves the dataset documented.

    Args:
        root (Path): Public data root.
        dataset (str): Dataset name.
        now (datetime): Catalog event time.
        approved_for (Iterable[str], optional): Providers that may receive the dataset.

    Returns:
        (Ingested) The dataset, its row count, its vintage, and its catalog status.

    Raises:
        PublicDataError: If the dataset has no pull or its download does not convert.
        CatalogError: If the catalog refuses the documentation.
    """
    origin = source(dataset)
    latest = _latest_pull(root, dataset)
    rows = origin.convert(_object(root, latest.download))
    vintage = datasets.save(
        root / "datasets",
        dataset,
        canonical_bytes({"dataset": dataset, "rows": rows, "schema_version": 1}),
    )
    documentation = Documented(
        dataset,
        ACTOR,
        now,
        origin.owner,
        origin.fields,
        (),
        vintage,
        origin.known_gaps,
        tuple(
            Vintage(aspect, Evidence(latest.download, summary))
            for aspect, summary in origin.evidence.items()
        ),
        tuple(sorted(set(approved_for))),
    )
    catalog = load_catalog(root)
    try:
        entry = catalog.entry(dataset)
    except CatalogError:
        entry = None
    unchanged = entry is not None and entry.documentation == replace(
        documentation, actor=entry.documentation.actor, at=entry.documentation.at
    )
    if not unchanged or entry.status == "revoked":
        if entry is not None and entry.status in ("validated", "approved"):
            reason = f"Ingest replaced the documentation in force; vintage {vintage}"
            catalog = catalog.append(Revoked(dataset, ACTOR, now, reason))
        catalog = catalog.append(documentation)
        _save_catalog(root, catalog)
    return Ingested(dataset, len(rows), vintage, catalog.entry(dataset).status)


def validate(
    root: Path, dataset: str, *, actor: str, evidence: bytes, summary: str, now: datetime
) -> Status:
    """Record a manual validation attestation and keep its evidence bytes.

    Args:
        root (Path): Public data root.
        dataset (str): Documented dataset name.
        actor (str): Person attesting the validation.
        evidence (bytes): Validation notes or report.
        summary (str): What the validation checked.
        now (datetime): Catalog event time.

    Returns:
        (Status) The dataset's new status.

    Raises:
        CatalogError: If the catalog refuses the attestation.
    """
    catalog = load_catalog(root).append(
        Validated(dataset, actor, now, Evidence(_keep(root, evidence), summary))
    )
    _save_catalog(root, catalog)
    return catalog.entry(dataset).status


def approve(root: Path, dataset: str, *, actor: str, now: datetime) -> Status:
    """Approve a validated dataset for agent reads through the dataset adapter.

    Args:
        root (Path): Public data root.
        dataset (str): Validated dataset name.
        actor (str): Person approving agent use.
        now (datetime): Catalog event time.

    Returns:
        (Status) The dataset's new status.

    Raises:
        CatalogError: If the catalog refuses the approval.
    """
    catalog = load_catalog(root).append(Approved(dataset, actor, now, datasets.ADAPTER))
    _save_catalog(root, catalog)
    return catalog.entry(dataset).status


def register_paper(
    root: Path, paper: Paper, markdown: bytes
) -> tuple[Generation, LocatorGeneration]:
    """Register a paper's converted Markdown and publish its chunks and located elements.

    The paper goes in the source store under `root / "sources"`. A repeated call returns the
    generations as first published.

    Args:
        root (Path): Public data root.
        paper (Paper): Declaration and Markdown digest of the paper.
        markdown (bytes): Converted Markdown.

    Returns:
        (tuple[Generation, LocatorGeneration]) The paragraph generation and the version 2
            Markdown locator generation of one occurrence.

    Raises:
        PublicDataError: If the Markdown's SHA-256 differs from the paper's, since an occurrence
            id leaves the bytes out.
        SourceError: If the store refuses the source or a publication.
    """
    if hashlib.sha256(markdown).hexdigest() != paper.markdown:
        raise PublicDataError(f"The Markdown does not have the SHA-256 {paper.markdown}")
    store = root / "sources"
    store.mkdir(parents=True, exist_ok=True)
    occurrence = register_source(store, markdown, paper.declaration)
    return (
        publish_generation(store, occurrence.id, TEXT_PIPELINE),
        publish_locations(store, occurrence.id, MARKDOWN_LOCATOR_V2),
    )


def load_catalog(root: Path) -> Catalog:
    """Return the root's catalog, or an empty catalog before the first ingest."""
    path = root / "catalog.json"
    return Catalog.from_bytes(path.read_bytes()) if path.exists() else Catalog()


def _save_catalog(root: Path, catalog: Catalog) -> None:
    staged = root / f".catalog.{uuid4()}"
    try:
        staged.write_bytes(catalog.to_bytes())
        staged.replace(root / "catalog.json")
    finally:
        staged.unlink(missing_ok=True)


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "quant-research"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _keep(root: Path, data: bytes) -> str:
    directory = root / "objects"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        return keep_object(directory, data)
    except ObjectConflict as error:
        raise PublicDataError(f"Stored object {error} does not match its digest") from error


def _object(root: Path, digest: str) -> bytes:
    data = (root / "objects" / digest).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise PublicDataError(f"Stored object {digest} does not match its digest")
    return data


def _latest_pull(root: Path, dataset: str) -> Pull:
    paths = sorted((root / "pulls" / dataset).glob("*.json"))
    if not paths:
        raise PublicDataError(f"No pull of {dataset!r}; run data pull first")
    try:
        value = parse_fields(
            canonical_json(paths[-1].read_bytes(), "pull record"),
            "dataset download retrieved_at url",
            "pull record",
        )
        return Pull(
            parse_text(value["dataset"], "dataset"),
            parse_digest(value["download"]),
            parse_time(value["retrieved_at"]),
            parse_text(value["url"], "url"),
        )
    except PackageError as error:
        raise PublicDataError(f"Invalid pull record {paths[-1].name}: {error}") from error


def _available(day: date) -> str:
    return format_time(datetime.combine(day + timedelta(days=1), time(), UTC))


def _decimal(text: str) -> str:
    if not _DECIMAL.fullmatch(text):
        raise PublicDataError("A value is not a decimal number")
    return text


def _day(text: object) -> date:
    if not isinstance(text, str) or date.fromisoformat(text).isoformat() != text:
        raise PublicDataError("A date is not ISO YYYY-MM-DD")
    return date.fromisoformat(text)


def _alfred_rows(data: bytes) -> Rows:
    try:
        response = json.loads(data.decode("utf-8"))
        if (response["realtime_start"], response["realtime_end"]) != _ALL_PERIODS:
            raise PublicDataError("The response does not cover every real-time period")
        observations = response["observations"]
        if response["count"] != len(observations):
            raise PublicDataError("The response holds only part of the series")
        periods = defaultdict(list)
        for item in observations:
            periods[_day(item["date"])].append(
                (_day(item["realtime_start"]), _day(item["realtime_end"]), item["value"])
            )
    except (ValueError, KeyError, TypeError) as error:
        raise PublicDataError(f"Malformed ALFRED response: {error}") from error
    rows = []
    for observed, versions in sorted(periods.items()):
        versions.sort()
        ends = [end for _, end, _ in versions]
        starts = [start + timedelta(days=-1) for start, _, _ in versions[1:]]
        if ends[:-1] != starts or ends[-1] != date.max or versions[0][0] > versions[0][1]:
            raise PublicDataError(f"Real-time periods of {observed} are not contiguous")
        for revision, (start, _, value) in enumerate(versions, 1):
            if not isinstance(value, str):
                raise PublicDataError("A value is not text")
            available = _available(start)
            rows.append(
                {
                    "arrived_at": available,
                    "key": observed.isoformat(),
                    "published_at": available,
                    "revision": revision,
                    "values": {"value": None if value == "." else _decimal(value)},
                }
            )
    return rows


def _french_rows(data: bytes) -> Rows:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            (member,) = archive.namelist()
            lines = archive.read(member).decode("ascii").splitlines()
        start = lines.index(_FRENCH_HEADER) + 1
    except (ValueError, zipfile.BadZipFile) as error:
        raise PublicDataError(f"Malformed Ken French file: {error}") from error
    rows, month = [], None
    for line in lines[start:]:
        if not line.strip():
            break
        match = _FRENCH_ROW.fullmatch(line)
        values = [value.strip() for value in match.group(3).split(",")] if match else []
        if len(values) != 4:
            raise PublicDataError("A monthly row is not a month and four values")
        following = date(int(match.group(1)), int(match.group(2)), 1)
        if month is not None and following != _next_month(month):
            raise PublicDataError("Monthly rows are not consecutive months")
        month = following
        available = format_time(datetime.combine(_next_month(month), time(), UTC))
        rows.append(
            {
                "arrived_at": available,
                "key": month.isoformat(),
                "published_at": available,
                "revision": 1,
                "values": dict(zip(_FRENCH_FIELDS, map(_decimal, values), strict=True)),
            }
        )
    if not rows:
        raise PublicDataError("The Ken French file has no monthly rows")
    return rows


def _next_month(month: date) -> date:
    return date(month.year + month.month // 12, month.month % 12 + 1, 1)
