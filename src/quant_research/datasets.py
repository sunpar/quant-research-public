from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

from quant_research.access import (
    PERIOD_MAPPING,
    PeriodMapping,
    Provider,
    RestrictionLabel,
    Restrictions,
    RunContext,
    derive,
)
from quant_research.catalog import Catalog, CatalogError, Documented
from quant_research.records import (
    PackageError,
    canonical_bytes,
    canonical_json,
    format_time,
    parse_array,
    parse_digest,
    parse_fields,
    parse_name,
    parse_text,
    parse_time,
    write_once,
)
from quant_research.retrieval import Unit, corpus_members
from quant_research.sources import (
    TEXT_PIPELINE,
    Generation,
    SourceDeclaration,
    publish_generation,
    register_source,
)

ADAPTER = "quant_research.datasets/v1"

type Reason = Literal["unknown_dataset", "documented", "validated", "revoked", "other_adapter"]

_MESSAGES: dict[Reason, str] = {
    "unknown_dataset": "No dataset this run may read has this name",
    "documented": "The dataset is documented and not approved for agent use",
    "validated": "The dataset is validated and not approved for agent use",
    "revoked": "The dataset is revoked and not approved for agent use",
    "other_adapter": "The dataset is approved through a different adapter",
}


class DatasetError(ValueError):
    """Report a bad context, a malformed fixture or record, or a replay that cannot match."""


class Values(dict[str, object]):
    """A row's values, which refuse every change, because every read of a vintage shares them.

    A dict subclass rather than a read-only mapping, so dataclasses.asdict and json encode it as
    they encode a dict.
    """

    def _refuse(self, *args: object, **kwargs: object) -> None:
        raise TypeError("A row's values cannot change")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _refuse


@dataclass(frozen=True)
class Row:
    """One version of one row, available once both its release and its arrival have passed."""

    key: str
    revision: int
    published_at: datetime
    arrived_at: datetime
    values: Values

    @property
    def available_at(self) -> datetime:
        return max(self.published_at, self.arrived_at)


@dataclass(frozen=True)
class ReadRecord:
    """Commit one supplied read to what produced it, so the read can be replayed from the bytes.

    vintage is the SHA-256 of the dataset file read and catalog the digest of the catalog revision
    that approved the read. context is `RunContext.identity`. supplied lists the
    (key, revision) identity of each supplied row, in supplied order.
    """

    dataset: str
    vintage: str
    catalog: str
    cutoff: datetime
    context: str
    supplied: tuple[tuple[str, int], ...]

    def to_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON with the schema version and every field."""
        return canonical_bytes(
            {
                "catalog": self.catalog,
                "context": self.context,
                "cutoff": format_time(self.cutoff),
                "dataset": self.dataset,
                "schema_version": 1,
                "supplied": [{"key": key, "revision": revision} for key, revision in self.supplied],
                "vintage": self.vintage,
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ReadRecord:
        """Parse canonical record bytes.

        Args:
            data (bytes): Canonical UTF-8 JSON, as `to_bytes` returns it.

        Returns:
            (ReadRecord) The record those bytes encode.

        Raises:
            DatasetError: If the bytes are not canonical or encode a malformed record.
        """
        try:
            value = _load(
                data, "catalog context cutoff dataset schema_version supplied vintage", "record"
            )
            return cls(
                parse_name(value["dataset"]),
                parse_digest(value["vintage"]),
                parse_digest(value["catalog"]),
                parse_time(value["cutoff"]),
                parse_digest(value["context"]),
                tuple(
                    _identity(parse_fields(raw, "key revision", "identity"))
                    for raw in parse_array(value["supplied"], "supplied")
                ),
            )
        except (ValueError, RecursionError) as error:
            raise DatasetError(f"Invalid read record: {error}") from error

    def replay(self, data: bytes) -> tuple[Row, ...]:
        """Return the recorded rows, in recorded order, from the fixture bytes that were read.

        Needs no catalog or context, because the caller already holds the bytes.

        Args:
            data (bytes): Dataset file bytes whose SHA-256 is the recorded vintage.

        Returns:
            (tuple) The rows the read supplied.

        Raises:
            DatasetError: If the bytes are not the recorded vintage, are malformed, or lack a
                recorded identity.
        """
        if hashlib.sha256(data).hexdigest() != self.vintage:
            raise DatasetError(f"Bytes are not the recorded vintage of {self.dataset!r}")
        rows = {(row.key, row.revision): row for row in _parse(data, self.dataset)}
        missing = [identity for identity in self.supplied if identity not in rows]
        if missing:
            raise DatasetError(f"Recorded identity {missing[0]!r} is not in {self.dataset!r}")
        return tuple(rows[identity] for identity in self.supplied)

    def validate(self, data: bytes, *, catalog: Catalog, context: RunContext) -> None:
        """Require the complete read that the pinned catalog and context authorize.

        Raise DatasetError if admission, vintage bytes, or the exact ordered record differs.
        """
        admission = _admit(catalog, self.dataset, context)
        if isinstance(admission, Refused):
            raise DatasetError(admission.message)
        vintage = admission.documentation.vintage
        if hashlib.sha256(data).hexdigest() != vintage:
            raise DatasetError(f"Bytes are not the documented vintage of {self.dataset!r}")
        rows = _select(_parse(data, self.dataset), self.dataset, context)
        expected = ReadRecord(
            self.dataset,
            vintage,
            catalog.digest,
            context.cutoff,
            context.identity,
            tuple((row.key, row.revision) for row in rows),
        )
        if self != expected:
            raise DatasetError("Read record differs from its pinned authority")


@dataclass(frozen=True)
class Supplied:
    """The rows a run may read, ordered by key, with the record that replays them.

    restrictions holds the documentation's access labels and provider approvals, available at
    the run's cutoff. The record leaves them out, because its catalog digest pins that
    documentation.
    """

    rows: tuple[Row, ...]
    record: ReadRecord
    restrictions: Restrictions


@dataclass(frozen=True)
class Refused:
    """A read the catalog or the run context does not allow. It names no label and no time."""

    reason: Reason

    @property
    def message(self) -> str:
        return _MESSAGES[self.reason]


@dataclass(frozen=True)
class _Admission:
    documentation: Documented
    restrictions: Restrictions


def _admit(catalog: Catalog, dataset: str, context: RunContext) -> _Admission | Refused:
    if not isinstance(context, RunContext):
        raise DatasetError(f"context must be a RunContext, not {type(context).__name__}")
    try:
        entry = catalog.entry(parse_name(dataset))
    except (CatalogError, PackageError):
        return Refused("unknown_dataset")
    documentation = entry.documentation
    restrictions = _restrictions(documentation, context.cutoff)
    if not context.permits(dataset) or context.read_denials(restrictions):
        return Refused("unknown_dataset")
    if entry.status != "approved":
        return Refused(entry.status)
    if entry.adapter != ADAPTER:
        return Refused("other_adapter")
    return _Admission(documentation, restrictions)


def _restrictions(documentation: Documented, available_at: datetime) -> Restrictions:
    return Restrictions(
        available_at=available_at,
        labels=frozenset(RestrictionLabel(label) for label in documentation.access),
        approved_for=frozenset(Provider(name) for name in documentation.approved_for),
    )


def read(catalog: Catalog, store: Path, dataset: str, context: RunContext) -> Supplied | Refused:
    """Return, for each key, the latest revision of a dataset that the run may read.

    Approval is the catalog's current state, and the read opens the vintage that the
    documentation in force names. Catalog event times are wall-clock governance and are never
    compared with `context.cutoff`, which is a simulated information horizon. Every gate runs
    before the dataset file is opened. A name that is not a logical name cannot name a dataset
    file, so it is refused as unknown. A dataset outside the run's scope, or one whose labels the
    run is not cleared for, gets the same refusal before its status is checked, so a refusal never
    shows that it exists. A key whose every revision comes after the cutoff is omitted without a
    reason.

    Mapped contexts require canonical `YYYY-MM-01` keys and use midnight UTC month labels.
    The read selects the inclusive union of the dataset's permitted intervals, then the latest
    revision available at the cutoff. Malformed mapped keys raise DatasetError before interval
    or cutoff exclusion, including keys on rows that would otherwise be omitted.

    Args:
        catalog (Catalog): Catalog whose latest event for the dataset must approve this adapter.
        store (Path): Directory holding `<dataset>/<vintage>.json`, as `save` writes it.
        dataset (str): Dataset name.
        context (RunContext): Trusted run authority from issue_context or narrow.

    Returns:
        (Supplied | Refused) The rows, the record that replays them, and the dataset's
            restrictions, or the refusal.

    Raises:
        DatasetError: If context is not a RunContext, or the documented vintage is unreadable,
            malformed, or not the bytes stored under its name.
    """
    admission = _admit(catalog, dataset, context)
    if isinstance(admission, Refused):
        return admission
    vintage = admission.documentation.vintage
    path = store / dataset / f"{vintage}.json"
    _verified(path, dataset, vintage)
    rows = _select(_stored(path, dataset, vintage), dataset, context)
    record = ReadRecord(
        dataset,
        vintage,
        catalog.digest,
        context.cutoff,
        context.identity,
        tuple((row.key, row.revision) for row in rows),
    )
    return Supplied(rows, record, admission.restrictions)


def _select(rows: tuple[Row, ...], dataset: str, context: RunContext) -> tuple[Row, ...]:
    latest: dict[str, Row] = {}
    periods = next(
        (item for item in context.dataset_periods or () if item.dataset == dataset), None
    )
    for row in rows:
        if periods is not None:
            try:
                month = date.fromisoformat(row.key)
                if month.day != 1 or month.isoformat() != row.key:
                    raise ValueError("Noncanonical month key")
            except ValueError:
                raise DatasetError(f"Invalid dataset {dataset!r}") from None
            label = datetime.combine(month, time(), UTC)
            if not any(interval.start <= label <= interval.end for interval in periods.intervals):
                continue
        if row.available_at > context.cutoff:
            continue
        if row.key not in latest or row.revision > latest[row.key].revision:
            latest[row.key] = row
    return tuple(sorted(latest.values(), key=lambda row: row.key))


@dataclass(frozen=True)
class Listed:
    """One dataset a run may read, as its documentation describes it.

    restrictions are those a read of the dataset carries.
    """

    dataset: str
    owner: str
    fields: tuple[str, ...]
    restrictions: Restrictions


@dataclass(frozen=True)
class Listing:
    """The datasets a run may read, in name order, under the catalog revision named by catalog."""

    catalog: str
    datasets: tuple[Listed, ...]


def listing(catalog: Catalog, context: RunContext) -> Listing:
    """Return every dataset that read would supply rather than refuse, without opening a file.

    A dataset outside the run's scope or clearances never appears, as read refuses it as a name
    that does not exist.

    Args:
        catalog (Catalog): Catalog whose entries to list.
        context (RunContext): Trusted run authority from issue_context or narrow.

    Returns:
        (Listing) Each readable dataset's name, owner, fields, and read restrictions.

    Raises:
        DatasetError: If context is not a RunContext.
    """
    listed = []
    for entry in catalog.search():
        admission = _admit(catalog, entry.name, context)
        if isinstance(admission, _Admission):
            documentation = admission.documentation
            listed.append(
                Listed(
                    entry.name, documentation.owner, documentation.fields, admission.restrictions
                )
            )
    return Listing(catalog.digest, tuple(listed))


@dataclass(frozen=True)
class Description:
    """A readable dataset's contract: the units of its contract parts that the run may read.

    restrictions derive over the read's restrictions and the declaration of each part returned.
    """

    catalog: str
    dataset: str
    units: tuple[Unit, ...]
    restrictions: Restrictions


def describe(
    catalog: Catalog, sources: Path, dataset: str, context: RunContext
) -> Description | Refused:
    """Return the contract parts of a dataset that the context's corpus holds and the run may read.

    A part available after the cutoff, or beyond the run's clearances, is not a corpus member
    the run may read, so it is absent without a reason. The dataset store is never opened.

    Args:
        catalog (Catalog): Catalog that must approve the dataset as read does.
        sources (Path): Source store that holds the context's corpus.
        dataset (str): Dataset name.
        context (RunContext): Trusted run authority whose generation names a published corpus.

    Returns:
        (Description | Refused) The contract units in order of availability, then of part, or
            the refusal read gives.

    Raises:
        DatasetError: If context is not a RunContext, or the revision of a contract part in the
            corpus does not name its part.
        RetrievalError: If the corpus is unpublished, damaged, or differs from its pins.
        SourceError: If a published generation fails verification.
    """
    admission = _admit(catalog, dataset, context)
    if isinstance(admission, Refused):
        return admission
    parts = sorted(
        (
            generation
            for generation in corpus_members(sources, context)
            if generation.occurrence.declaration.location == _location(dataset)
        ),
        key=lambda generation: (
            generation.occurrence.declaration.source_available_at,
            _part_index(generation),
        ),
    )
    return Description(
        catalog.digest,
        dataset,
        tuple(
            Unit(generation.occurrence.id, generation.id, chunk.start, chunk.end, chunk.text)
            for generation in parts
            for chunk in generation.chunks
        ),
        derive(
            admission.restrictions,
            *(generation.occurrence.declaration.restrictions for generation in parts),
        ),
    )


def publish_contract(sources: Path, catalog: Catalog, dataset: str) -> tuple[Generation, ...]:
    """Register and publish each part of a dataset's contract as its own source occurrence.

    The contract is quant_research/contracts/<dataset>.json. Each part takes the access labels
    and provider approvals of the dataset's documentation in force, so it reaches the runs and
    routes the dataset does, and its own availability, so a run sees only the conclusions known
    by its cutoff. Repeating the call writes nothing new and returns equal generations.

    Args:
        sources (Path): Existing source store directory.
        catalog (Catalog): Catalog that documents the dataset.
        dataset (str): Dataset name.

    Returns:
        (tuple) One text generation per part, in part order, or () when the dataset has no
            contract.

    Raises:
        DatasetError: If no catalog entry has the name, or the contract file is malformed.
        SourceError: If a part cannot be registered or published.
    """
    try:
        documentation = catalog.entry(parse_name(dataset)).documentation
    except (CatalogError, PackageError) as error:
        raise DatasetError(f"No catalog entry names dataset {dataset!r}") from error
    path = resources.files("quant_research").joinpath("contracts", f"{dataset}.json")
    if not path.is_file():
        return ()
    generations = []
    parts, mapping = _contract(path.read_bytes(), dataset)
    for index, part in enumerate(parts):
        data = part.text.encode()
        restrictions = _restrictions(documentation, part.available_at)
        period = f"period mapping {mapping}, " if index == 0 and mapping is not None else ""
        declaration = SourceDeclaration(
            media_type="text/markdown",
            access_domain="repository",
            location=_location(dataset),
            revision=f"Part {index}, {period}text SHA-256 {hashlib.sha256(data).hexdigest()}",
            acquisition="Written in quant_research/contracts",
            rights="Project documentation",
            labels=restrictions.labels,
            approved_for=restrictions.approved_for,
            published_at=None,
            source_available_at=part.available_at,
            availability_evidence=part.availability_evidence,
            received_at=part.received_at,
        )
        occurrence = register_source(sources, data, declaration)
        generations.append(publish_generation(sources, occurrence.id, TEXT_PIPELINE))
    return tuple(generations)


def contract_of(generation: Generation) -> str | None:
    """Return the dataset whose contract part a generation holds, or None for any other source.

    Raises:
        DatasetError: If the generation is at a contract location, but its revision does not
            name its part as publish_contract does.
    """
    location = generation.occurrence.declaration.location
    dataset = location.removeprefix(_location(""))
    if dataset == location:
        return None
    _part_index(generation)
    return dataset


@dataclass(frozen=True)
class PeriodContract:
    dataset: str
    generation: str
    manifest: str
    mapping: PeriodMapping


def period_contract(generation: Generation) -> PeriodContract | None:
    """Read monthly authority from a verified generation's structural declaration."""
    declaration = generation.occurrence.declaration
    if not declaration.location.startswith(_location("")):
        return None
    if ", period mapping" not in declaration.revision:
        return None
    try:
        dataset = parse_name(contract_of(generation))
    except ValueError as error:
        raise DatasetError("Invalid dataset period contract") from error
    match = re.fullmatch(
        r"Part 0, period mapping ([^,]+), text SHA-256 ([0-9a-f]{64})", declaration.revision
    )
    if (
        match is None
        or match[1] != PERIOD_MAPPING
        or match[2] != generation.occurrence.blob_digest
        or match[2] != generation.text_digest
        or generation.extractor != TEXT_PIPELINE.extractor
    ):
        raise DatasetError("Invalid dataset period contract")
    return PeriodContract(dataset, generation.id, generation.digest, match[1])


def _location(dataset: str) -> str:
    return f"{ADAPTER}/contract/{dataset}"


def _part_index(generation: Generation) -> int:
    part = re.match(r"Part ([0-9]+),", generation.occurrence.declaration.revision)
    if part is None:
        raise DatasetError(f"The revision of contract generation {generation.id} names no part")
    return int(part[1])


@dataclass(frozen=True)
class _Part:
    available_at: datetime
    availability_evidence: str
    received_at: datetime
    text: str


def _part(raw: object) -> _Part:
    value = parse_fields(raw, "available_at availability_evidence received_at text", "part")
    return _Part(
        parse_time(value["available_at"]),
        parse_text(value["availability_evidence"], "availability_evidence"),
        parse_time(value["received_at"]),
        parse_text(value["text"], "text"),
    )


def _contract(data: bytes, dataset: str) -> tuple[tuple[_Part, ...], str | None]:
    try:
        raw = canonical_json(data, "contract")
        version = raw.get("schema_version") if isinstance(raw, dict) else None
        if type(version) is not int or version not in (1, 2):
            raise PackageError("Unsupported contract schema version")
        names = "dataset parts schema_version" + (" period_mapping" if version == 2 else "")
        value = parse_fields(raw, names, "contract")
        if value["dataset"] != dataset:
            raise PackageError(f"Contract holds dataset {value['dataset']!r}")
        parts = tuple(_part(raw) for raw in parse_array(value["parts"], "parts"))
        mapping = value.get("period_mapping")
        if version == 2 and (mapping != PERIOD_MAPPING or not parts):
            raise PackageError(
                "A monthly contract requires a supported mapping and structural part"
            )
        return parts, mapping
    except (ValueError, RecursionError) as error:
        raise DatasetError(f"Invalid contract of {dataset!r}: {error}") from error


def save(store: Path, dataset: str, data: bytes) -> str:
    """Keep a dataset file under its vintage, so every earlier vintage stays readable.

    Args:
        store (Path): Dataset store directory.
        dataset (str): Dataset name, which the file must hold.
        data (bytes): Canonical dataset file bytes.

    Returns:
        (str) The vintage, the SHA-256 of the bytes.

    Raises:
        DatasetError: If the name is not a logical name, the bytes are not a valid file of that
            dataset, or different bytes already hold the vintage's path.
    """
    try:
        directory = store / parse_name(dataset)
    except PackageError as error:
        raise DatasetError(f"Invalid dataset name: {error}") from error
    _parse(data, dataset)
    vintage = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    if not write_once(directory / f"{vintage}.json", data):
        raise DatasetError(f"Stored bytes of {dataset!r} do not match vintage {vintage}")
    return vintage


def _load(data: bytes, names: str, label: str) -> dict:
    value = parse_fields(json.loads(data.decode("utf-8")), names, label)
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise PackageError(f"Unsupported schema version {value['schema_version']!r}")
    if canonical_bytes(value) != data:
        raise PackageError(f"{label} is not canonical UTF-8 JSON")
    return value


def _identity(value: dict) -> tuple[str, int]:
    revision = value["revision"]
    if type(revision) is not int or revision < 1:
        raise PackageError(f"revision must be a positive integer, not {revision!r}")
    return parse_text(value["key"], "key"), revision


def _row(raw: object) -> Row:
    value = parse_fields(raw, "arrived_at key published_at revision values", "row")
    if not isinstance(value["values"], dict):
        raise PackageError("values must be an object")
    key, revision = _identity(value)
    return Row(
        key,
        revision,
        parse_time(value["published_at"]),
        parse_time(value["arrived_at"]),
        Values(value["values"]),
    )


def _verified(path: Path, dataset: str, vintage: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise DatasetError(f"Cannot read dataset {dataset!r}: {error}") from error
    if hashlib.sha256(data).hexdigest() != vintage:
        raise DatasetError(f"Stored bytes of {dataset!r} are not its documented vintage")
    return data


@lru_cache(maxsize=8)
def _stored(path: Path, dataset: str, vintage: str) -> tuple[Row, ...]:
    """Parse a stored vintage once per process; read still verifies the file on every call.

    The file is content-addressed, so an entry never goes stale, and its rows' values refuse
    change, so no reader's edit reaches another read. A refused file raises and is not cached.
    """
    return _parse(_verified(path, dataset, vintage), dataset)


def _parse(data: bytes, dataset: str) -> tuple[Row, ...]:
    try:
        value = _load(data, "dataset rows schema_version", "dataset file")
        if value["dataset"] != dataset:
            raise PackageError(f"File holds dataset {value['dataset']!r}")
        rows = tuple(_row(raw) for raw in parse_array(value["rows"], "rows"))
        if len({(row.key, row.revision) for row in rows}) != len(rows):
            raise PackageError("Row identities must be unique")
    # Rows past the cutoff or clearance are parsed too, so the message quotes none of the file.
    except (ValueError, RecursionError) as error:
        raise DatasetError(f"Invalid dataset {dataset!r}") from error
    return rows
