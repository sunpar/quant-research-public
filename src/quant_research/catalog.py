from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from types import MappingProxyType
from typing import ClassVar, Literal, get_args

from quant_research.records import (
    PackageError,
    canonical_bytes,
    json_default,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
)

type Status = Literal["documented", "validated", "approved", "revoked"]
type Aspect = Literal["publication", "revision", "arrival"]

VINTAGE_GAPS: Mapping[Aspect, str] = MappingProxyType(
    {aspect: f"No {aspect} vintage evidence" for aspect in get_args(Aspect.__value__)}
)


class CatalogError(ValueError):
    """Report a malformed catalog value, a refused transition, or noncanonical catalog bytes."""


@dataclass(frozen=True)
class Evidence:
    """Commit a claim to the SHA-256 digest of material that someone can produce and recheck."""

    digest: str
    summary: str


@dataclass(frozen=True)
class Vintage:
    aspect: Aspect
    evidence: Evidence


@dataclass(frozen=True)
class Documented:
    """Register a dataset, or amend its documentation while no validation or approval holds.

    vintage is the SHA-256 of the dataset bytes this documentation describes, so validation and
    approval cover exactly those bytes. approved_for names the providers that may receive them.
    """

    dataset: str
    actor: str
    at: datetime
    owner: str
    fields: tuple[str, ...]
    access: tuple[str, ...]
    vintage: str
    known_gaps: tuple[str, ...] = ()
    vintages: tuple[Vintage, ...] = ()
    approved_for: tuple[str, ...] = ()
    kind: ClassVar[Status] = "documented"


@dataclass(frozen=True)
class Validated:
    """Attest the documentation in force with manual validation evidence."""

    dataset: str
    actor: str
    at: datetime
    evidence: Evidence
    kind: ClassVar[Status] = "validated"


@dataclass(frozen=True)
class Approved:
    """Approve agent use through a named timing-and-access adapter. The catalog never runs it."""

    dataset: str
    actor: str
    at: datetime
    adapter: str
    kind: ClassVar[Status] = "approved"


@dataclass(frozen=True)
class Revoked:
    """Withdraw a validation or an approval. History keeps both."""

    dataset: str
    actor: str
    at: datetime
    reason: str
    kind: ClassVar[Status] = "revoked"


type Event = Documented | Validated | Approved | Revoked

_SOURCES: dict[Status, frozenset[Status | None]] = {
    "documented": frozenset({None, "documented", "revoked"}),
    "validated": frozenset({"documented"}),
    "approved": frozenset({"validated"}),
    "revoked": frozenset({"validated", "approved"}),
}


@dataclass(frozen=True)
class Entry:
    """Hold one dataset's accepted events, oldest first. Status is the latest event's kind."""

    name: str
    history: tuple[Event, ...]

    @property
    def status(self) -> Status:
        return self.history[-1].kind

    @property
    def documentation(self) -> Documented:
        return next(event for event in reversed(self.history) if isinstance(event, Documented))

    @property
    def gaps(self) -> tuple[str, ...]:
        """Return known gaps, then the gap for each vintage aspect that lacks evidence."""
        return (*self.documentation.known_gaps, *_vintage_gaps(self.documentation))

    @property
    def adapter(self) -> str | None:
        """Return the adapter reference while the latest event approves, otherwise None."""
        latest = self.history[-1]
        return latest.adapter if isinstance(latest, Approved) else None


@dataclass(frozen=True)
class Catalog:
    """Keep dataset events in an append-only log with derived, searchable entries.

    Every construction, including `append` and `from_bytes`, parses each event strictly and
    replays the transition table, so no catalog holds an approval its evidence does not support.
    """

    events: tuple[Event, ...] = ()
    _entries: dict[str, Entry] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            events = tuple(_event(raw) for raw in json.loads(self.to_bytes())["events"])
        except PackageError as error:
            raise CatalogError(f"Invalid catalog event: {error}") from error
        entries: dict[str, Entry] = {}
        for event in events:
            entries[event.dataset] = _step(entries.get(event.dataset), event)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "_entries", entries)

    def append(self, event: Event) -> Catalog:
        """Return a new catalog with one more event. This catalog is unchanged.

        Args:
            event (Event): Next event for one dataset.

        Returns:
            (Catalog) This catalog's events followed by `event`.

        Raises:
            CatalogError: If the event is malformed or the transition table refuses it.
        """
        return Catalog((*self.events, event))

    def entry(self, name: str) -> Entry:
        """Return one dataset's entry.

        Args:
            name (str): Exact dataset name.

        Returns:
            (Entry) The dataset's accepted events and derived state.

        Raises:
            CatalogError: If no event names that dataset.
        """
        entry = self._entries.get(name)
        if entry is None:
            raise CatalogError(f"Unknown dataset {name!r}")
        return entry

    def search(self, text: str = "", *, status: Status | None = None) -> tuple[Entry, ...]:
        """Return the entries that match every term and the status, ordered by name.

        Args:
            text (str, optional): Whitespace-separated terms. Each must appear, ignoring case, in
                the name, the owner, or a field. Empty text matches every entry.
            status (Status | None, optional): Exact status to keep, or None for every status.

        Returns:
            (tuple) Matching entries ordered by name.
        """
        terms = text.casefold().split()
        matches = []
        for name, entry in sorted(self._entries.items()):
            values = (name, entry.documentation.owner, *entry.documentation.fields)
            if (status is None or entry.status == status) and all(
                any(term in value.casefold() for value in values) for term in terms
            ):
                matches.append(entry)
        return tuple(matches)

    def to_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON with the schema version and every event in order."""
        events = [{"kind": event.kind, **asdict(event)} for event in self.events]
        return canonical_bytes(
            json.loads(json.dumps({"schema_version": 1, "events": events}, default=json_default))
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Catalog:
        """Parse canonical catalog bytes and replay every event.

        Args:
            data (bytes): Canonical UTF-8 JSON, as `to_bytes` returns it.

        Returns:
            (Catalog) The catalog those bytes encode.

        Raises:
            CatalogError: If the bytes are not canonical, or encode a malformed event or a refused
                transition.
        """
        try:
            value = parse_fields(
                json.loads(data.decode("utf-8")), "schema_version events", "catalog"
            )
            if type(value["schema_version"]) is not int or value["schema_version"] != 1:
                raise CatalogError(f"Unsupported schema version {value['schema_version']!r}")
            catalog = cls(tuple(_event(raw) for raw in parse_array(value["events"], "events")))
            if catalog.to_bytes() != data:
                raise CatalogError("Catalog is not canonical UTF-8 JSON")
            return catalog
        except (ValueError, RecursionError) as error:
            raise CatalogError(f"Invalid catalog: {error}") from error

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the canonical bytes, which identifies this revision."""
        return hashlib.sha256(self.to_bytes()).hexdigest()


def _step(entry: Entry | None, event: Event) -> Entry:
    status = entry.status if entry else None
    if status not in _SOURCES[event.kind]:
        raise CatalogError(
            f"Cannot record {event.kind} for {event.dataset!r} while {status or 'unregistered'}"
        )
    if entry and event.at < entry.history[-1].at:
        raise CatalogError(f"Event for {event.dataset!r} predates its latest event")
    if isinstance(event, Approved) and _vintage_gaps(entry.documentation):
        raise CatalogError(f"Approval of {event.dataset!r} needs evidence for every vintage aspect")
    return Entry(event.dataset, (*entry.history, event) if entry else (event,))


def _vintage_gaps(documentation: Documented) -> tuple[str, ...]:
    present = {vintage.aspect for vintage in documentation.vintages}
    return tuple(gap for aspect, gap in VINTAGE_GAPS.items() if aspect not in present)


def _texts(value: object, label: str) -> tuple[str, ...]:
    return tuple(parse_text(item, label) for item in parse_array(value, label))


def _evidence(raw: object) -> Evidence:
    value = parse_fields(raw, "digest summary", "evidence")
    return Evidence(parse_digest(value["digest"]), parse_text(value["summary"], "evidence summary"))


def _vintage(raw: object) -> Vintage:
    value = parse_fields(raw, "aspect evidence", "vintage")
    if value["aspect"] not in get_args(Aspect.__value__):
        raise CatalogError(f"Unsupported vintage aspect {value['aspect']!r}")
    return Vintage(value["aspect"], _evidence(value["evidence"]))


_PARSERS = {
    "dataset": lambda value: parse_text(value, "dataset"),
    "actor": lambda value: parse_text(value, "actor"),
    "at": parse_time,
    "owner": lambda value: parse_text(value, "owner"),
    "fields": lambda value: _texts(value, "fields"),
    "access": lambda value: _texts(value, "access"),
    "vintage": parse_digest,
    "approved_for": lambda value: _texts(value, "approved_for"),
    "known_gaps": lambda value: _texts(value, "known_gaps"),
    "vintages": lambda value: tuple(_vintage(item) for item in parse_array(value, "vintages")),
    "evidence": _evidence,
    "adapter": lambda value: parse_text(value, "adapter"),
    "reason": lambda value: parse_text(value, "reason"),
}


def _event(raw: object) -> Event:
    kind = raw.get("kind") if isinstance(raw, dict) else None
    event_type = next((item for item in get_args(Event.__value__) if item.kind == kind), None)
    if event_type is None:
        raise CatalogError(f"Unsupported catalog event kind {kind!r}")
    names = [item.name for item in fields(event_type)]
    value = parse_fields(raw, " ".join(("kind", *names)), f"{kind} event")
    return event_type(**{name: _PARSERS[name](value[name]) for name in names})
