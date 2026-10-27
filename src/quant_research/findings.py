"""Record claims, findings, and lessons as reviewed states over an append-only event log.

Records are immutable values addressed by their content. Events are the only way state
changes, and every status is a fold over them:

    proposed --review--> reviewed --admit (lessons only)--> admitted
    proposed | reviewed | admitted | suspended --suspend--> suspended --review--> reviewed
    proposed | reviewed | admitted | suspended --retract--> retracted (terminal)
    proposed | reviewed | admitted | suspended --supersede--> superseded (terminal)

A claim or finding stands while reviewed and a lesson while admitted, each only while its
support holds. Support holds while every record reached through supporting links, including
records beneath a suspended one, has its kind's standing status or is suspended. The record
itself and every reached record that is not suspended must also keep a supporting item other
than a citation of a suspended record, unless it has no supporting links. A suspension thus
never hides a retraction or supersession beneath it. An admitted lesson that stands is
eligible for reuse.

A declaration states the restrictions of one external evidence ID, and a repeat only tightens
them. A record's restrictions derive from every external item reached through its links of
either relation. They are unknown, so no run may read the record, while a reached item is
undeclared or a reached record has no evidence. Each log prefix has a head, a SHA-256 chain
over its events that starts at NO_MEMORY. A run reads the lessons eligible in the prefix whose
head its context pins, when it may read their restrictions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from functools import cached_property
from itertools import pairwise
from typing import Literal, get_args

from quant_research.access import NO_MEMORY, MemoryPin, Restrictions, RunContext, derive
from quant_research.records import (
    canonical_bytes,
    json_default,
    parse_array,
    parse_fields,
    parse_time,
)

type Relation = Literal["supports", "contradicts"]
type Origin = Literal["extracted", "hypothesis", "summary"]
type Outcome = Literal["supports_idea", "rejects_idea", "inconclusive", "invalid_implementation"]
type Action = Literal["review", "admit", "retract", "suspend"]
type Status = Literal["proposed", "reviewed", "admitted", "suspended", "retracted", "superseded"]


class ReviewError(ValueError):
    """Report an event that the review state machine rejects."""


@dataclass(frozen=True)
class Cites:
    record_id: str
    relation: Relation


@dataclass(frozen=True)
class External:
    """Cite evidence owned outside the ledger, such as an artifact digest. It always stands."""

    evidence_id: str
    relation: Relation


type Evidence = Cites | External


@dataclass(frozen=True)
class Scope:
    universe: str
    horizon: str
    start: datetime
    end: datetime
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Claim:
    statement: str
    origin: Origin
    author: str
    scope: Scope
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Finding:
    statement: str
    protocol: str
    outcome: Outcome
    author: str
    scope: Scope
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Lesson:
    statement: str
    author: str
    scope: Scope
    evidence: tuple[Evidence, ...]


type Record = Claim | Finding | Lesson


@dataclass(frozen=True)
class Proposal:
    record: Record
    at: datetime


@dataclass(frozen=True)
class Review:
    record_id: str
    action: Action
    reviewer: str
    at: datetime
    reason: str


@dataclass(frozen=True)
class Supersession:
    record_id: str
    replacement: str
    reviewer: str
    at: datetime
    reason: str


@dataclass(frozen=True)
class Declaration:
    """State the restrictions of one external evidence ID.

    The operator or the ingestion code that holds the evidence's own restrictions appends one.
    A record states no restrictions, so its author cannot understate what it cites.
    """

    evidence_id: str
    restrictions: Restrictions
    declarer: str
    at: datetime
    reason: str


type Event = Proposal | Review | Supersession | Declaration

_LIVE: tuple[Status, ...] = ("proposed", "reviewed", "admitted")
_TRANSITIONS: dict[tuple[Status, str], Status] = {
    ("proposed", "review"): "reviewed",
    ("suspended", "review"): "reviewed",
    ("reviewed", "admit"): "admitted",
    **{(status, "suspend"): "suspended" for status in (*_LIVE, "suspended")},
    **{(status, "retract"): "retracted" for status in (*_LIVE, "suspended")},
    **{(status, "supersede"): "superseded" for status in (*_LIVE, "suspended")},
}
_STANDING: dict[type, Status] = {Claim: "reviewed", Finding: "reviewed", Lesson: "admitted"}


def record_id(record: Record) -> str:
    """Return the content address of a record.

    Args:
        record (Record): Claim, finding, or lesson.

    Returns:
        (str) SHA-256 hex digest of the canonical JSON of the record and its kind.
    """
    return hashlib.sha256(canonical_bytes(_value(record))).hexdigest()


def event_value(event: Event) -> dict:
    """Return the canonical JSON value of an event, the value that each head link chains.

    Args:
        event (Event): Proposal, review, supersession, or declaration.

    Returns:
        (dict) The event's kind and fields, with a proposal's record under its own kind, times
        as UTC text ending in Z, and restriction labels and providers as sorted lists.
    """
    return _value(event)


def parse_event(value: object) -> Event:
    """Rebuild an event from the value that event_value returns.

    Only the shape is checked here. The ledger that folds the event checks its text, choices,
    and the records it names, so any text a ledger accepts parses again.

    Args:
        value (object): Canonical JSON value of one event.

    Returns:
        (Event) The event whose event_value equals value.

    Raises:
        ReviewError: If value is not the canonical value of an event.
    """
    try:
        event = _item(value, _EVENTS)
        if event_value(event) != value:
            raise ReviewError("Event value is not canonical")
        return event
    except (TypeError, ValueError) as error:
        raise ReviewError(f"Invalid review event: {error}") from error


def _value(item: Record | Event) -> dict:
    data = asdict(item)
    if isinstance(item, Proposal):
        data["record"] = _value(item.record)
    return json.loads(
        json.dumps(
            {"kind": _kind(item), **data},
            default=lambda value: (
                sorted(value) if isinstance(value, frozenset) else json_default(value)
            ),
        )
    )


def _kind(item: Record | Event) -> str:
    return type(item).__name__.lower()


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ReviewError(f"Expected text, not {type(value).__name__}")
    return value


def _strings(value: object, label: str) -> list[str]:
    return [_string(item) for item in parse_array(value, label)]


def _scope(raw: object) -> Scope:
    value = parse_fields(raw, "universe horizon start end assumptions", "scope")
    return Scope(
        _string(value["universe"]),
        _string(value["horizon"]),
        parse_time(value["start"]),
        parse_time(value["end"]),
        tuple(_strings(value["assumptions"], "assumptions")),
    )


def _link(raw: object) -> Evidence:
    if isinstance(raw, dict) and "record_id" in raw:
        value = parse_fields(raw, "record_id relation", "cited record")
        return Cites(_string(value["record_id"]), _string(value["relation"]))
    value = parse_fields(raw, "evidence_id relation", "external evidence")
    return External(_string(value["evidence_id"]), _string(value["relation"]))


def _evidence(value: object) -> tuple[Evidence, ...]:
    return tuple(_link(item) for item in parse_array(value, "evidence"))


def _claim(value: dict) -> Claim:
    return Claim(
        _string(value["statement"]),
        _string(value["origin"]),
        _string(value["author"]),
        _scope(value["scope"]),
        _evidence(value["evidence"]),
    )


def _finding(value: dict) -> Finding:
    return Finding(
        _string(value["statement"]),
        _string(value["protocol"]),
        _string(value["outcome"]),
        _string(value["author"]),
        _scope(value["scope"]),
        _evidence(value["evidence"]),
    )


def _lesson(value: dict) -> Lesson:
    return Lesson(
        _string(value["statement"]),
        _string(value["author"]),
        _scope(value["scope"]),
        _evidence(value["evidence"]),
    )


def _proposal(value: dict) -> Proposal:
    return Proposal(_item(value["record"], _RECORDS), parse_time(value["at"]))


def _review(value: dict) -> Review:
    return Review(
        _string(value["record_id"]),
        _string(value["action"]),
        _string(value["reviewer"]),
        parse_time(value["at"]),
        _string(value["reason"]),
    )


def _supersession(value: dict) -> Supersession:
    return Supersession(
        _string(value["record_id"]),
        _string(value["replacement"]),
        _string(value["reviewer"]),
        parse_time(value["at"]),
        _string(value["reason"]),
    )


def _declaration(value: dict) -> Declaration:
    restrictions = parse_fields(
        value["restrictions"], "available_at labels approved_for", "restrictions"
    )
    return Declaration(
        _string(value["evidence_id"]),
        Restrictions(
            available_at=parse_time(restrictions["available_at"]),
            labels=frozenset(_strings(restrictions["labels"], "labels")),
            approved_for=frozenset(_strings(restrictions["approved_for"], "approved_for")),
        ),
        _string(value["declarer"]),
        parse_time(value["at"]),
        _string(value["reason"]),
    )


_RECORDS: dict[str, tuple[type, Callable[[dict], Record]]] = {
    "claim": (Claim, _claim),
    "finding": (Finding, _finding),
    "lesson": (Lesson, _lesson),
}
_EVENTS: dict[str, tuple[type, Callable[[dict], Event]]] = {
    "proposal": (Proposal, _proposal),
    "review": (Review, _review),
    "supersession": (Supersession, _supersession),
    "declaration": (Declaration, _declaration),
}


def _item[T](raw: object, kinds: dict[str, tuple[type, Callable[[dict], T]]]) -> T:
    kind = raw.get("kind") if isinstance(raw, dict) else None
    if kind not in kinds:
        raise ReviewError(f"Unknown kind {kind!r}")
    item_type, parse = kinds[kind]
    names = " ".join(["kind", *(field.name for field in fields(item_type))])
    return parse(parse_fields(raw, names, kind))


def _choice(value: object, choices: object, label: str) -> None:
    if value not in get_args(choices.__value__):
        raise ReviewError(f"Unsupported {label} {value!r}")


def _text(*values: object) -> None:
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ReviewError("Record and review text must be nonblank strings")


@dataclass(frozen=True)
class _Entry:
    record: Record
    status: Status
    history: tuple[Event, ...]


class Ledger:
    """Hold an append-only review log and the record states folded from it."""

    def __init__(self, events: tuple[Event, ...] = ()) -> None:
        """Fold a new or stored log, raising ReviewError at the first illegal event.

        Args:
            events (tuple): Events in non-decreasing time order.
        """
        if any(event.at.utcoffset() is None for event in events):
            raise ReviewError("Event times must include a timezone")
        # Datetimes sharing one tzinfo compare by wall clock, which misorders a repeated hour.
        if any(
            later.at.astimezone(UTC) < earlier.at.astimezone(UTC)
            for earlier, later in pairwise(events)
        ):
            raise ReviewError("Event times must not decrease")
        self._events = tuple(events)
        self._entries: dict[str, _Entry] = {}
        self._declared: dict[str, Restrictions] = {}
        for event in events:
            self._fold(event)

    @property
    def events(self) -> tuple[Event, ...]:
        """Return the log that every status and eligibility answer derives from."""
        return self._events

    @property
    def head(self) -> MemoryPin:
        """Return the chained SHA-256 of every event, the memory pin that names this exact log."""
        return self._heads[-1]

    def apply(self, event: Event) -> Ledger:
        """Return a new ledger with the event appended, leaving this ledger unchanged."""
        return Ledger((*self.events, event))

    def as_of(self, at: datetime) -> Ledger:
        """Return the ledger built from the events at or before a time."""
        return Ledger(tuple(event for event in self.events if event.at.astimezone(UTC) <= at))

    def record(self, record_id: str) -> Record:
        """Return the record proposed under a content address."""
        return self._entry(record_id).record

    def status(self, record_id: str) -> Status:
        """Return the current status of a record."""
        return self._entry(record_id).status

    def history(self, record_id: str) -> tuple[Event, ...]:
        """Return the events that moved a record, oldest first."""
        return self._entry(record_id).history

    def stands(self, record_id: str) -> bool:
        """Return whether a record has its kind's standing status and its support holds.

        Args:
            record_id (str): Content address of a proposed record.

        Returns:
            (bool) True for a reviewed claim or finding, or an admitted lesson, whose support holds.
        """
        entry = self._entry(record_id)
        return entry.status == _STANDING[type(entry.record)] and self._supported(record_id)

    def eligible_lessons(self) -> tuple[str, ...]:
        """Return admitted lessons that stand, in proposal order."""
        return tuple(
            identity
            for identity, entry in self._entries.items()
            if isinstance(entry.record, Lesson) and self.stands(identity)
        )

    def restrictions(self, record_id: str) -> Restrictions | None:
        """Return the restrictions a record derives from every external item it reaches.

        Items are reached through supporting and contradicting links, transitively through cited
        records whatever their status, because a statement can carry anything its author read.

        Args:
            record_id (str): Content address of a proposed record.

        Returns:
            (Restrictions | None) The derived restrictions, or None when a reached record has no
            evidence or a reached item is undeclared, so no run may read the record.
        """
        records = [
            self._entry(record_id).record,
            *(
                self._entries[identity].record
                for identity in self._reach(record_id, get_args(Relation.__value__))
            ),
        ]
        items = {
            item.evidence_id
            for record in records
            for item in record.evidence
            if isinstance(item, External)
        }
        if not all(record.evidence for record in records) or not items <= self._declared.keys():
            return None
        # Every reached record has evidence, so each chain ends in an item and derive gets an input.
        return derive(*(self._declared[item] for item in items))

    def admissible_lessons(self, context: RunContext) -> tuple[str, ...]:
        """Return IDs of lessons a run may read at its memory pin, in proposal order."""
        return tuple(identity for identity, _ in self.admissible_lesson_restrictions(context))

    def admissible_lesson_restrictions(
        self, context: RunContext
    ) -> tuple[tuple[str, Restrictions], ...]:
        """Return admissible lesson IDs and their restrictions at the run's memory pin.

        A lesson is admissible when it is eligible in the log prefix whose head is the pin, so
        the suspensions recorded by then apply, and the run may read the restrictions that the
        prefix derives for it. Review times never meet the cutoff.

        Args:
            context (RunContext): Trusted authority of the run.

        Returns:
            (tuple) Pairs of lesson IDs and restrictions, in proposal order.

        Raises:
            ReviewError: If context is not a RunContext, or no prefix of this log has its pin.
        """
        # A stand-in whose read_denials returns () would make every lesson readable.
        if not isinstance(context, RunContext):
            raise ReviewError(f"context must be a RunContext, not {type(context).__name__}")
        try:
            length = self._heads.index(context.memory)
        except ValueError:
            raise ReviewError("No prefix of the review log has the run's memory pin") from None
        pinned = Ledger(self._events[:length])
        return tuple(
            (identity, found)
            for identity in pinned.eligible_lessons()
            if (found := pinned.restrictions(identity)) is not None
            and not context.read_denials(found)
        )

    @cached_property
    def _heads(self) -> tuple[MemoryPin, ...]:
        heads = [NO_MEMORY]
        for event in self._events:
            link = canonical_bytes({"event": _value(event), "previous": heads[-1]})
            heads.append(MemoryPin(hashlib.sha256(link).hexdigest()))
        return tuple(heads)

    def _entry(self, record_id: str) -> _Entry:
        if record_id not in self._entries:
            raise ReviewError(f"Unknown record {record_id}")
        return self._entries[record_id]

    def _reach(self, record_id: str, relations: tuple[Relation, ...]) -> set[str]:
        found: set[str] = set()
        pending = [record_id]
        while pending:
            cited = {
                item.record_id
                for item in self._entries[pending.pop()].record.evidence
                if isinstance(item, Cites) and item.relation in relations
            }
            pending.extend(cited - found)
            found |= cited
        return found

    def _supported(self, record_id: str) -> bool:
        for identity in (record_id, *self._reach(record_id, ("supports",))):
            entry = self._entries[identity]
            if identity != record_id and entry.status == "suspended":
                continue
            if identity != record_id and entry.status != _STANDING[type(entry.record)]:
                return False
            links = [item for item in entry.record.evidence if item.relation == "supports"]
            if links and all(
                isinstance(item, Cites) and self._entries[item.record_id].status == "suspended"
                for item in links
            ):
                return False
        return True

    def _fold(self, event: Event) -> None:
        if isinstance(event, Proposal):
            self._propose(event)
            return
        if isinstance(event, Declaration):
            _text(event.evidence_id, event.declarer, event.reason)
            # A first declaration derives with itself, which changes nothing.
            known = self._declared.get(event.evidence_id, event.restrictions)
            self._declared[event.evidence_id] = derive(known, event.restrictions)
            return
        _text(event.reviewer, event.reason)
        if isinstance(event, Review):
            _choice(event.action, Action, "review action")
        entry = self._entry(event.record_id)
        kind = _kind(entry.record)
        action = event.action if isinstance(event, Review) else "supersede"
        status = _TRANSITIONS.get((entry.status, action))
        if status is None:
            raise ReviewError(f"Cannot {action} a {kind} that is {entry.status}")
        if action == "admit":
            if not isinstance(entry.record, Lesson):
                raise ReviewError(f"Cannot admit a {kind}; only lessons are admitted")
            if not any(item.relation == "supports" for item in entry.record.evidence):
                raise ReviewError("Cannot admit a lesson without a supporting link")
            if not self._supported(event.record_id):
                raise ReviewError("Cannot admit a lesson while a supporting record does not stand")
        if isinstance(event, Supersession):
            replacement = self._entry(event.replacement)
            if type(replacement.record) is not type(entry.record):
                raise ReviewError(f"Cannot supersede a {kind} with a {_kind(replacement.record)}")
            if replacement.status not in _LIVE:
                raise ReviewError(f"Cannot supersede with a {replacement.status} replacement")
            if event.record_id in {
                event.replacement,
                *self._reach(event.replacement, ("supports",)),
            }:
                raise ReviewError(f"The replacement rests on the {kind} it would supersede")
        self._entries[event.record_id] = replace(
            entry, status=status, history=(*entry.history, event)
        )

    def _propose(self, event: Proposal) -> None:
        record = event.record
        _text(
            record.statement,
            record.author,
            record.scope.universe,
            record.scope.horizon,
            *record.scope.assumptions,
            *(item.evidence_id for item in record.evidence if isinstance(item, External)),
        )
        identity = record_id(record)
        if identity in self._entries:
            raise ReviewError(f"Record {identity} is already proposed")
        for item in record.evidence:
            _choice(item.relation, Relation, "evidence relation")
        if isinstance(record, Claim):
            _choice(record.origin, Origin, "claim origin")
        if isinstance(record, Finding):
            _choice(record.outcome, Outcome, "finding outcome")
            _text(record.protocol)
        cited = [self._entry(item.record_id) for item in record.evidence if isinstance(item, Cites)]
        if isinstance(record, Claim) and record.origin == "summary" and not cited:
            raise ReviewError("A summary claim must cite the records it summarizes")
        if record.scope.end.astimezone(UTC) < record.scope.start.astimezone(UTC):
            raise ReviewError("Scope period ends before it starts")
        self._entries[identity] = _Entry(record, "proposed", (event,))
