"""Trace an invalidated piece of external evidence to the records that lose support.

An `Invalidation` names one evidence ID, the SHA-256 digest that records cite through
`External`, and why it no longer holds. `assess` turns it into an `Impact`:

    Impact(invalidation, withdrawn, effects)
    Effect(record_id, kind, outcome, via, supported)

`withdrawn` is the evidence ID, then for a corrected or revoked dataset vintage the digest of
every package whose trials declare that vintage as an input. Each effect is a record that cites
a withdrawn ID or another affected record as support. `via` lists those citations in evidence
order, and `supported` says whether an independent supporting item remains. `assess` first
replays the `earlier` invalidations still in force against the same ledger, so their withdrawn
evidence and the records they leave without support no longer count as remaining support.
Only the current invalidation decides which records are affected. `suspend` appends a
suspension review for each record that lost all support. Records are never rewritten, and a
later review restores them. `catch_up` brings a ledger that moved on up to date by assessing and
applying each invalidation in force in replay order, passing the ones before it as `earlier`.
It repeats until nothing new is appended, because a later invalidation's suspension can remove
the last support an earlier one left a record, so that record's history names both causes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Set
from dataclasses import asdict, dataclass
from datetime import datetime
from operator import attrgetter
from typing import Literal, get_args

from quant_research.findings import (
    Cites,
    Event,
    External,
    Finding,
    Ledger,
    Outcome,
    Proposal,
    Review,
    record_id,
)
from quant_research.records import (
    canonical_bytes,
    json_default,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
)

type Kind = Literal["corrected_vintage", "revoked_vintage", "invalid_package", "retracted_source"]
type RecordKind = Literal["claim", "finding", "lesson"]

_REPLAY_ORDER = attrgetter("at", "evidence_id", "kind", "actor", "reason")


class InvalidationError(ValueError):
    """Report a malformed invalidation or noncanonical impact bytes."""


@dataclass(frozen=True)
class Invalidation:
    evidence_id: str
    kind: Kind
    actor: str
    at: datetime
    reason: str


@dataclass(frozen=True)
class Effect:
    record_id: str
    kind: RecordKind
    outcome: Outcome | None
    via: tuple[str, ...]
    supported: bool


@dataclass(frozen=True)
class Impact:
    invalidation: Invalidation
    withdrawn: tuple[str, ...]
    effects: tuple[Effect, ...]

    def to_bytes(self) -> bytes:
        """Return canonical UTF-8 JSON with the schema version, the invalidation, and effects."""
        return canonical_bytes(
            json.loads(json.dumps({"schema_version": 1, **asdict(self)}, default=json_default))
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Impact:
        """Parse canonical impact bytes.

        Args:
            data (bytes): Canonical UTF-8 JSON, as `to_bytes` returns it.

        Returns:
            (Impact) The impact those bytes encode.

        Raises:
            InvalidationError: If the bytes are not canonical or encode a malformed field.
        """
        try:
            value = parse_fields(
                json.loads(data.decode("utf-8")),
                "schema_version invalidation withdrawn effects",
                "impact",
            )
            if type(value["schema_version"]) is not int or value["schema_version"] != 1:
                raise InvalidationError(f"Unsupported schema version {value['schema_version']!r}")
            impact = cls(
                _invalidation(value["invalidation"]),
                tuple(parse_digest(item) for item in parse_array(value["withdrawn"], "withdrawn")),
                tuple(_effect(item) for item in parse_array(value["effects"], "effects")),
            )
            if impact.to_bytes() != data:
                raise InvalidationError("Impact is not canonical UTF-8 JSON")
            return impact
        except (ValueError, RecursionError) as error:
            raise InvalidationError(f"Invalid impact: {error}") from error


def assess(
    ledger: Ledger,
    package_inputs: Mapping[str, Set[str]],
    invalidation: Invalidation,
    *,
    earlier: Iterable[Invalidation] = (),
) -> Impact:
    """Find every record whose support reaches withdrawn evidence through citations.

    Args:
        ledger (Ledger): Review log whose records cite evidence.
        package_inputs (Mapping): Each package digest that records may cite, with the input
            digests that any of its trials declares.
        invalidation (Invalidation): Evidence that no longer holds.
        earlier (Iterable): Every other invalidation in force, in any order. Each is replayed
            against this ledger first, ordered by time, then evidence ID, kind, actor, and
            reason. A record one of them leaves without
            support counts as lost until its history holds that invalidation's suspension.
            From then on the ledger's status decides, so a re-review restores it.

    Returns:
        (Impact) Withdrawn evidence IDs and the affected records in proposal order.

    Raises:
        InvalidationError: If an invalidation is malformed.
    """
    replay = (
        *sorted(map(parse_invalidation, earlier), key=_REPLAY_ORDER),
        parse_invalidation(invalidation),
    )
    invalid: set[str] = set()
    lost: set[str] = set()
    for current in replay:
        evidence_id = current.evidence_id
        readers = {
            digest
            for digest, inputs in package_inputs.items()
            if current.kind in ("corrected_vintage", "revoked_vintage") and evidence_id in inputs
        }
        withdrawn = (evidence_id, *sorted(readers))
        invalid.update(withdrawn)
        affected: set[str] = set()
        effects = []
        for event in ledger.events:
            if not isinstance(event, Proposal):
                continue
            record = event.record
            support = [item for item in record.evidence if item.relation == "supports"]
            via = tuple(
                item.evidence_id if isinstance(item, External) else item.record_id
                for item in support
                if (isinstance(item, External) and item.evidence_id in withdrawn)
                or (isinstance(item, Cites) and item.record_id in affected)
            )
            if not via:
                continue
            supported = any(
                item.evidence_id not in invalid
                if isinstance(item, External)
                else item.record_id not in lost and ledger.stands(item.record_id)
                for item in support
            )
            identity = record_id(record)
            affected.add(identity)
            if not supported and not _recorded(ledger.history(identity), current):
                lost.add(identity)
            effects.append(
                Effect(
                    identity,
                    type(record).__name__.lower(),
                    record.outcome if isinstance(record, Finding) else None,
                    via,
                    supported,
                )
            )
    return Impact(current, withdrawn, tuple(effects))


def suspend(ledger: Ledger, impact: Impact, *, at: datetime | None = None) -> Ledger:
    """Append a suspension for each record that lost all support to an invalidation.

    Records already retracted or superseded, and records whose history already holds this
    invalidation's suspension, receive nothing, so applying one impact twice appends nothing new.

    Args:
        ledger (Ledger): Review log that the impact was assessed against.
        impact (Impact): Assessed invalidation.
        at (datetime | None, optional): Time of the suspensions. Defaults to the invalidation's
            time. A later time records an invalidation replayed after the log moved on.

    Returns:
        (Ledger) The ledger with the suspensions appended.
    """
    invalidation = impact.invalidation
    return Ledger(
        (
            *ledger.events,
            *(
                Review(
                    effect.record_id,
                    "suspend",
                    invalidation.actor,
                    invalidation.at if at is None else at,
                    _reason(invalidation),
                )
                for effect in impact.effects
                if not effect.supported
                and ledger.status(effect.record_id) not in ("retracted", "superseded")
                and not _recorded(ledger.history(effect.record_id), invalidation)
            ),
        )
    )


def catch_up(
    ledger: Ledger,
    package_inputs: Mapping[str, Set[str]],
    invalidations: Iterable[Invalidation],
    *,
    at: datetime,
) -> Ledger:
    """Assess and apply every invalidation in force against a ledger that moved on.

    Passes repeat until one appends nothing. A later invalidation's suspension can remove the
    last support an earlier invalidation left a record, so the next pass suspends the record for
    the earlier one too and its history names both causes.

    Args:
        ledger (Ledger): Review log to bring up to date.
        package_inputs (Mapping): Each package digest that records may cite, with the input
            digests that any of its trials declares.
        invalidations (Iterable): Every invalidation in force, in any order. Each is assessed
            in the order `assess` replays them, against the ledger the ones before it left.
        at (datetime): Time of every appended suspension, no earlier than the ledger's last
            event.

    Returns:
        (Ledger) The ledger with each invalidation's suspensions appended as `suspend` does.

    Raises:
        InvalidationError: If an invalidation is malformed.
    """
    ordered = sorted(map(parse_invalidation, invalidations), key=_REPLAY_ORDER)
    while True:
        before = len(ledger.events)
        for index, current in enumerate(ordered):
            ledger = suspend(
                ledger, assess(ledger, package_inputs, current, earlier=ordered[:index]), at=at
            )
        if len(ledger.events) == before:
            return ledger


def parse_invalidation(invalidation: Invalidation) -> Invalidation:
    """Check an invalidation's fields and normalize its time to UTC.

    Args:
        invalidation (Invalidation): Invalidation as a caller built it or storage returned it.

    Returns:
        (Invalidation) The same invalidation with its time in UTC.

    Raises:
        InvalidationError: If a field is malformed.
    """
    try:
        return _invalidation(json.loads(json.dumps(asdict(invalidation), default=json_default)))
    except ValueError as error:
        raise InvalidationError(f"Invalid invalidation: {error}") from error


def _reason(invalidation: Invalidation) -> str:
    return f"{invalidation.kind} {invalidation.evidence_id}: {invalidation.reason}"


def _recorded(history: tuple[Event, ...], invalidation: Invalidation) -> bool:
    return any(
        isinstance(event, Review)
        and event.action == "suspend"
        and event.reviewer == invalidation.actor
        and event.reason == _reason(invalidation)
        for event in history
    )


def _invalidation(raw: object) -> Invalidation:
    value = parse_fields(raw, "evidence_id kind actor at reason", "invalidation")
    if value["kind"] not in get_args(Kind.__value__):
        raise InvalidationError(f"Unsupported invalidation kind {value['kind']!r}")
    return Invalidation(
        parse_digest(value["evidence_id"]),
        value["kind"],
        parse_text(value["actor"], "actor"),
        parse_time(value["at"]),
        parse_text(value["reason"], "reason"),
    )


def _effect(raw: object) -> Effect:
    value = parse_fields(raw, "record_id kind outcome via supported", "effect")
    if value["kind"] not in get_args(RecordKind.__value__):
        raise InvalidationError(f"Unsupported record kind {value['kind']!r}")
    outcomes = get_args(Outcome.__value__) if value["kind"] == "finding" else (None,)
    if value["outcome"] not in outcomes:
        raise InvalidationError(f"Unsupported {value['kind']} outcome {value['outcome']!r}")
    if type(value["supported"]) is not bool:
        raise InvalidationError("Effect support must be a boolean")
    return Effect(
        parse_digest(value["record_id"]),
        value["kind"],
        value["outcome"],
        tuple(parse_digest(item) for item in parse_array(value["via"], "via")),
        value["supported"],
    )
