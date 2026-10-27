from __future__ import annotations

import json
from collections import ChainMap, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import NamedTuple
from uuid import UUID

from quant_research.packages import Outcome, parse_outcome
from quant_research.records import (
    canonical_bytes,
    json_default,
    parse_array,
    parse_fields,
    parse_text,
    parse_time,
    parse_uuid,
)


class AcceptanceError(ValueError):
    """Report an invalid event, a conflicting duplicate, or a refused acceptance."""


@dataclass(frozen=True)
class Submitted:
    trial_id: UUID
    run_id: UUID
    attempt_id: UUID
    predecessor: UUID | None
    submitted_at: datetime


@dataclass(frozen=True)
class Completed:
    """Report an attempt's outcome and the time the attempt itself finished."""

    attempt_id: UUID
    outcome: Outcome
    finished_at: datetime


@dataclass(frozen=True)
class Accepted:
    """Accept a completed attempt as its run's result, replacing the one it supersedes."""

    attempt_id: UUID
    supersedes: UUID | None
    actor: str
    reason: str
    at: datetime


type Event = Submitted | Completed | Accepted


@dataclass(frozen=True)
class Attempt:
    submission: Submitted
    completion: Completed | None


@dataclass(frozen=True)
class Execution:
    """Hold one run's attempts in submission order and its decisions in record order."""

    trial_id: UUID
    run_id: UUID
    attempts: tuple[Attempt, ...]
    decisions: tuple[Accepted, ...]

    @property
    def accepted(self) -> Attempt | None:
        """Return the attempt the latest decision accepted, or None before any decision."""
        if not self.decisions:
            return None
        return next(
            attempt
            for attempt in self.attempts
            if attempt.submission.attempt_id == self.decisions[-1].attempt_id
        )


def _submitted(value: dict) -> Submitted:
    return Submitted(
        parse_uuid(value["trial_id"]),
        parse_uuid(value["run_id"]),
        parse_uuid(value["attempt_id"]),
        parse_uuid(value["predecessor"]) if value["predecessor"] is not None else None,
        parse_time(value["submitted_at"]),
    )


def _completed(value: dict) -> Completed:
    return Completed(
        parse_uuid(value["attempt_id"]),
        parse_outcome(value["outcome"]),
        parse_time(value["finished_at"]),
    )


def _accepted(value: dict) -> Accepted:
    return Accepted(
        parse_uuid(value["attempt_id"]),
        parse_uuid(value["supersedes"]) if value["supersedes"] is not None else None,
        parse_text(value["actor"], "actor"),
        parse_text(value["reason"], "acceptance reason"),
        parse_time(value["at"]),
    )


_EVENTS: dict[str, tuple[type, Callable[[dict], Event]]] = {
    "submitted": (Submitted, _submitted),
    "completed": (Completed, _completed),
    "accepted": (Accepted, _accepted),
}
_KINDS = {event_type: kind for kind, (event_type, _) in _EVENTS.items()}


def _event(raw: object) -> Event:
    kind = raw.get("kind") if isinstance(raw, dict) else None
    if not isinstance(kind, str) or kind not in _EVENTS:
        raise AcceptanceError(f"Unknown event kind {kind!r}")
    event_type, parse = _EVENTS[kind]
    names = " ".join(["kind", *(item.name for item in fields(event_type))])
    return parse(parse_fields(raw, names, f"{kind} event"))


def _data(event: Event) -> dict:
    return json.loads(
        json.dumps(
            {"kind": _KINDS[type(event)], **asdict(event)}, default=json_default, allow_nan=False
        )
    )


def _identities(event: Submitted) -> tuple[tuple[UUID, tuple[str, UUID | None]], ...]:
    return (
        (event.trial_id, ("trial", None)),
        (event.run_id, ("run", event.trial_id)),
        (event.attempt_id, ("attempt", event.run_id)),
    )


class _State(NamedTuple):
    recorded: set[Event]
    identities: dict[UUID, tuple[str, UUID | None]]
    submissions: dict[UUID, Submitted]
    completions: dict[UUID, Completed]
    decisions: defaultdict[UUID, list[Accepted]]


class AcceptanceLog:
    """Record attempts and explicit acceptance decisions, one accepted result per run.

    record checks each event against the fold of recorded events, appends at most one
    event, and raises AcceptanceError for an invalid, conflicting, or refused event.
    """

    def __init__(self, events: Iterable[Event] = ()) -> None:
        """Record each event in order.

        Args:
            events (Iterable[Event], optional): Events to pass through record.
        """
        self._events: list[Event] = []
        self._state = _State(set(), {}, {}, {}, defaultdict(list))
        for event in events:
            self.record(event)

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def record(self, event: Event) -> None:
        """Append an event in its stored form unless an equal event is already recorded.

        Args:
            event (Event): Submission, completion, or acceptance decision.

        Notes:
            A retry may be submitted before its predecessor completes, and a late completion
            still records after another attempt was accepted. An acceptance names a completed
            attempt and supersedes exactly the run's currently accepted attempt, or None.
        """
        try:
            # Canonical bytes are what dump_log writes, so a recorded log always dumps.
            event = _event(json.loads(canonical_bytes(_data(event))))
        except (ValueError, RecursionError) as error:
            raise AcceptanceError(f"Invalid event: {error}") from error
        state = self._state
        if event in state.recorded:
            return
        match event:
            case Submitted():
                if event.attempt_id in state.submissions:
                    raise AcceptanceError(
                        f"Attempt {event.attempt_id} is already submitted differently"
                    )
                # The overlay leaves a refused event unrecorded yet refuses an ID it gives twice.
                claimed = ChainMap({}, state.identities)
                for identifier, identity in _identities(event):
                    if claimed.setdefault(identifier, identity) != identity:
                        raise AcceptanceError(
                            f"ID {identifier} already names another kind or parent"
                        )
                if event.predecessor is None and event.run_id in state.identities:
                    raise AcceptanceError(
                        "A repeat requires a new run ID; a retry needs a predecessor"
                    )
                if event.predecessor is not None and (
                    state.identities.get(event.predecessor) != ("attempt", event.run_id)
                ):
                    raise AcceptanceError(
                        f"Retry predecessor {event.predecessor} must be a submitted attempt "
                        f"in run {event.run_id}"
                    )
                state.identities.update(_identities(event))
                state.submissions[event.attempt_id] = event
            case Completed():
                if event.attempt_id not in state.submissions:
                    raise AcceptanceError(f"Attempt {event.attempt_id} is not submitted")
                if (prior := state.completions.get(event.attempt_id)) is not None:
                    raise AcceptanceError(
                        f"Attempt {event.attempt_id} already completed as "
                        f"{prior.outcome.status!r} at {prior.finished_at.isoformat()}"
                    )
                state.completions[event.attempt_id] = event
            case Accepted():
                if (submission := state.submissions.get(event.attempt_id)) is None:
                    raise AcceptanceError(f"Attempt {event.attempt_id} is not submitted")
                if event.attempt_id not in state.completions:
                    raise AcceptanceError(
                        f"Attempt {event.attempt_id} has no completion; "
                        "an interrupted attempt cannot be accepted"
                    )
                run_id = submission.run_id
                decisions = state.decisions.get(run_id, ())
                current = decisions[-1].attempt_id if decisions else None
                if current is None and event.supersedes is not None:
                    raise AcceptanceError(f"Run {run_id} has no accepted attempt to supersede")
                if current is not None and event.supersedes is None:
                    raise AcceptanceError(
                        f"Run {run_id} already accepted attempt {current}; "
                        "a new acceptance must supersede it"
                    )
                if event.supersedes != current:
                    raise AcceptanceError(
                        f"Run {run_id} currently accepts attempt {current}, not {event.supersedes}"
                    )
                state.decisions[run_id].append(event)
        state.recorded.add(event)
        self._events.append(event)

    def executions(self) -> dict[UUID, Execution]:
        """Return each run's attempts and decisions, keyed by run ID in submission order."""
        state = self._state
        runs: dict[UUID, list[Attempt]] = {}
        for submission in state.submissions.values():
            runs.setdefault(submission.run_id, []).append(
                Attempt(submission, state.completions.get(submission.attempt_id))
            )
        return {
            run_id: Execution(
                attempts[0].submission.trial_id,
                run_id,
                tuple(attempts),
                tuple(state.decisions.get(run_id, ())),
            )
            for run_id, attempts in runs.items()
        }


def dump_log(log: AcceptanceLog) -> bytes:
    """Encode every recorded event, in record order, as canonical UTF-8 JSON.

    Args:
        log (AcceptanceLog): Log to encode.

    Returns:
        (bytes) Content that load_log accepts and reproduces exactly.
    """
    return canonical_bytes({"schema_version": 1, "events": [_data(event) for event in log.events]})


def load_log(content: bytes) -> AcceptanceLog:
    """Replay stored events through record and require the canonical encoding.

    Args:
        content (bytes): Output of dump_log.

    Returns:
        (AcceptanceLog) The log that dump_log encodes back to the same bytes.
    """
    try:
        value = parse_fields(
            json.loads(content.decode("utf-8")), "schema_version events", "acceptance log"
        )
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise AcceptanceError(f"Unsupported schema version {value['schema_version']!r}")
        log = AcceptanceLog(_event(item) for item in parse_array(value["events"], "events"))
        if dump_log(log) != content:
            raise AcceptanceError("Log is not the canonical encoding of distinct recorded events")
        return log
    except (ValueError, RecursionError) as error:
        raise AcceptanceError(f"Cannot load acceptance log: {error}") from error
