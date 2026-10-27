"""Record pause, resume, cancel, annotate, and steer decisions in a campaign's steering log.

The log is append-only, and a campaign's control status is a fold over it:

    running --pause--> paused --resume--> running
    running | paused --cancel--> cancelled (terminal)
    running | paused --steer--> unchanged
    running | paused | cancelled --annotate--> unchanged

Pausing a paused campaign, resuming a running one, or cancelling a cancelled one is recorded and
changes nothing. A steer records the decision that links a campaign revision to the child that
changes its methodology or scope, so earlier results stay with the revision that produced them.
The log records intent; a later runner enforces it.

An event may declare the restrictions of its text. declared() returns only events that did, so a
runner that builds prompts from it never sends undeclared text. A declaration is stored under a
"restrictions" key that an undeclared event omits, so every log written without one keeps its
bytes and its schema version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from datetime import datetime
from itertools import pairwise
from typing import Literal, get_args
from uuid import UUID

from quant_research.access import Restrictions
from quant_research.campaigns import Campaign, CampaignRevision, unique
from quant_research.providers import Material, dump_material, load_material
from quant_research.records import (
    canonical_bytes,
    json_default,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
    parse_uuid,
)

type Command = Literal["pause", "resume", "cancel", "annotate"]
type Status = Literal["running", "paused", "cancelled"]

_COMMANDS = get_args(Command.__value__)
_STATUSES = get_args(Status.__value__)
REVISION_BYTES = 65536


class SteeringError(ValueError):
    """Report a refused steering event or an invalid stored steering log."""


def check_revision_size(content: bytes) -> None:
    """Refuse steer revision text above the per-update UTF-8 byte limit."""
    if len(content) > REVISION_BYTES:
        raise SteeringError(f"A steer revision holds at most {REVISION_BYTES} UTF-8 bytes")


@dataclass(frozen=True)
class Control:
    """Record a pause, resume, cancel, or annotation, whose reason is the note itself.

    restrictions declares the restrictions of the reason's text, and None leaves it undeclared.
    """

    action: Command
    actor: str
    at: datetime
    reason: str
    restrictions: Restrictions | None = None


@dataclass(frozen=True)
class Steer:
    """Record a methodology or scope decision from a revision to the child that makes it.

    restrictions declares the restrictions of the reason's text, as for Control.
    """

    actor: str
    at: datetime
    reason: str
    parent: str
    revision: str
    action: Literal["steer"] = "steer"
    restrictions: Restrictions | None = None


type Event = Control | Steer

_TRANSITIONS: dict[tuple[Status, str], Status] = {
    ("running", "pause"): "paused",
    ("paused", "pause"): "paused",
    ("running", "resume"): "running",
    ("paused", "resume"): "running",
    **{(status, "cancel"): "cancelled" for status in _STATUSES},
    **{(status, "annotate"): status for status in _STATUSES},
    ("running", "steer"): "running",
    ("paused", "steer"): "paused",
}


def _fold(events: tuple[Event, ...]) -> Status:
    status: Status = "running"
    for event in events:
        following = _TRANSITIONS.get((status, event.action))
        if following is None:
            raise SteeringError(f"Cannot {event.action} a {status} campaign")
        status = following
    return status


@dataclass(frozen=True)
class SteeringLog:
    """Hold one campaign's steering events, oldest first."""

    campaign_id: UUID
    events: tuple[Event, ...] = ()

    @property
    def status(self) -> Status:
        """Return the control status folded from the events."""
        return _fold(self.events)

    def record(
        self,
        action: Command,
        *,
        actor: str,
        at: datetime,
        reason: str,
        restrictions: Restrictions | None = None,
    ) -> SteeringLog:
        """Return the log with a pause, resume, cancel, or annotation appended.

        Args:
            action (Command): Command to record.
            actor (str): Person who issues the command.
            at (datetime): Timezone-aware time, no earlier than the last event.
            reason (str): Why the command is issued, or the note itself for an annotation.
            restrictions (Restrictions | None, optional): Declared restrictions of reason.

        Returns:
            (SteeringLog) The extended log, or this log if an identical event is already recorded.
        """
        try:
            if action not in _COMMANDS:
                raise SteeringError(
                    f"Unsupported command {action!r}; record a steer with steer(revision)"
                )
            return self._append(Control(action, actor, at, reason, restrictions))
        except ValueError as error:
            raise SteeringError(f"Cannot record {action!r}: {error}") from error

    def steer(
        self, revision: CampaignRevision, *, at: datetime, restrictions: Restrictions | None = None
    ) -> SteeringLog:
        """Return the log with the decision that a child revision records appended.

        Create the child with campaigns.revise, on its parent's branch or a new one. The steer
        takes its actor and reason from the child and is dated when the log records it.

        Args:
            revision (CampaignRevision): Child revision of this campaign that changes at least
                one specification section of its parent and that the log has not steered to.
            at (datetime): Timezone-aware time, no earlier than the child's creation or the
                last event.
            restrictions (Restrictions | None, optional): Declared restrictions of the child's
                reason.

        Returns:
            (SteeringLog) The extended log, or this log if an identical steer is already recorded.
        """
        try:
            if at.utcoffset() is None:
                raise SteeringError(f"The steer time {at.isoformat()} has no timezone")
            return self._append(_steer(revision, self.campaign_id, at, restrictions))
        except ValueError as error:
            raise SteeringError(f"Cannot steer campaign {self.campaign_id}: {error}") from error

    def declared(self, after: int = 0) -> tuple[tuple[Event, Restrictions], ...]:
        """Return the events from index after on that declared their text's restrictions.

        Args:
            after (int, optional): Index of the first event to consider.

        Returns:
            (tuple[tuple[Event, Restrictions], ...]) Each declared event with its declaration, in
                log order.
        """
        return tuple(
            (event, event.restrictions)
            for event in self.events[after:]
            if event.restrictions is not None
        )

    def _append(self, event: Event) -> SteeringLog:
        # Check before parsing, which rejects the duplicate that an exact retry would add.
        if event in self.events:
            return self
        return _parse(log_bytes(SteeringLog(self.campaign_id, (*self.events, event))))


def _steer(
    revision: CampaignRevision, campaign_id: UUID, at: datetime, restrictions: Restrictions | None
) -> Steer:
    revision_id = revision.revision_id
    if revision.campaign_id != campaign_id:
        raise SteeringError(
            f"Revision {revision_id} belongs to campaign {revision.campaign_id}, not {campaign_id}"
        )
    if revision.parent is None:
        raise SteeringError(f"Revision {revision_id} is the root; a steer revises a revision")
    if not revision.changes:
        raise SteeringError(
            f"Revision {revision_id} changes nothing; a steer must change methodology or scope; "
            "record an annotation instead"
        )
    if at < revision.created_at:
        raise SteeringError(
            f"The steer at {at.isoformat()} is dated before revision {revision_id}, created at "
            f"{revision.created_at.isoformat()}"
        )
    return Steer(
        revision.author,
        at,
        revision.reason,
        revision.parent,
        revision_id,
        restrictions=restrictions,
    )


def _event(raw: object) -> Event:
    # The declaration is the one optional key, so an undeclared event keeps its original bytes.
    declared = " restrictions" if isinstance(raw, dict) and "restrictions" in raw else ""
    if isinstance(raw, dict) and raw.get("action") == "steer":
        value = parse_fields(
            raw, f"action actor at reason parent revision{declared}", "steer event"
        )
        return Steer(
            parse_text(value["actor"], "actor"),
            parse_time(value["at"]),
            parse_text(value["reason"], "reason"),
            parse_digest(value["parent"]),
            parse_digest(value["revision"]),
            restrictions=_declaration(value["restrictions"]) if declared else None,
        )
    value = parse_fields(raw, f"action actor at reason{declared}", "event")
    if value["action"] not in _COMMANDS:
        raise SteeringError(f"Unsupported action {value['action']!r}")
    return Control(
        value["action"],
        parse_text(value["actor"], "actor"),
        parse_time(value["at"]),
        parse_text(value["reason"], "reason"),
        _declaration(value["restrictions"]) if declared else None,
    )


def dump_declaration(restrictions: Restrictions) -> str:
    """Return declared restrictions as JSON text, in the form a steering log stores them.

    The material codec of providers is the public encoding of restrictions, so a declaration
    rides it instead of a private copy.

    Args:
        restrictions (Restrictions): Restrictions to encode.

    Returns:
        (str) Sorted-key JSON of available_at as UTC text and labels and approved_for as sorted
            lists.
    """
    return json.dumps(_encoded(restrictions), sort_keys=True)


def load_declaration(text: str) -> Restrictions:
    """Return the restrictions that dump_declaration wrote, refusing any other form.

    Args:
        text (str): Text that dump_declaration wrote.

    Returns:
        (Restrictions) The declared restrictions.
    """
    try:
        return _declaration(json.loads(text))
    except ValueError as error:
        raise SteeringError(f"Invalid declaration: {error}") from error


def _encoded(restrictions: Restrictions) -> object:
    return json.loads(dump_material(Material("", restrictions)))["restrictions"]


def _declaration(raw: object) -> Restrictions:
    restrictions = load_material(json.dumps({"text": "", "restrictions": raw})).restrictions
    # One declaration has one stored form, so unsorted or repeated names are refused.
    if _encoded(restrictions) != raw:
        raise SteeringError("Declared restrictions are not in canonical form")
    return restrictions


def _parse(content: bytes) -> SteeringLog:
    value = parse_fields(
        json.loads(content.decode("utf-8")), "schema_version campaign_id events", "steering log"
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise SteeringError(f"Unsupported schema version {value['schema_version']!r}")
    if canonical_bytes(value) != content:
        raise SteeringError("Steering log is not canonical UTF-8 JSON")
    events = tuple(_event(item) for item in parse_array(value["events"], "events"))
    for earlier, later in pairwise(events):
        if later.at < earlier.at:
            raise SteeringError(
                f"The {later.action} at {later.at.isoformat()} is dated before the "
                f"{earlier.action} at {earlier.at.isoformat()}"
            )
    unique(events, "event")
    unique((event.revision for event in events if isinstance(event, Steer)), "steer to revision")
    _fold(events)
    return SteeringLog(parse_uuid(value["campaign_id"]), events)


def log_bytes(log: SteeringLog) -> bytes:
    """Return the canonical stored form of a steering log.

    Args:
        log (SteeringLog): Log to serialize.

    Returns:
        (bytes) Sorted-key UTF-8 JSON with the events in log order and a trailing newline. An
            event holds restrictions only when it declared them.
    """
    return canonical_bytes(
        {
            "schema_version": 1,
            "campaign_id": str(log.campaign_id),
            "events": [_stored(event) for event in log.events],
        }
    )


def _stored(event: Event) -> dict[str, object]:
    # asdict would recurse into the declaration, whose frozensets json_default refuses.
    value = {
        item.name: getattr(event, item.name)
        for item in fields(event)
        if item.name != "restrictions"
    }
    value = json.loads(json.dumps(value, default=json_default))
    if event.restrictions is not None:
        value["restrictions"] = _encoded(event.restrictions)
    return value


def read_log(content: bytes, campaign: Campaign) -> SteeringLog:
    """Verify and open a stored steering log against its campaign's revision history.

    Args:
        content (bytes): Stored log, as log_bytes writes it.
        campaign (Campaign): Opened campaign that holds every revision the log steers to.

    Returns:
        (SteeringLog) The log, each steer equal to the one its named revision derives at the
            steer's time.
    """
    try:
        log = _parse(content)
        if log.campaign_id != campaign.campaign_id:
            raise SteeringError(
                f"Log belongs to campaign {log.campaign_id}, not {campaign.campaign_id}"
            )
        revisions = {revision.revision_id: revision for revision in campaign.revisions}
        for event in log.events:
            if not isinstance(event, Steer):
                continue
            if event.revision not in revisions:
                raise SteeringError(
                    f"Steer names revision {event.revision}, which the campaign does not hold"
                )
            rebuilt = _steer(
                revisions[event.revision], log.campaign_id, event.at, event.restrictions
            )
            if rebuilt != event:
                raise SteeringError(
                    f"Steer to revision {event.revision} disagrees with that revision's author, "
                    "reason, or parent"
                )
        return log
    except (ValueError, RecursionError) as error:
        raise SteeringError(f"Cannot read steering log: {error}") from error
