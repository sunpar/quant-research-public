"""Hold restricted confirmation reports for authorized review, and audit every decision.

A review store is a directory used for nothing else:

    <store>/reports/<report id>.json   write-once canonical report, named by its exposure id
    <store>/log/<index>.json           write-once canonical log entry, from 0.json without gaps

Each log entry holds one event and the SHA-256 of the entry before it. The log is the only
authority. Deposits, grants, revocations, opens, and refusals are its events, and every grant
state and decision is a fold over it. Reading the log verifies the chain and refolds every
event, so an edited, removed, or reordered entry raises, and so does an open or refusal the
entries before it do not allow. Rewriting the newest entries, or any entry together with all
entries after it, leaves a valid chain, which only a head kept outside the store reveals.

Stored reports hold the outcomes, and no event, error, or exposure does. Opening a report
returns the exposure to append to the lineage history. Nothing here issues or widens a
RunContext, so D025 still denies confirmation material to every run, the reviewer's included.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Literal, NewType, get_args
from uuid import UUID, uuid4

from quant_research.confirmation import Period, RestrictedConfirmationReport
from quant_research.packages import ExposureEvent, parse_exposure, parse_outcome
from quant_research.records import (
    append_log,
    canonical_bytes,
    canonical_json,
    format_time,
    json_default,
    parse_array,
    parse_digest,
    parse_fields,
    parse_text,
    parse_time,
    parse_uuid,
    read_log,
    write_once,
)

# The confirmation exposure's id, already in the lineage. A digest of the stored report would
# depend only on the restricted outcomes and would reach the log and every review exposure.
ReportId = NewType("ReportId", UUID)
GrantId = NewType("GrantId", UUID)

type Purpose = Literal["human_review", "audit"]
type Denial = Literal["no_grant", "wrong_purpose", "expired", "revoked"]

_PURPOSES = get_args(Purpose.__value__)
_DENIALS = get_args(Denial.__value__)


class ReportStoreError(ValueError):
    """Report invalid input, a conflicting deposit, or a damaged or tampered review store."""


class OpenRefused(ReportStoreError):
    """Report an open that no active grant allows, raised after its Refused event is recorded."""

    def __init__(self, reasons: tuple[Denial, ...]) -> None:
        super().__init__(f"Open refused: {', '.join(reasons)}")
        self.reasons = reasons


@dataclass(frozen=True)
class ForReport:
    report: ReportId


@dataclass(frozen=True)
class ForCampaign:
    """Cover every report deposited under the campaign, including reports deposited later."""

    campaign: UUID


type Scope = ForReport | ForCampaign


@dataclass(frozen=True, kw_only=True)
class _Stamped:
    id: UUID
    actor: str
    at: datetime


@dataclass(frozen=True, kw_only=True)
class Deposited(_Stamped):
    """Bind a stored report to its campaign; digest is the SHA-256 of the stored report."""

    report: ReportId
    campaign: UUID
    digest: str
    shortlist: str
    period: Period


@dataclass(frozen=True, kw_only=True)
class Granted(_Stamped):
    """Let reviewer open the scope for one purpose before expires_at; actor is the grantor."""

    scope: Scope
    reviewer: str
    purpose: Purpose
    expires_at: datetime


@dataclass(frozen=True, kw_only=True)
class Revoked(_Stamped):
    grant: GrantId
    reason: str


@dataclass(frozen=True, kw_only=True)
class Opened(_Stamped):
    """Record that actor, the reviewer, was given the report under the grant."""

    report: ReportId
    purpose: Purpose
    grant: GrantId


@dataclass(frozen=True, kw_only=True)
class Refused(_Stamped):
    """Record that actor, the reviewer, asked for the report and was refused."""

    report: ReportId
    purpose: Purpose
    reasons: tuple[Denial, ...]


type Event = Deposited | Granted | Revoked | Opened | Refused

_EVENTS: dict[str, type] = {
    event_type.__name__.lower(): event_type
    for event_type in (Deposited, Granted, Revoked, Opened, Refused)
}


@dataclass(frozen=True)
class Opening:
    """The exact stored report, and the exposure to append to the lineage history."""

    report: RestrictedConfirmationReport
    exposure: ExposureEvent


@dataclass(frozen=True)
class Audit:
    """The verified log, the exposure of each open in log order, and the newest entry's digest.

    Keep head outside the store. A later read_audit given it detects any edit or removal of
    the entries up to it.
    """

    events: tuple[Event, ...]
    exposures: tuple[ExposureEvent, ...]
    head: str | None


def deposit_report(
    store: Path,
    report: RestrictedConfirmationReport,
    *,
    campaign: UUID,
    actor: str,
    at: datetime,
) -> ReportId:
    """Store a restricted report once and bind it to its campaign.

    Args:
        store (Path): Existing directory used only as a review store.
        report (RestrictedConfirmationReport): Report that run_confirmation returned.
        campaign (UUID): Campaign that owns the report, which carries none itself.
        actor (str): Declared principal who deposits it.
        at (datetime): Timezone-aware time, no earlier than the newest event.

    Returns:
        (ReportId) The report's confirmation exposure id, equal on every repeat.

    Raises:
        ReportStoreError: If the report does not survive canonical storage unchanged, it is
            deposited under another campaign or with other bytes, an input is invalid, or the
            store is damaged.
    """
    if not isinstance(report, RestrictedConfirmationReport):
        raise ReportStoreError("Only a restricted confirmation report can be deposited")
    report_id = ReportId(report.exposure.id)
    content = _report_bytes(report)
    digest = hashlib.sha256(content).hexdigest()

    def decide(state: _State) -> Event:
        if (known := state.deposits.get(report_id)) is not None:
            if (known.campaign, known.digest) != (campaign, digest):
                raise ReportStoreError(
                    f"Report {report_id} is deposited under another campaign or with other bytes"
                )
            return known
        return Deposited(
            id=uuid4(),
            actor=actor,
            at=at,
            report=report_id,
            campaign=campaign,
            digest=digest,
            shortlist=report.shortlist,
            period=report.period,
        )

    def store_report() -> None:
        # A crash before the event's entry, or a lost race whose retry is refused, leaves a
        # report that no event names. Nothing can open it, and a valid repeat deposit completes it.
        (store / "reports").mkdir(exist_ok=True)
        if not write_once(store / "reports" / f"{report_id}.json", content):
            raise ReportStoreError(f"Report {report_id} is stored with other bytes")

    _commit(store, at, decide, store_report)
    return report_id


def grant_review(
    store: Path,
    *,
    scope: Scope,
    reviewer: str,
    purpose: Purpose,
    expires_at: datetime,
    actor: str,
    at: datetime,
) -> GrantId:
    """Let one reviewer open the scope's reports for one purpose until a deadline.

    Args:
        store (Path): Existing review store.
        scope (Scope): ForReport naming a deposited report, or ForCampaign.
        reviewer (str): Declared principal who may open.
        purpose (Purpose): The one purpose the grant allows.
        expires_at (datetime): First instant the grant no longer allows an open, after at.
        actor (str): Declared principal who grants it.
        at (datetime): Timezone-aware time, no earlier than the newest event.

    Returns:
        (GrantId) The new grant, or the identical grant that is already active.

    Raises:
        ReportStoreError: If a report scope names no deposited report, expires_at is not after
            at, an input is invalid, or the store is damaged.
    """
    request = (scope, reviewer, purpose, expires_at)

    def decide(state: _State) -> Event:
        for grant, granted in state.grants.items():
            same = (granted.scope, granted.reviewer, granted.purpose, granted.expires_at)
            if same == request and state.active(grant, at):
                return granted
        return Granted(
            id=uuid4(),
            actor=actor,
            at=at,
            scope=scope,
            reviewer=reviewer,
            purpose=purpose,
            expires_at=expires_at,
        )

    event, _ = _commit(store, at, decide)
    return GrantId(event.id)


def revoke_review(store: Path, grant: GrantId, *, actor: str, at: datetime, reason: str) -> None:
    """End a grant. Revoking a revoked grant changes nothing.

    Args:
        store (Path): Existing review store.
        grant (GrantId): Recorded grant to end.
        actor (str): Declared principal who revokes it.
        at (datetime): Timezone-aware time, no earlier than the newest event.
        reason (str): Why the grant ends.

    Raises:
        ReportStoreError: If the grant is unknown, an input is invalid, or the store is damaged.
    """
    _commit(
        store,
        at,
        lambda state: state.revocations.get(
            grant, Revoked(id=uuid4(), actor=actor, at=at, grant=grant, reason=reason)
        ),
    )


def open_report(
    store: Path, report: ReportId, *, reviewer: str, purpose: Purpose, at: datetime
) -> Opening:
    """Give an authorized reviewer the stored report, recording the open or the refusal.

    The decision and its event come from one verified log position. A refusal is recorded,
    then raised. An authorized open verifies the stored report before it records the open.

    Args:
        store (Path): Existing review store.
        report (ReportId): Report to open.
        reviewer (str): Declared principal who asks to read it.
        purpose (Purpose): Why they read it; an active grant must allow this purpose.
        at (datetime): Timezone-aware time of the open, no earlier than the newest event.

    Returns:
        (Opening) The report equal to the deposited one, and the reviewer's result exposure.

    Raises:
        OpenRefused: If no active grant covers the reviewer, report, and purpose. Its reasons
            equal the recorded Refused event's.
        ReportStoreError: If an input is invalid, the log is damaged or tampered, or the stored
            report fails verification. None of these records anything.
    """
    opened: RestrictedConfirmationReport | None = None

    def decide(state: _State) -> Event:
        nonlocal opened
        grant, denials = _judge(state, report, reviewer, purpose, at)
        if denials:
            return Refused(
                id=uuid4(), actor=reviewer, at=at, report=report, purpose=purpose, reasons=denials
            )
        opened = _load_report(store, state.deposits[report])
        return Opened(
            id=uuid4(), actor=reviewer, at=at, report=report, purpose=purpose, grant=grant
        )

    event, state = _commit(store, at, decide)
    if isinstance(event, Refused):
        raise OpenRefused(event.reasons)
    return Opening(opened, _exposure(event, state.deposits[report]))


def read_audit(store: Path, anchor: str | None = None) -> Audit:
    """Verify the whole log and return its events and the exposures of its opens.

    Args:
        store (Path): Existing review store.
        anchor (str | None): Head of an earlier Audit, kept outside the store. Without one, an
            edit or removal of the newest entries goes unnoticed.

    Returns:
        (Audit) The verified log. No stored report is read.

    Raises:
        ReportStoreError: If an entry is missing, extra, not canonical, not chained to the entry
            before it, or records an event the entries before it do not allow, or if no entry
            has the anchor's digest.
    """
    try:
        events, state, digests = _read_log(store)
    except (OSError, ValueError, RecursionError) as error:
        raise ReportStoreError(f"Cannot read review store {store}: {error}") from error
    if anchor is not None and anchor not in digests:
        raise ReportStoreError(f"Review log no longer holds entry {anchor}")
    return Audit(
        events,
        tuple(
            _exposure(event, state.deposits[event.report])
            for event in events
            if isinstance(event, Opened)
        ),
        digests[-1] if digests else None,
    )


@dataclass
class _State:
    """Fold of the log. Reads and writes apply the same rules, so a written entry always reads."""

    deposits: dict[ReportId, Deposited] = field(default_factory=dict)
    grants: dict[GrantId, Granted] = field(default_factory=dict)
    revocations: dict[GrantId, Revoked] = field(default_factory=dict)
    ids: set[UUID] = field(default_factory=set)
    newest: datetime | None = None

    def active(self, grant: GrantId, at: datetime) -> bool:
        return grant not in self.revocations and at < self.grants[grant].expires_at

    def apply(self, event: Event) -> None:
        if event.id in self.ids:
            raise ReportStoreError(f"Event {event.id} is recorded twice")
        if self.newest is not None and event.at < self.newest:
            raise ReportStoreError(f"Event {event.id} is dated before the event it follows")
        match event:
            case Deposited():
                if event.report in self.deposits:
                    raise ReportStoreError(f"Report {event.report} is already deposited")
                self.deposits[event.report] = event
            case Granted():
                if event.expires_at <= event.at:
                    raise ReportStoreError("A grant must expire after it is issued")
                if isinstance(event.scope, ForReport) and event.scope.report not in self.deposits:
                    raise ReportStoreError(f"Report {event.scope.report} is not deposited")
                self.grants[GrantId(event.id)] = event
            case Revoked():
                if event.grant not in self.grants or event.grant in self.revocations:
                    raise ReportStoreError(f"Grant {event.grant} is unknown or already revoked")
                self.revocations[event.grant] = event
            case Opened():
                decision = _judge(self, event.report, event.actor, event.purpose, event.at)
                if decision != (event.grant, ()):
                    raise ReportStoreError(f"Open {event.id} has no active grant {event.grant}")
            case Refused():
                decision = _judge(self, event.report, event.actor, event.purpose, event.at)
                if decision != (None, event.reasons):
                    raise ReportStoreError(f"Refusal {event.id} misstates its denials")
        self.ids.add(event.id)
        self.newest = event.at


def _judge(
    state: _State, report: ReportId, reviewer: str, purpose: Purpose, at: datetime
) -> tuple[GrantId | None, tuple[Denial, ...]]:
    """Return the first active grant that allows the open, or every reason none does."""
    deposit = state.deposits.get(report)
    scopes = () if deposit is None else (ForReport(report), ForCampaign(deposit.campaign))
    covering = [
        grant
        for grant, granted in state.grants.items()
        if granted.reviewer == reviewer and granted.scope in scopes
    ]
    if not covering:
        return None, ("no_grant",)
    matching = [grant for grant in covering if state.grants[grant].purpose == purpose]
    if not matching:
        return None, ("wrong_purpose",)
    for grant in matching:
        if state.active(grant, at):
            return grant, ()
    return None, tuple(
        sorted({"revoked" if grant in state.revocations else "expired" for grant in matching})
    )


def _commit(
    store: Path,
    at: datetime,
    decide: Callable[[_State], Event],
    prepare: Callable[[], None] = lambda: None,
) -> tuple[Event, _State]:
    """Append the event decide makes from the verified log, unless the log already holds it.

    prepare runs once the new entry is valid and before it is written, so an invalid event
    leaves no file behind. A writer that loses the next entry's number to another rereads the
    longer log and decides again, so every decision holds at the position where it is recorded.
    """

    def attempt() -> tuple[int, tuple[Event, _State], bytes | None]:
        events, state, digests = _read_log(store)
        if state.newest is not None and at < state.newest:
            raise ReportStoreError("A request cannot be dated before the newest event")
        event = decide(state)
        if event.id in state.ids:
            return len(events), (event, state), None
        data = _data(event)
        if _parse_event(data) != event:
            raise ReportStoreError("Event does not survive canonical storage unchanged")
        state.apply(event)
        entry = {"schema_version": 1, "previous": digests[-1] if digests else None}
        content = canonical_bytes({**entry, "event": data})
        prepare()
        return len(events), (event, state), content

    try:
        format_time(at)
        (store / "log").mkdir(exist_ok=True)
        return append_log(store / "log", attempt)
    except (OSError, ValueError, RecursionError) as error:
        raise ReportStoreError(f"Cannot update review store {store}: {error}") from error


def _read_log(store: Path) -> tuple[tuple[Event, ...], _State, tuple[str, ...]]:
    """Verify each entry's form and link and fold its event; return events, state, digests."""
    directory = store / "log"
    values = read_log(directory, "Log entry", "Log entries") if directory.exists() else ()
    state, events, digests = _State(), [], []
    for index, value in enumerate(values):
        entry = _versioned(value, "previous event", "log entry")
        if entry["previous"] != (digests[-1] if digests else None):
            raise ReportStoreError(f"Log entry {index} does not follow the entry before it")
        event = _parse_event(entry["event"])
        state.apply(event)
        events.append(event)
        # read_log yields only values whose canonical bytes are the stored bytes.
        digests.append(hashlib.sha256(canonical_bytes(value)).hexdigest())
    return tuple(events), state, tuple(digests)


def _exposure(opened: Opened, deposit: Deposited) -> ExposureEvent:
    # save_package needs each exposure's material among the package's artifacts. The frozen
    # shortlist manifest can be one, and the report never can.
    return ExposureEvent(
        opened.id,
        opened.actor,
        opened.at,
        deposit.shortlist,
        deposit.period.dataset_id,
        deposit.period.start,
        deposit.period.end,
        "result",
        f"Opened restricted confirmation report {opened.report} for {opened.purpose}",
    )


def _versioned(raw: object, names: str, label: str) -> dict:
    value = parse_fields(raw, f"schema_version {names}", label)
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ReportStoreError(f"Unsupported {label} schema version {value['schema_version']!r}")
    return value


def _choice(value: object, choices: tuple[str, ...], label: str) -> str:
    if value not in choices:
        raise ReportStoreError(f"Unknown {label} {value!r}")
    return value


def _period(raw: object) -> Period:
    value = parse_fields(raw, "dataset_id start end", "period")
    return Period(value["dataset_id"], parse_time(value["start"]), parse_time(value["end"]))


def _scope(raw: object) -> Scope:
    if isinstance(raw, dict) and raw.keys() == {"report"}:
        return ForReport(ReportId(parse_uuid(raw["report"])))
    return ForCampaign(parse_uuid(parse_fields(raw, "campaign", "scope")["campaign"]))


# Each field name means one thing in every event, so one parser per name covers all kinds.
_FIELDS: dict[str, Callable[[object], object]] = {
    "id": parse_uuid,
    "actor": lambda value: parse_text(value, "actor"),
    "at": parse_time,
    "report": parse_uuid,
    "campaign": parse_uuid,
    "digest": parse_digest,
    "shortlist": parse_digest,
    "period": _period,
    "scope": _scope,
    "reviewer": lambda value: parse_text(value, "reviewer"),
    "purpose": lambda value: _choice(value, _PURPOSES, "purpose"),
    "expires_at": parse_time,
    "grant": parse_uuid,
    "reason": lambda value: parse_text(value, "revocation reason"),
    "reasons": lambda value: tuple(
        _choice(item, _DENIALS, "denial") for item in parse_array(value, "reasons")
    ),
}


def _data(event: Event) -> dict:
    return json.loads(
        json.dumps(
            {"kind": type(event).__name__.lower(), **asdict(event)},
            default=json_default,
            allow_nan=False,
        )
    )


def _parse_event(raw: object) -> Event:
    kind = raw.get("kind") if isinstance(raw, dict) else None
    if not isinstance(kind, str) or kind not in _EVENTS:
        raise ReportStoreError(f"Unknown event kind {kind!r}")
    names = [item.name for item in fields(_EVENTS[kind])]
    value = parse_fields(raw, " ".join(["kind", *names]), f"{kind} event")
    return _EVENTS[kind](**{name: _FIELDS[name](value[name]) for name in names})


def _report_bytes(report: RestrictedConfirmationReport) -> bytes:
    # Encoding and parse errors quote the offending value, which can be a restricted outcome.
    try:
        data = json.loads(json.dumps(asdict(report), default=json_default, allow_nan=False))
        content = canonical_bytes({"schema_version": 1, **data})
        survives = _parse_report(content) == report
    except (TypeError, ValueError, RecursionError):
        survives = False
    if not survives:
        raise ReportStoreError("Report does not survive canonical storage unchanged")
    return content


def _parse_report(content: bytes) -> RestrictedConfirmationReport:
    value = _versioned(
        canonical_json(content, "Stored report"),
        "shortlist period exposure results",
        "report",
    )
    results = [parse_array(item, "result") for item in parse_array(value["results"], "results")]
    return RestrictedConfirmationReport(
        parse_digest(value["shortlist"]),
        _period(value["period"]),
        parse_exposure(value["exposure"]),
        tuple((parse_digest(candidate), parse_outcome(outcome)) for candidate, outcome in results),
    )


def _load_report(store: Path, deposit: Deposited) -> RestrictedConfirmationReport:
    content = (store / "reports" / f"{deposit.report}.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != deposit.digest:
        raise ReportStoreError(f"Report {deposit.report} changed after deposit")
    report = _parse_report(content)
    if report.exposure.id != deposit.report:
        raise ReportStoreError(f"Report {deposit.report} holds another confirmation")
    return report
