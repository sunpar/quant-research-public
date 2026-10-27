from __future__ import annotations

import json
import operator
from collections import ChainMap, Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, astuple, dataclass, field, fields
from decimal import MAX_PREC, ROUND_HALF_EVEN, Context, Decimal, Inexact, localcontext
from typing import Literal, NamedTuple, get_args
from uuid import UUID

from quant_research.packages import Status
from quant_research.records import (
    canonical_bytes,
    canonical_json,
    parse_array,
    parse_fields,
    parse_uuid,
)

_STATUSES = get_args(Status.__value__)
# Context() fills any omitted field from decimal.DefaultContext, which a caller may change.
EXACT = Context(
    prec=MAX_PREC,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[Inexact],
)
# The fields a provider may leave unreported.
REPORTABLE = ("input_tokens", "output_tokens", "cost_usd")


class LedgerError(ValueError):
    """Report an invalid usage record, identity, or ledger transition."""


class InexactUsage(LedgerError):
    """Report usage whose sum the ledger cannot hold exactly."""


@dataclass(frozen=True)
class Usage:
    """Resources one charge used, or one estimate holds.

    unreported names the fields of REPORTABLE that a provider left unreported. Each such field
    holds the reported lower bound, zero when nothing was reported, so sums and limit checks
    count the known part.
    """

    compute_seconds: Decimal = Decimal(0)
    wall_seconds: Decimal = Decimal(0)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal(0)
    unreported: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for item in _AMOUNTS:
            _amount(getattr(self, item.name), type(item.default), item.name)
        if not isinstance(self.unreported, frozenset) or not self.unreported <= set(REPORTABLE):
            raise LedgerError(f"unreported must be a frozenset of names in {REPORTABLE}")

    def __add__(self, other: Usage) -> Usage:
        return _componentwise(operator.add, self, other)


_AMOUNTS = tuple(item for item in fields(Usage) if item.name != "unreported")


@dataclass(frozen=True)
class Budget:
    """Declare campaign limits; None leaves a limit undeclared and unlimited."""

    compute_seconds: Decimal | None = None
    wall_seconds: Decimal | None = None
    model_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: Decimal | None = None
    failures_per_run: int | None = None

    def __post_init__(self) -> None:
        units = {item.name: type(item.default) for item in _AMOUNTS}
        units["failures_per_run"] = int
        for name, cap in asdict(self).items():
            if cap is not None:
                _amount(cap, units[name], name)


def _amount(value: object, unit: type, label: str) -> None:
    if type(value) is not unit or (unit is Decimal and not value.is_finite()) or value < 0:
        raise LedgerError(f"{label} must be a finite nonnegative {unit.__name__}")
    # Wider exponents would make an exact sum allocate digits without bound.
    if (
        unit is Decimal
        and not EXACT.Emin <= value.as_tuple().exponent <= value.adjusted() <= EXACT.Emax
    ):
        raise LedgerError(f"{label} digits must lie between 1E{EXACT.Emin} and 1E+{EXACT.Emax}")


def _componentwise(
    operation: Callable[[object, object], object], left: Usage, right: Usage
) -> Usage:
    with localcontext(EXACT):
        try:
            return Usage(
                **{
                    item.name: operation(getattr(left, item.name), getattr(right, item.name))
                    for item in _AMOUNTS
                },
                unreported=left.unreported | right.unreported,
            )
        except Inexact as error:
            raise InexactUsage("Usage arithmetic must stay exact") from error


@dataclass(frozen=True)
class Work:
    campaign_id: UUID
    trial_id: UUID
    run_id: UUID
    attempt_id: UUID

    def __post_init__(self) -> None:
        if any(identifier.int == 0 for identifier in astuple(self)):
            raise LedgerError("Work IDs must not be the nil UUID")


@dataclass(frozen=True)
class Reserved:
    work: Work
    estimate: Usage


@dataclass(frozen=True)
class Charged:
    id: UUID
    attempt_id: UUID
    role: str
    usage: Usage

    def __post_init__(self) -> None:
        if self.id.int == 0:
            raise LedgerError("Charge ID must not be the nil UUID")
        if not isinstance(self.role, str) or not self.role.strip() or "\x00" in self.role:
            raise LedgerError("Charge role must be nonempty text")


@dataclass(frozen=True)
class Released:
    attempt_id: UUID
    outcome: Status

    def __post_init__(self) -> None:
        if self.outcome not in _STATUSES:
            raise LedgerError("Unsupported outcome")


type Event = Reserved | Charged | Released


@dataclass(frozen=True)
class Refusal:
    limit: str
    committed: Decimal | int
    requested: Decimal | int
    cap: Decimal | int


@dataclass(frozen=True)
class Decision:
    refusals: tuple[Refusal, ...]

    @property
    def admitted(self) -> bool:
        return not self.refusals


@dataclass(frozen=True)
class CostLine:
    """Spend by campaign, role, and outcome.

    unreported counts, per field, the charges that left it unreported, so usage holds a lower
    bound of each field it names.
    """

    campaign_id: UUID
    role: str
    outcome: Status | Literal["open"]
    usage: Usage
    unreported: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Stall:
    campaign_id: UUID
    trial_id: UUID
    run_id: UUID
    failures: int


class _State(NamedTuple):
    identities: ChainMap[UUID, tuple[str, UUID | None]]
    reservations: ChainMap[UUID, Reserved]
    charges: ChainMap[UUID, Charged]
    outcomes: dict[UUID, Status]
    # By campaign: the charges of released attempts, and the charges of each open attempt.
    released: dict[UUID, Usage]
    open: dict[UUID, dict[UUID, Usage]]
    # By run: the open and failed attempts, which hold its failure slots.
    slots: Counter[UUID]


def _identities(work: Work) -> tuple[tuple[UUID, tuple[str, UUID | None]], ...]:
    return (
        (work.campaign_id, ("campaign", None)),
        (work.trial_id, ("trial", work.campaign_id)),
        (work.run_id, ("run", work.trial_id)),
        (work.attempt_id, ("attempt", work.run_id)),
    )


def _exposure(state: _State, campaign_id: UUID) -> Usage:
    released, spent = state.released[campaign_id], dict(state.open[campaign_id])
    # A Ledger method writes the reservation or charge it is checking into the fold's overlays.
    for attempt_id, reservation in state.reservations.maps[0].items():
        if reservation.work.campaign_id == campaign_id:
            spent[attempt_id] = Usage()
    for charge in state.charges.maps[0].values():
        if charge.attempt_id in spent:
            spent[charge.attempt_id] += charge.usage
        elif state.reservations[charge.attempt_id].work.campaign_id == campaign_id:
            released += charge.usage
    return sum(
        (
            _componentwise(max, state.reservations[attempt_id].estimate, usage)
            for attempt_id, usage in spent.items()
        ),
        released,
    )


class Ledger:
    """Admit attempts within declared campaign budgets and record their actual usage.

    Every method checks the fold of recorded events, appends at most one event, and
    raises LedgerError for an invalid or conflicting request.
    """

    def __init__(self, budgets: Mapping[UUID, Budget]) -> None:
        self._budgets = dict(budgets)
        self._events: list[Event] = []
        self._state = _State(
            ChainMap({campaign_id: ("campaign", None) for campaign_id in self._budgets}),
            ChainMap(),
            ChainMap(),
            {},
            dict.fromkeys(self._budgets, Usage()),
            {campaign_id: {} for campaign_id in self._budgets},
            Counter(),
        )
        self._folded = 0

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def reserve(self, work: Work, estimate: Usage) -> Decision:
        """Reserve an attempt's estimate if it fits every declared limit of its campaign.

        Args:
            work (Work): Campaign, trial, run, and attempt identities of the new work.
            estimate (Usage): Usage to hold for the attempt until its release.

        Returns:
            (Decision) Every exceeded limit; an admitted decision records the reservation.

        Notes:
            Released attempts hold their charges. Open attempts hold the larger of their
            estimate and charges in each resource, so an overrun never frees capacity.
            Failed and open attempts each hold one of their run's failure slots.
        """
        if work.campaign_id not in self._budgets:
            raise LedgerError(f"Unknown campaign {work.campaign_id}")
        event = Reserved(work, estimate)
        state = self._fold()
        for identifier, identity in _identities(work):
            if state.identities.setdefault(identifier, identity) != identity:
                raise LedgerError(f"ID {identifier} already names another kind or parent")
        if work.attempt_id in state.reservations:
            if state.reservations[work.attempt_id] != event:
                raise LedgerError(f"Attempt {work.attempt_id} is already reserved differently")
            return Decision(())
        exposure = _exposure(state, work.campaign_id)
        committed = asdict(exposure) | {"failures_per_run": state.slots[work.run_id]}
        requested = asdict(estimate) | {"failures_per_run": 1}
        # The headroom under a cap is always exact, while exposure plus estimate can overflow.
        with localcontext(EXACT):
            refusals = tuple(
                Refusal(name, committed[name], requested[name], cap)
                for name, cap in asdict(self._budgets[work.campaign_id]).items()
                if cap is not None and requested[name] > cap - committed[name]
            )
        if not refusals:
            state.reservations[work.attempt_id] = event
            # Undeclared limits refuse nothing, yet the campaign's next exposure must be exact.
            _exposure(state, work.campaign_id)
            self._events.append(event)
        return Decision(refusals)

    def charge(self, id: UUID, attempt_id: UUID, role: str, usage: Usage) -> None:
        """Record actual usage of a reserved attempt, including after its release.

        Args:
            id (UUID): Charge identity; repeating an identical charge records nothing.
            attempt_id (UUID): Reserved attempt that used the resources.
            role (str): Reporting role that incurred the usage.
            usage (Usage): Measured usage, recorded even beyond the campaign budget.
        """
        event = Charged(id, attempt_id, role, usage)
        state = self._fold()
        if attempt_id not in state.reservations:
            raise LedgerError(f"Attempt {attempt_id} has no reservation")
        if state.charges.get(id) == event:
            return
        if id in state.identities:
            raise LedgerError(f"Charge ID {id} is already recorded or names another kind")
        state.charges[id] = event
        # The campaign's exposure bounds every later sum for it, so it must be exact now.
        _exposure(state, state.reservations[attempt_id].work.campaign_id)
        self._events.append(event)

    def release(self, attempt_id: UUID, outcome: Status) -> None:
        """Record an attempt's outcome and return its unused reservation to the campaign.

        Args:
            attempt_id (UUID): Reserved attempt to release.
            outcome (Status): Development outcome of the attempt.
        """
        event = Released(attempt_id, outcome)
        state = self._fold()
        if attempt_id not in state.reservations:
            raise LedgerError(f"Attempt {attempt_id} has no reservation")
        prior = state.outcomes.get(attempt_id)
        if prior is None:
            self._events.append(event)
        elif prior != outcome:
            raise LedgerError(f"Attempt {attempt_id} was already released as {prior!r}")

    def report(self) -> tuple[CostLine, ...]:
        """Sum charges exactly by campaign, role, and released outcome or "open"."""
        state = self._fold()
        totals: defaultdict[tuple[UUID, str, str], Usage] = defaultdict(Usage)
        unreported: defaultdict[tuple[UUID, str, str], Counter[str]] = defaultdict(Counter)
        for charge in state.charges.values():
            campaign_id = state.reservations[charge.attempt_id].work.campaign_id
            key = campaign_id, charge.role, state.outcomes.get(charge.attempt_id, "open")
            totals[key] += charge.usage
            unreported[key].update(charge.usage.unreported)
        return tuple(
            CostLine(*key, usage, dict(unreported[key])) for key, usage in sorted(totals.items())
        )

    def stalls(self) -> tuple[Stall, ...]:
        """List runs whose failed attempts reached their campaign's failures_per_run."""
        state = self._fold()
        failures = Counter(
            (reservation.work.campaign_id, reservation.work.trial_id, reservation.work.run_id)
            for attempt_id, reservation in state.reservations.items()
            if state.outcomes.get(attempt_id) == "failed"
        )
        return tuple(
            Stall(campaign_id, trial_id, run_id, count)
            for (campaign_id, trial_id, run_id), count in sorted(failures.items())
            if (limit := self._budgets[campaign_id].failures_per_run) is not None and count >= limit
        )

    def _fold(self) -> _State:
        state = self._state
        for event in self._events[self._folded :]:
            match event:
                case Reserved(work=work):
                    state.identities.update(_identities(work))
                    state.reservations[work.attempt_id] = event
                    state.open[work.campaign_id][work.attempt_id] = Usage()
                    state.slots[work.run_id] += 1
                case Charged(id=charge_id, attempt_id=attempt_id, usage=usage):
                    state.identities[charge_id] = ("charge", attempt_id)
                    state.charges[charge_id] = event
                    campaign_id = state.reservations[attempt_id].work.campaign_id
                    if attempt_id in state.open[campaign_id]:
                        state.open[campaign_id][attempt_id] += usage
                    else:
                        state.released[campaign_id] += usage
                case Released(attempt_id=attempt_id, outcome=outcome):
                    state.outcomes[attempt_id] = outcome
                    work = state.reservations[attempt_id].work
                    campaign_id = work.campaign_id
                    if outcome != "failed":
                        state.slots[work.run_id] -= 1
                    state.released[campaign_id] += state.open[campaign_id].pop(attempt_id)
        self._folded = len(self._events)
        # Callers write identities, reservations, and charges before they may refuse or raise.
        return state._replace(
            identities=state.identities.new_child(),
            reservations=state.reservations.new_child(),
            charges=state.charges.new_child(),
        )


def json_default(value: object) -> object:
    """Encode an unreported set as a sorted list and any other usage or ledger value as its str.

    str() writes UUIDs canonically and Decimals exactly under EXACT, and a parser refuses any
    other value by its field without quoting it.
    """
    return sorted(value) if isinstance(value, frozenset) else str(value)


def dump_ledger(ledger: Ledger, *, event_count: int | None = None) -> bytes:
    """Return a ledger's budgets and events as canonical JSON that load_ledger restores.

    Args:
        ledger (Ledger): Ledger to export.
        event_count (int | None, optional): Immutable event prefix to export.

    Returns:
        (bytes) {"schema_version": 1, "budgets": {campaign: limits}, "events": [{kind: event}]}.
    """
    events = ledger.events
    if event_count is not None:
        if type(event_count) is not int or not 0 <= event_count <= len(events):
            raise LedgerError("Ledger event count must name an available prefix")
        events = events[:event_count]
    value = {
        "schema_version": 1,
        "budgets": {
            str(campaign_id): asdict(budget) for campaign_id, budget in ledger._budgets.items()
        },
        "events": [{_KINDS[type(event)]: asdict(event)} for event in events],
    }
    # EXACT pins the exponent letter that str() writes for a Decimal.
    with localcontext(EXACT):
        return canonical_bytes(json.loads(json.dumps(value, default=json_default)))


def load_ledger(content: bytes) -> Ledger:
    """Replay dumped events through a fresh Ledger's own methods.

    Args:
        content (bytes): Bytes that dump_ledger wrote.

    Returns:
        (Ledger) The ledger whose dump is exactly content.

    Raises:
        LedgerError: The bytes are not a dump, or an event is refused or repeats one.
    """
    try:
        value = parse_fields(
            canonical_json(content, "Ledger"), "schema_version budgets events", "ledger"
        )
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise LedgerError(f"Unsupported ledger schema version {value['schema_version']!r}")
        if not isinstance(value["budgets"], dict):
            raise LedgerError("budgets must be a JSON object")
        ledger = Ledger(
            {
                parse_uuid(campaign_id, "campaign UUID"): _budget(budget)
                for campaign_id, budget in value["budgets"].items()
            }
        )
        for index, raw in enumerate(parse_array(value["events"], "events")):
            _apply(ledger, _event(raw))
            if len(ledger.events) != index + 1:
                raise LedgerError(f"Ledger event {index} is refused or repeats an event")
    except (ValueError, RecursionError) as error:
        raise LedgerError(f"Cannot load ledger: {error}") from error
    if dump_ledger(ledger) != content:
        raise LedgerError("Ledger bytes are not the dump of their events")
    return ledger


def _apply(ledger: Ledger, event: Event) -> Decision | None:
    match event:
        case Reserved(work, estimate):
            return ledger.reserve(work, estimate)
        case Charged(charge_id, attempt_id, role, usage):
            return ledger.charge(charge_id, attempt_id, role, usage)
        case Released(attempt_id, outcome):
            return ledger.release(attempt_id, outcome)


_KINDS = {Reserved: "reserve", Charged: "charge", Released: "release"}


def _budget(raw: object) -> Budget:
    value = parse_fields(raw, " ".join(item.name for item in fields(Budget)), "budget")
    decimals = {item.name for item in _AMOUNTS if type(item.default) is Decimal}
    return Budget(
        **{
            name: _decimal(cap) if name in decimals and cap is not None else cap
            for name, cap in value.items()
        }
    )


def _event(raw: object) -> Event:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise LedgerError("A ledger event is a JSON object with one event")
    match raw:
        case {"reserve": value}:
            value = parse_fields(value, "work estimate", "reservation")
            work = parse_fields(value["work"], "campaign_id trial_id run_id attempt_id", "work")
            return Reserved(
                Work(**{name: parse_uuid(text, f"{name} UUID") for name, text in work.items()}),
                parse_usage(value["estimate"]),
            )
        case {"charge": value}:
            value = parse_fields(value, "id attempt_id role usage", "charge")
            return Charged(
                parse_uuid(value["id"], "charge id UUID"),
                parse_uuid(value["attempt_id"], "attempt_id UUID"),
                value["role"],
                parse_usage(value["usage"]),
            )
        case {"release": value}:
            value = parse_fields(value, "attempt_id outcome", "release")
            return Released(parse_uuid(value["attempt_id"], "attempt_id UUID"), value["outcome"])
    raise LedgerError(f"Unknown ledger event {sorted(raw)!r}")


def parse_usage(raw: object) -> Usage:
    """Return the usage stored as a plain JSON object."""
    value = parse_fields(raw, " ".join(item.name for item in fields(Usage)), "usage")
    names = parse_array(value["unreported"], "unreported")
    if any(type(name) is not str for name in names) or names != sorted(set(names)):
        raise LedgerError("unreported must list distinct field names in sorted order")
    return Usage(
        **{
            item.name: _decimal(value[item.name])
            if type(item.default) is Decimal
            else value[item.name]
            for item in _AMOUNTS
        },
        unreported=frozenset(names),
    )


def _decimal(value: object) -> Decimal:
    if isinstance(value, str):
        # The exact context leaves invalid text as NaN, which never round-trips.
        with localcontext(EXACT):
            amount = Decimal(value)
            if str(amount) == value:
                return amount
    raise LedgerError("Invalid decimal amount")
