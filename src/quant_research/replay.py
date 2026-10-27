from __future__ import annotations

import hashlib
import json
import re
import sys
import sysconfig
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from quant_research.packages import (
    ArtifactInput,
    EnvironmentRecord,
    ExecutionAttempt,
    ExposureEvent,
    LineageLink,
    Outcome,
    PackageRecord,
    Specification,
    TrialRecord,
    VerifiedPackage,
    open_package,
    save_package,
)
from quant_research.records import unique_keys

SIGNAL_PROFILE = "level-difference/v1"
EVALUATION_PROFILE = "descriptive-pearson/v1"
PRECISION = 34
_NUMBER_EXPONENT_MIN = -100
_NUMBER_EXPONENT_MAX = 100
_PAIR_ADJUSTED_EXPONENT_MAX = _NUMBER_EXPONENT_MAX + 1
_SCALED_PAIR_DIGITS = _PAIR_ADJUSTED_EXPONENT_MAX - _NUMBER_EXPONENT_MIN + 1
_MOMENT_PRODUCT_DEGREE = 4
_MOMENT_PRECISION_ALLOWANCE = 16
_CONTRACT_CONSTANTS = {
    "version": 1,
    "designation": "synthetic",
    "level_unit": "synthetic_level",
    "label_unit": "synthetic_target",
    "signal_profile": SIGNAL_PROFILE,
    "evaluation_profile": EVALUATION_PROFILE,
    "decimal_precision": PRECISION,
}
ARITHMETIC = Context(
    prec=PRECISION,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    traps=[DivisionByZero, InvalidOperation, Overflow],
)
LIMITATIONS = (
    "Synthetic development diagnostic only; no statistical null acceptance or alpha claim",
    "No significance, portfolio, execution, or real-framework acceptance conclusion",
    "Caller-supplied environment lock; installation and dependency reconciliation unverified",
    "Software exposure events do not establish human exposure or complete disclosure",
)


class ReplayError(ValueError):
    """Report an invalid replay request before creating an execution attempt."""


class FixtureError(ReplayError):
    """Identify a malformed fixture record rather than a missing available input."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class NewTrial:
    question: str
    history: tuple[VerifiedPackage, ...] = ()
    disclosures: tuple[ExposureEvent, ...] = ()
    reason: str = "Continues supplied development history"


@dataclass(frozen=True)
class Repeat:
    previous: VerifiedPackage


@dataclass(frozen=True)
class Retry:
    previous: VerifiedPackage
    attempt_id: UUID


type RunIntent = NewTrial | Repeat | Retry


@dataclass(frozen=True)
class Session:
    id: str
    occurred_at: datetime


@dataclass(frozen=True)
class Contract:
    version: int
    designation: Literal["synthetic"]
    dataset_id: str
    sessions: tuple[Session, ...]
    level_unit: Literal["synthetic_level"]
    label_unit: Literal["synthetic_target"]
    evaluation_cutoff: datetime
    signal_profile: str
    evaluation_profile: str
    decimal_precision: int

    @property
    def span(self) -> tuple[datetime, datetime]:
        return (
            min(self.sessions[0].occurred_at, self.evaluation_cutoff),
            max(self.sessions[-1].occurred_at, self.evaluation_cutoff),
        )


@dataclass(frozen=True)
class Membership:
    entity_id: str
    first_session: str
    last_session: str
    known_at: datetime
    included: bool


@dataclass(frozen=True)
class VersionKey:
    entity_id: str
    session_id: str
    revision: int


@dataclass(frozen=True)
class Observation:
    key: VersionKey
    published_at: datetime
    arrived_at: datetime
    level: Decimal


@dataclass(frozen=True)
class Decision:
    entity_id: str
    session_id: str
    at: datetime


@dataclass(frozen=True)
class Label:
    entity_id: str
    decision_at: datetime
    available_at: datetime
    value: Decimal


@dataclass(frozen=True)
class Fixture:
    contract: Contract
    memberships: tuple[Membership, ...]
    observations: tuple[Observation, ...]
    decisions: tuple[Decision, ...]
    labels: tuple[Label, ...]

    @property
    def span(self) -> tuple[datetime, datetime]:
        times = (
            *self.contract.span,
            *(row.known_at for row in self.memberships),
            *(time for row in self.observations for time in (row.published_at, row.arrived_at)),
            *(row.at for row in self.decisions),
            *(time for row in self.labels for time in (row.decision_at, row.available_at)),
        )
        return min(times), max(times)


@dataclass(frozen=True)
class Selection:
    decision: Decision
    vintage: datetime | None
    membership: Membership | None
    current: VersionKey | None
    previous: VersionKey | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Signal:
    decision: Decision
    value: Decimal
    current: VersionKey
    previous: VersionKey


@dataclass(frozen=True)
class Pair:
    decision: Decision
    signal: Decimal
    label: Decimal


@dataclass(frozen=True)
class LabelSelection:
    decision: Decision
    label: Label | None
    reason: Literal["label_missing", "label_unavailable"] | None


@dataclass(frozen=True)
class Diagnostic:
    pairs: int
    coefficient: Decimal
    centered_cross: Decimal
    signal_variance: Decimal
    label_variance: Decimal


@dataclass(frozen=True)
class Inconclusive:
    pairs: int
    reason: Literal["insufficient_pairs", "zero_variance"]


@dataclass(frozen=True)
class Rejected:
    code: str
    detail: str


@dataclass(frozen=True)
class ReplayResult:
    selections: tuple[Selection, ...]
    signals: tuple[Signal, ...]
    label_selections: tuple[LabelSelection, ...]
    pairs: tuple[Pair, ...]
    diagnostic: Diagnostic | Inconclusive | Rejected


def _json_default(value: Decimal | datetime) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    raise TypeError(f"Unsupported replay value {value!r}")


def _json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            default=_json_default,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _fields(value: object, names: str, label: str) -> dict:
    if not isinstance(value, dict) or value.keys() != set(names.split()):
        raise FixtureError("invalid_fields", f"{label} must have exactly these fields: {names}")
    return value


def _array(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise FixtureError("invalid_array", f"{label} must be an array")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise FixtureError("invalid_identifier", f"{label} must be a stable synthetic identifier")
    return value


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise FixtureError("invalid_time", f"Invalid UTC timestamp {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise FixtureError("invalid_time", f"Invalid UTC timestamp {value!r}") from error
    if parsed.utcoffset() != timedelta(0) or _json_default(parsed) != value:
        raise FixtureError("invalid_time", f"Expected a canonical UTC timestamp, got {value!r}")
    return parsed


def _number(value: object) -> Decimal:
    if not isinstance(value, str):
        raise FixtureError("invalid_number", "Levels and labels must be decimal strings")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise FixtureError("invalid_number", f"Invalid decimal {value!r}") from error
    if not number.is_finite():
        raise FixtureError("nonfinite_number", "Levels and labels must be finite")
    if not re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", value, re.ASCII):
        raise FixtureError("invalid_number", f"Invalid decimal syntax {value!r}")
    if (
        len(number.as_tuple().digits) > PRECISION
        or not _NUMBER_EXPONENT_MIN <= number.as_tuple().exponent <= _NUMBER_EXPONENT_MAX
        or not _NUMBER_EXPONENT_MIN <= number.adjusted() <= _NUMBER_EXPONENT_MAX
    ):
        raise FixtureError(
            "number_out_of_profile",
            f"Profile v1 accepts at most {PRECISION} significant digits and exponents within "
            f"[{_NUMBER_EXPONENT_MIN}, {_NUMBER_EXPONENT_MAX}]",
        )
    return number


def _nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number {value!r}")


def _read_contract(raw: bytes) -> tuple[dict, Contract]:
    try:
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_keys, parse_constant=_nonfinite
        )
    except (ValueError, RecursionError) as error:
        raise FixtureError("invalid_json", f"Cannot parse fixture JSON: {error}") from error
    if not isinstance(document, dict) or "contract" not in document:
        raise FixtureError("missing_contract", "A declared synthetic contract is required")
    value = _fields(
        document["contract"],
        "version designation dataset_id sessions level_unit label_unit evaluation_cutoff "
        "signal_profile evaluation_profile decimal_precision",
        "contract",
    )
    for key, expected in _CONTRACT_CONSTANTS.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise FixtureError("unsupported_contract", f"{key} must be {expected!r}")
    sessions = []
    for item in _array(value["sessions"], "sessions"):
        row = _fields(item, "id occurred_at", "session")
        sessions.append(Session(_identifier(row["id"], "session ID"), _time(row["occurred_at"])))
    if (
        not sessions
        or len({session.id for session in sessions}) != len(sessions)
        or len({session.occurred_at for session in sessions}) != len(sessions)
    ):
        raise FixtureError("invalid_calendar", "Sessions require distinct IDs and UTC times")
    return document, Contract(
        **_CONTRACT_CONSTANTS,
        dataset_id=_identifier(value["dataset_id"], "dataset ID"),
        sessions=tuple(sorted(sessions, key=lambda session: session.occurred_at)),
        evaluation_cutoff=_time(value["evaluation_cutoff"]),
    )


def _fixture(document: dict, contract: Contract) -> Fixture:
    value = _fields(
        document, "contract entities memberships observations decisions labels", "fixture"
    )
    entities = tuple(
        _identifier(item, "entity ID") for item in _array(value["entities"], "entities")
    )
    entity_ids = set(entities)
    if len(entity_ids) != len(entities):
        raise FixtureError("duplicate_entity", "Entity IDs must be distinct")
    sessions = {session.id: index for index, session in enumerate(contract.sessions)}

    def entity(item: object) -> str:
        identity = _identifier(item, "entity ID")
        if identity not in entity_ids:
            raise FixtureError("unknown_entity", f"Unknown entity {identity!r}")
        return identity

    def session(item: object) -> str:
        identity = _identifier(item, "session ID")
        if identity not in sessions:
            raise FixtureError("unknown_session", f"Unknown session {identity!r}")
        return identity

    memberships = []
    membership_intervals = defaultdict(list)
    for item in _array(value["memberships"], "memberships"):
        row = _fields(item, "entity_id first_session last_session known_at included", "membership")
        membership = Membership(
            entity(row["entity_id"]),
            session(row["first_session"]),
            session(row["last_session"]),
            _time(row["known_at"]),
            row["included"],
        )
        first, last = sessions[membership.first_session], sessions[membership.last_session]
        if type(membership.included) is not bool or first > last:
            raise FixtureError(
                "invalid_membership", "Membership needs a boolean and ordered interval"
            )
        intervals = membership_intervals[membership.entity_id, membership.known_at]
        if any(
            first <= prior_last and prior_first <= last for prior_first, prior_last in intervals
        ):
            raise FixtureError(
                "ambiguous_membership",
                "Membership intervals known at the same time must not overlap",
            )
        intervals.append((first, last))
        memberships.append(membership)
    observations = []
    versions = set()
    for item in _array(value["observations"], "observations"):
        row = _fields(
            item, "entity_id session_id revision published_at arrived_at level unit", "observation"
        )
        if type(row["revision"]) is not int or row["revision"] < 1:
            raise FixtureError(
                "invalid_revision", "Observation revisions must be positive integers"
            )
        key = VersionKey(entity(row["entity_id"]), session(row["session_id"]), row["revision"])
        if key in versions:
            raise FixtureError("duplicate_observation", f"Duplicate observation identity {key}")
        if row["unit"] != contract.level_unit:
            raise FixtureError(
                "invalid_unit", "Observation unit differs from the declared level unit"
            )
        versions.add(key)
        observations.append(
            Observation(
                key, _time(row["published_at"]), _time(row["arrived_at"]), _number(row["level"])
            )
        )
    decisions = []
    decision_ids = set()
    for item in _array(value["decisions"], "decisions"):
        row = _fields(item, "entity_id session_id at", "decision")
        decision = Decision(entity(row["entity_id"]), session(row["session_id"]), _time(row["at"]))
        key = decision.entity_id, decision.at
        if key in decision_ids:
            raise FixtureError(
                "duplicate_decision", "Decision entity/time identities must be distinct"
            )
        if decision.at > contract.evaluation_cutoff:
            raise FixtureError(
                "invalid_decision_time", "Decisions must not follow evaluation cutoff"
            )
        decision_ids.add(key)
        decisions.append(decision)
    labels = []
    label_ids = set()
    for item in _array(value["labels"], "labels"):
        row = _fields(item, "entity_id decision_at available_at value unit", "label")
        label = Label(
            entity(row["entity_id"]),
            _time(row["decision_at"]),
            _time(row["available_at"]),
            _number(row["value"]),
        )
        key = label.entity_id, label.decision_at
        if key in label_ids:
            raise FixtureError(
                "duplicate_label", "Label entity/decision identities must be distinct"
            )
        if key not in decision_ids:
            raise FixtureError("unknown_decision", "A label must reference a declared decision")
        if label.available_at <= label.decision_at:
            raise FixtureError(
                "invalid_label_time", "A label must become available after its decision"
            )
        if row["unit"] != contract.label_unit:
            raise FixtureError("invalid_unit", "Label unit differs from the declared target unit")
        label_ids.add(key)
        labels.append(label)
    return Fixture(
        contract,
        tuple(memberships),
        tuple(observations),
        tuple(sorted(decisions, key=lambda row: (sessions[row.session_id], row.entity_id, row.at))),
        tuple(labels),
    )


def _replay(fixture: Fixture) -> ReplayResult:
    sessions = fixture.contract.sessions
    session_indices = {session.id: index for index, session in enumerate(sessions)}
    observations = defaultdict(list)
    for observation in fixture.observations:
        observations[observation.key.entity_id, observation.key.session_id].append(observation)
    memberships = defaultdict(list)
    for membership in fixture.memberships:
        memberships[membership.entity_id].append(membership)
    selections = []
    signals = []
    for decision in fixture.decisions:
        index = session_indices[decision.session_id]
        records = memberships[decision.entity_id]
        vintage = max(
            (item.known_at for item in records if item.known_at <= decision.at), default=None
        )
        effective = [
            item
            for item in records
            if session_indices[item.first_session] <= index <= session_indices[item.last_session]
        ]
        member = next((item for item in effective if item.known_at == vintage), None)
        if member is None or not member.included:
            reason = (
                "membership_excluded"
                if member
                else "membership_superseded"
                if any(item.known_at <= decision.at for item in effective)
                else "membership_unavailable"
                if effective
                else "membership_missing"
            )
            selections.append(Selection(decision, vintage, member, None, None, (reason,)))
            continue
        reasons = []
        selected = []
        for role, position in (("current", index), ("previous", index - 1)):
            if position < 0:
                reasons.append("previous_session_missing")
                selected.append(None)
                continue
            session = sessions[position]
            versions = observations[decision.entity_id, session.id]
            available = max(
                (
                    item
                    for item in versions
                    if session.occurred_at <= decision.at
                    and item.published_at <= decision.at
                    and item.arrived_at <= decision.at
                ),
                key=lambda item: item.key.revision,
                default=None,
            )
            selected.append(available)
            if available is None:
                reason = (
                    "session_not_occurred"
                    if session.occurred_at > decision.at
                    else "unavailable"
                    if versions
                    else "missing"
                )
                reasons.append(f"{role}_{reason}")
        current, previous = selected
        selections.append(
            Selection(
                decision,
                vintage,
                member,
                current.key if current else None,
                previous.key if previous else None,
                tuple(reasons),
            )
        )
        if current is not None and previous is not None:
            signals.append(
                Signal(decision, current.level - previous.level, current.key, previous.key)
            )
    labels = {(label.entity_id, label.decision_at): label for label in fixture.labels}
    label_selections = []
    pairs = []
    for signal in signals:
        label = labels.get((signal.decision.entity_id, signal.decision.at))
        if label is None:
            label_selections.append(LabelSelection(signal.decision, None, "label_missing"))
        elif label.available_at > fixture.contract.evaluation_cutoff:
            label_selections.append(LabelSelection(signal.decision, None, "label_unavailable"))
        else:
            label_selections.append(LabelSelection(signal.decision, label, None))
            pairs.append(Pair(signal.decision, signal.value, label.value))
    eligible_pairs = tuple(pairs)
    diagnostic = _diagnostic(eligible_pairs)
    return ReplayResult(
        tuple(selections), tuple(signals), tuple(label_selections), eligible_pairs, diagnostic
    )


def _diagnostic(pairs: tuple[Pair, ...]) -> Diagnostic | Inconclusive:
    count = len(pairs)
    if count < 3:
        return Inconclusive(count, "insufficient_pairs")
    with localcontext(ARITHMETIC) as context:
        context.prec = (
            _MOMENT_PRODUCT_DEGREE * (_SCALED_PAIR_DIGITS + len(str(count)))
            + _MOMENT_PRECISION_ALLOWANCE
        )
        signal_sum = sum(pair.signal for pair in pairs)
        label_sum = sum(pair.label for pair in pairs)
        cross = count * sum(pair.signal * pair.label for pair in pairs) - signal_sum * label_sum
        signal_variance = count * sum(pair.signal**2 for pair in pairs) - signal_sum**2
        label_variance = count * sum(pair.label**2 for pair in pairs) - label_sum**2
        if signal_variance == 0 or label_variance == 0:
            return Inconclusive(count, "zero_variance")
        numerator = cross**2
        denominator = signal_variance * label_variance
    with localcontext(ARITHMETIC):
        coefficient = (numerator / denominator).sqrt().copy_sign(cross)
        return Diagnostic(
            count, coefficient, cross / count, signal_variance / count, label_variance / count
        )


def _sources(directory: Traversable, prefix: str) -> Iterator[tuple[str, str]]:
    for item in sorted(directory.iterdir(), key=lambda item: item.name):
        if item.is_dir():
            yield from _sources(item, f"{prefix}{item.name}/")
        elif item.name.endswith(".py"):
            yield f"{prefix}{item.name}", item.read_bytes().decode("utf-8")


def _merge_history(
    history: tuple[VerifiedPackage, ...], disclosures: tuple[ExposureEvent, ...]
) -> tuple[dict[UUID, TrialRecord], dict[UUID, ExposureEvent], dict[str, ArtifactInput]]:
    trials = {}
    exposures = {}
    artifacts = {}
    for package in history:
        for trial in package.record.trials:
            previous = trials.get(trial.trial_id)
            if previous is None:
                trials[trial.trial_id] = trial
                continue
            if replace(previous, attempts=()) != replace(trial, attempts=()):
                raise ReplayError(f"Conflicting trial identity {trial.trial_id}")
            attempts = {attempt.id: attempt for attempt in previous.attempts}
            for attempt in trial.attempts:
                if attempts.setdefault(attempt.id, attempt) != attempt:
                    raise ReplayError(f"Conflicting attempt identity {attempt.id}")
            trials[trial.trial_id] = replace(previous, attempts=tuple(attempts.values()))
        for artifact in package.artifacts():
            if artifacts.setdefault(artifact.name, artifact) != artifact:
                raise ReplayError(f"Conflicting artifact name {artifact.name!r}")
    for event in (
        *(event for package in history for event in package.record.exposures),
        *disclosures,
    ):
        if exposures.setdefault(event.id, event) != event:
            raise ReplayError(f"Conflicting exposure identity {event.id}")
    return trials, exposures, artifacts


def replay_fixture(
    fixture: Path, destination: Path, *, intent: RunIntent, lock: bytes
) -> VerifiedPackage:
    """Replay a declared synthetic fixture and preserve all supplied history.

    Args:
        fixture (Path): UTF-8 fixture JSON in the fixed version 1 format.
        destination (Path): Parent directory for portable package revisions.
        intent (RunIntent): New scientific trial, deliberate repeat, or execution retry.
        lock (bytes): Exact caller-supplied environment lock. Replay does not install or verify it.

    Returns:
        (VerifiedPackage) A diagnostic, inconclusive, or rejected development attempt.

    Raises:
        FixtureError: JSON or its contract cannot establish the synthetic trial context.
        ReplayError: Intent, history, or environment input is inconsistent. No trial is created.
        PackageError: Supplied history, output records, or package publication fails validation,
            or a Repeat or Retry names a package that selects no trial.
        OSError: Reading the fixture or packaged implementation source fails.

    A valid contract permits malformed rows to become a rejected attempt with the raw input.
    Row rejection never produces a coefficient. Missing eligible inputs are valid and recorded.
    History packages that select no trial supply their trials, exposures, and artifacts to a
    new trial without a lineage link.
    """
    if not isinstance(lock, bytes) or not lock:
        raise ReplayError("Supply nonempty environment lock bytes explicitly")
    if not isinstance(intent, (NewTrial, Repeat, Retry)):
        raise ReplayError("Use an explicit NewTrial, Repeat, or Retry intent")
    started_at = datetime.now(UTC)
    raw = fixture.read_bytes()
    input_read_at = datetime.now(UTC)
    document, contract = _read_contract(raw)
    supplied = intent.history if isinstance(intent, NewTrial) else (intent.previous,)
    history = tuple(open_package(package.path) for package in supplied)
    if any(
        opened.digest != package.digest for opened, package in zip(history, supplied, strict=True)
    ):
        raise ReplayError("Supplied history changed after opening")
    disclosures = intent.disclosures if isinstance(intent, NewTrial) else ()
    trials, exposures, artifacts = _merge_history(history, disclosures)
    previous = None if isinstance(intent, NewTrial) else history[0].record.selected_trial
    question = intent.question if isinstance(intent, NewTrial) else previous.specification.question
    implementation = _json(
        {
            "entrypoint": "quant_research.replay:replay_fixture",
            "sources": dict(_sources(resources.files("quant_research"), "quant_research/")),
        }
    )
    payloads = {
        "input": raw,
        "contract": _json(asdict(contract)),
        "implementation": implementation,
        "lock": lock,
    }
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    specification = Specification(
        question,
        SIGNAL_PROFILE,
        EVALUATION_PROFILE,
        f"{contract.dataset_id}/fixture-v1",
        "causal-replay/v1",
        (digests["input"],),
        digests["contract"],
        digests["implementation"],
        EnvironmentRecord(
            sys.implementation.name,
            ".".join(str(part) for part in sys.version_info[:3]),
            sysconfig.get_config_var("SOABI"),
            sysconfig.get_platform(),
            "not observed; caller-supplied lock",
            digests["lock"],
        ),
    )
    if previous is not None and previous.specification != specification:
        raise ReplayError("Repeat and retry require identical specification; use a linked NewTrial")
    predecessor = None
    run_id = uuid4()
    if isinstance(intent, Retry):
        predecessor = next(
            (attempt for attempt in previous.attempts if attempt.id == intent.attempt_id), None
        )
        if predecessor is None:
            raise ReplayError("Retry must name an existing attempt in the selected trial")
        run_id = predecessor.run_id
    attempt_id = uuid4()
    with localcontext(ARITHMETIC):
        try:
            parsed = _fixture(document, contract)
        except FixtureError as error:
            result = ReplayResult((), (), (), (), Rejected(error.code, str(error)))
            span, extent = contract.span, "is the declared calendar/evaluation span"
        else:
            result = _replay(parsed)
            span, extent = parsed.span, "spans every timestamp in the raw fixture"
        match result.diagnostic:
            case Rejected(code=code, detail=detail):
                status, detail = "rejected", f"Fixture rejected ({code}): {detail}"
            case Inconclusive(reason=reason):
                status, detail = "inconclusive", f"Development diagnostic is inconclusive: {reason}"
            case Diagnostic(coefficient=coefficient):
                status = "null" if coefficient == 0 else "successful"
                detail = f"Descriptive development Pearson coefficient {coefficient}"
        payloads.update(
            {
                "provenance": _json([asdict(row) for row in result.selections]),
                "signals": _json([asdict(row) for row in result.signals]),
                "evaluation": _json(
                    {
                        "labels": [asdict(row) for row in result.label_selections],
                        "pairs": [asdict(row) for row in result.pairs],
                    }
                ),
                "diagnostic": _json(
                    {
                        "profile": EVALUATION_PROFILE,
                        "status": status,
                        "decisions": len(result.selections),
                        "signals": len(result.signals),
                        **asdict(result.diagnostic),
                    }
                ),
            }
        )
    names = {role: f"replay-{attempt_id.hex}-{role}" for role in payloads}
    for role, data in payloads.items():
        artifacts[names[role]] = ArtifactInput(
            names[role], data, "text/plain" if role == "lock" else "application/json", role
        )
    attempt = ExecutionAttempt(
        attempt_id,
        run_id,
        predecessor.id if predecessor else None,
        started_at,
        datetime.now(UTC),
        Outcome(
            status,
            detail,
            tuple(names[role] for role in ("signals", "provenance", "evaluation", "diagnostic")),
            LIMITATIONS,
        ),
    )
    if previous is None:
        parents = {
            package.record.trial_id: trials[package.record.trial_id]
            for package in history
            if package.record.trial_id is not None
        }
        selected = TrialRecord(
            uuid4(),
            next(iter(parents.values())).lineage_id if parents else uuid4(),
            specification,
            (attempt,),
            tuple(
                LineageLink(parent.trial_id, parent.specification.spec_id, "follows", intent.reason)
                for parent in parents.values()
            ),
        )
    else:
        selected = replace(previous, attempts=(*previous.attempts, attempt))
    trials[selected.trial_id] = selected
    for kind, role, at, reason in (
        ("data", "input", input_read_at, "Software completed reading the raw synthetic fixture"),
        (
            "result",
            "diagnostic",
            attempt.finished_at,
            "Software produced the development diagnostic",
        ),
    ):
        event = ExposureEvent(
            uuid4(),
            "software:quant_research.replay/v1",
            at,
            hashlib.sha256(payloads[role]).hexdigest(),
            contract.dataset_id,
            *span,
            kind,
            f"{reason}; interval {extent}; human exposure is unknown",
        )
        exposures[event.id] = event
    record = PackageRecord(
        selected.trial_id,
        tuple(trials.values()),
        tuple(exposures.values()),
        "declared_inputs"
        if status != "rejected"
        and all(package.record.disclosure_coverage == "declared_inputs" for package in history)
        else "unknown",
    )
    return save_package(destination, record, tuple(artifacts.values()))
