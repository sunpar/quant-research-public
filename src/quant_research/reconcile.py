from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, Inexact, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Literal
from uuid import UUID

from quant_research import datasets, signals
from quant_research.access import RunContext
from quant_research.catalog import Catalog
from quant_research.datasets import Refused
from quant_research.packages import VerifiedPackage
from quant_research.records import canonical_bytes, format_time, unique_keys
from quant_research.replay import EVALUATION_PROFILE

METRICS = (
    "decisions",
    "signals",
    "pairs",
    "coefficient",
    "centered_cross",
    "signal_variance",
    "label_variance",
)
_DECIMAL = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", re.ASCII)
# An exact difference of two bounded numbers has about 3000 digits, so the record can print it
# under Python's 4300-digit limit on integer-to-text conversion.
_NUMBER_LIMIT = 1000

type Status = Literal["agreed", "disagreed", "missing_from_package", "missing_from_expected"]


class ReconcileError(ValueError):
    """Refuse a reference or package that cannot be reconciled."""


@dataclass(frozen=True)
class MetricComparison:
    name: str
    recorded: Fraction | None
    expected: Fraction | None
    tolerance: Fraction | None

    @property
    def difference(self) -> Fraction | None:
        if self.recorded is None or self.expected is None:
            return None
        return self.recorded - self.expected

    @property
    def status(self) -> Status:
        if self.recorded is None:
            return "missing_from_package"
        if self.expected is None:
            return "missing_from_expected"
        if self.tolerance is not None and abs(self.difference) <= self.tolerance:
            return "agreed"
        return "disagreed"


@dataclass(frozen=True)
class Reconciliation:
    package_digest: str
    attempt_id: UUID
    reference_digest: str
    reference: tuple[MetricComparison, ...]
    recomputation: tuple[MetricComparison, ...]

    @property
    def agreed(self) -> bool:
        return all(row.status == "agreed" for row in (*self.reference, *self.recomputation))

    def record(self) -> bytes:
        """Return canonical JSON that omits the package path, so a copy gives the same bytes."""
        return canonical_bytes(
            {
                "agreed": self.agreed,
                "attempt_id": str(self.attempt_id),
                "package_digest": self.package_digest,
                "reference_digest": self.reference_digest,
                "reference": _rows(self.reference),
                "recomputation": _rows(self.recomputation),
            }
        )


def _rows(rows: tuple[MetricComparison, ...]) -> dict:
    return {
        row.name: {
            "recorded": _text(row.recorded),
            "expected": _text(row.expected),
            "tolerance": _text(row.tolerance),
            "difference": _text(row.difference),
            "status": row.status,
        }
        for row in rows
    }


def _text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    numerator, denominator = value.numerator, value.denominator
    exact = Context(prec=len(str(numerator)) + 4 * len(str(denominator)), traps=[Inexact])
    return str(exact.divide(Decimal(numerator), Decimal(denominator)))


def _json(raw: bytes, label: str) -> object:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys)
        # The record re-encodes metric names, so a name UTF-8 cannot encode is refused here.
        canonical_bytes(value)
    except (ValueError, RecursionError) as error:
        raise ReconcileError(f"{label} is not strict JSON: {error}") from error
    return value


def _number(value: object, label: str) -> Fraction:
    with suppress(InvalidOperation):
        if type(value) is int or (isinstance(value, str) and _DECIMAL.fullmatch(value)):
            number = Decimal(value)
            if (
                len(number.as_tuple().digits) <= _NUMBER_LIMIT
                and abs(number.adjusted()) <= _NUMBER_LIMIT
            ):
                return Fraction(number)
    raise ReconcileError(
        f"{label} must be an integer or a decimal string with at most {_NUMBER_LIMIT} digits "
        f"after its leading zeros and an exponent within [-{_NUMBER_LIMIT}, {_NUMBER_LIMIT}], "
        f"not {value!r}"
    )


def _reference(raw: bytes) -> dict[str, tuple[Fraction, Fraction]]:
    document = _json(raw, "Reference")
    if not isinstance(document, dict) or document.keys() != {"version", "metrics"}:
        raise ReconcileError("Reference must have exactly the fields version and metrics")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ReconcileError(f"Unsupported reference version {document['version']!r}")
    metrics = document["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        raise ReconcileError("Reference metrics must be a nonempty object")
    expected = {}
    for name, entry in metrics.items():
        if not isinstance(entry, dict) or entry.keys() != {"value", "tolerance"}:
            raise ReconcileError(f"Reference metric {name!r} must declare a value and tolerance")
        tolerance = _number(entry["tolerance"], f"Tolerance for {name!r}")
        if tolerance < 0:
            raise ReconcileError(f"Tolerance for {name!r} is negative")
        expected[name] = (_number(entry["value"], f"Value for {name!r}"), tolerance)
    return expected


def _recorded(raw: bytes) -> dict[str, Fraction]:
    document = _json(raw, "Diagnostic")
    if not isinstance(document, dict) or document.get("profile") != EVALUATION_PROFILE:
        raise ReconcileError(f"Diagnostic profile must be {EVALUATION_PROFILE!r}")
    return {
        name: _number(document[name], f"Recorded {name}") for name in METRICS if name in document
    }


def _recompute(raw: bytes) -> dict[str, Fraction]:
    document = json.loads(raw)
    contract = document["contract"]
    context = Context(prec=contract["decimal_precision"], rounding=ROUND_HALF_EVEN)

    def rounded(value: Fraction) -> Decimal:
        return context.divide(Decimal(value.numerator), Decimal(value.denominator))

    cutoff = datetime.fromisoformat(contract["evaluation_cutoff"])
    sessions = sorted(
        (datetime.fromisoformat(row["occurred_at"]), row["id"]) for row in contract["sessions"]
    )
    order = {session_id: index for index, (_, session_id) in enumerate(sessions)}
    memberships = defaultdict(list)
    for row in document["memberships"]:
        memberships[row["entity_id"]].append(
            (
                datetime.fromisoformat(row["known_at"]),
                order[row["first_session"]],
                order[row["last_session"]],
                row["included"],
            )
        )
    versions = defaultdict(list)
    for row in document["observations"]:
        usable_at = max(map(datetime.fromisoformat, (row["published_at"], row["arrived_at"])))
        versions[row["entity_id"], row["session_id"]].append(
            (row["revision"], usable_at, Fraction(row["level"]))
        )
    labels = {
        (row["entity_id"], datetime.fromisoformat(row["decision_at"])): (
            datetime.fromisoformat(row["available_at"]),
            Fraction(row["value"]),
        )
        for row in document["labels"]
    }

    def level(entity: str, index: int, at: datetime) -> Fraction | None:
        occurred_at, session_id = sessions[index]
        usable = [
            (revision, value)
            for revision, usable_at, value in versions[entity, session_id]
            if usable_at <= at
        ]
        return max(usable)[1] if occurred_at <= at and usable else None

    signals = 0
    pairs = []
    for row in document["decisions"]:
        entity, index = row["entity_id"], order[row["session_id"]]
        at = datetime.fromisoformat(row["at"])
        vintage = max(
            (known_at for known_at, *_ in memberships[entity] if known_at <= at), default=None
        )
        member = [
            included
            for known_at, first, last, included in memberships[entity]
            if known_at == vintage and first <= index <= last
        ]
        if member != [True] or index == 0:
            continue
        current, previous = level(entity, index, at), level(entity, index - 1, at)
        if current is None or previous is None:
            continue
        signals += 1
        label = labels.get((entity, at))
        if label is not None and label[0] <= cutoff:
            pairs.append((Fraction(rounded(current - previous)), label[1]))

    counts = {
        "decisions": Fraction(len(document["decisions"])),
        "signals": Fraction(signals),
        "pairs": Fraction(len(pairs)),
    }
    if len(pairs) < 3:
        return counts
    signal_mean = sum(signal for signal, _ in pairs) / len(pairs)
    label_mean = sum(label for _, label in pairs) / len(pairs)
    cross = sum((signal - signal_mean) * (label - label_mean) for signal, label in pairs)
    signal_variance = sum((signal - signal_mean) ** 2 for signal, _ in pairs)
    label_variance = sum((label - label_mean) ** 2 for _, label in pairs)
    if not signal_variance or not label_variance:
        return counts
    magnitude = Fraction(context.sqrt(rounded(cross**2 / (signal_variance * label_variance))))
    return counts | {
        "coefficient": -magnitude if cross < 0 else magnitude,
        "centered_cross": Fraction(rounded(cross)),
        "signal_variance": Fraction(rounded(signal_variance)),
        "label_variance": Fraction(rounded(label_variance)),
    }


def _compare(
    recorded: dict[str, Fraction], expected: dict[str, tuple[Fraction, Fraction]]
) -> tuple[MetricComparison, ...]:
    return tuple(
        MetricComparison(name, recorded.get(name), *expected.get(name, (None, None)))
        for name in sorted(recorded.keys() | expected.keys())
    )


def reconcile(package: VerifiedPackage, reference: Path) -> Reconciliation:
    """Check the selected attempt's diagnostic against a reference and a recomputation.

    Args:
        package (VerifiedPackage): Opened package whose latest selected attempt is checked.
        reference (Path): Version 1 reference JSON with a declared tolerance for every metric.

    Returns:
        (Reconciliation) Rows for each side. Recomputed rows use tolerance 0 because they
            repeat the profile's declared roundings from the saved raw input.

    Raises:
        ReconcileError: The reference is malformed, the selected trial has no attempts, or
            the attempt has no diagnostic.
        OSError: The reference cannot be read.
        PackageError: The package selects no trial, or its artifact bytes changed after opening.
    """
    raw_reference = reference.read_bytes()
    expected = _reference(raw_reference)
    trial = package.record.selected_trial
    if not trial.attempts:
        raise ReconcileError("The selected trial has no attempts to reconcile")
    attempt = trial.attempts[-1]
    if attempt.outcome.status == "rejected":
        raise ReconcileError("A rejected attempt has no diagnostic to reconcile")
    diagnostic = next(
        (
            item
            for item in package.artifact_records
            if item.role == "diagnostic" and item.name in attempt.outcome.result_artifacts
        ),
        None,
    )
    source = next(
        (
            item
            for item in package.artifact_records
            if item.role == "input" and item.digest in trial.specification.input_digests
        ),
        None,
    )
    if diagnostic is None or source is None:
        raise ReconcileError("The selected attempt needs diagnostic and raw input artifacts")
    recorded = _recorded(package.read_artifact(diagnostic.name))
    recomputed = _recompute(package.read_artifact(source.name))
    return Reconciliation(
        package.digest,
        attempt.id,
        hashlib.sha256(raw_reference).hexdigest(),
        _compare(recorded, expected),
        _compare(recorded, {name: (value, Fraction(0)) for name, value in recomputed.items()}),
    )


@dataclass(frozen=True)
class SeriesProfile:
    """How to recompute a published series, and the months declared as known exceptions.

    reference names the dataset that holds the published series, and exclusions maps each
    declared month to its recorded reason.
    """

    name: str
    program: signals.Number
    reference: str
    tolerance: Fraction
    exclusions: Mapping[date, str]


# (month, ALFRED vintage the decision read, later release whose window FRED's value matches)
_REVISED_WINDOWS = (
    ("1960-12", "1961-01-13", "1961-02-09"),
    ("1963-01", "1963-01-31", "1963-03-07"),
    ("1963-12", "1964-01-10", "1964-02-06"),
    ("1966-12", "1967-01-09", "1967-02-09"),
    ("1967-12", "1968-01-10", "1968-02-07"),
    ("1970-12", "1971-01-08", "1971-02-05"),
    ("1972-12", "1973-01-05", "1973-02-02"),
    ("1976-12", "1977-01-12", "1977-02-04"),
    ("1977-12", "1978-01-11", "1978-02-03"),
    ("1981-12", "1982-01-08", "1982-02-05"),
)
# Half of FRED's last digit. Recomputed values lie on a 1/60 grid and FRED's on a 1/100 grid,
# so no difference sits exactly on the bound.
SAHM_REALTIME = SeriesProfile(
    "sahm-realtime/v1",
    signals.SAHM,
    "alfred.sahmrealtime",
    Fraction(1, 200),
    {
        **{
            date.fromisoformat(f"{month}-01"): (
                f"FRED's value equals the indicator from the {later} release, which revised the "
                f"window after the {read} vintage this decision read"
            )
            for month, read, later in _REVISED_WINDOWS
        },
        date(2021, 3, 1): (
            "FRED's 2.33 needs January 2020 at 3.6, its value before the 2021-01-08 revision, "
            "and no ALFRED vintage holding March 2021 has it"
        ),
    },
)


@dataclass(frozen=True)
class SeriesMonth:
    """One window month: the published value, the distinct recomputed values, and any reason."""

    month: date
    reference: Fraction | None
    recomputed: tuple[Fraction, ...]
    reason: str | None = None


@dataclass(frozen=True)
class SeriesReconciliation:
    profile: str
    cutoff: datetime
    inputs: Mapping[str, str]
    catalog: str
    start: date
    end: date
    matched: tuple[SeriesMonth, ...]
    excluded: tuple[SeriesMonth, ...]
    mismatched: tuple[SeriesMonth, ...]
    before_window: tuple[date, ...]

    @property
    def agreed(self) -> bool:
        return not self.mismatched

    def record(self) -> bytes:
        """Return canonical JSON that omits the data root, so a copied root gives the same bytes."""
        return canonical_bytes(
            {
                "agreed": self.agreed,
                "profile": self.profile,
                "cutoff": format_time(self.cutoff),
                "inputs": dict(self.inputs),
                "catalog": self.catalog,
                "window": {"start": self.start.isoformat(), "end": self.end.isoformat()},
                "matched": len(self.matched),
                "excluded": [
                    {"month": item.month.isoformat(), "reason": item.reason, **_values(item)}
                    for item in self.excluded
                ],
                "mismatched": [
                    {"month": item.month.isoformat(), **_values(item)} for item in self.mismatched
                ],
                "before_window": [month.isoformat() for month in self.before_window],
            }
        )


def _values(item: SeriesMonth) -> dict:
    return {
        "reference": None if item.reference is None else str(item.reference),
        "recomputed": [str(value) for value in item.recomputed],
    }


def _span(first: date, last: date) -> list[date]:
    count = (last.year - first.year) * 12 + last.month - first.month + 1
    return [signals.shift(first, offset) for offset in range(count)]


def _unobserved(
    month: date, reference: Fraction | None, decisions: tuple[signals.Decision, ...]
) -> str:
    if reference is None:
        return "FRED has no value and the program cannot compute the month"
    later = next(
        (item for item in decisions if item.observation is not None and item.observation > month),
        None,
    )
    if later is None:
        return "no decision before the run's cutoff observes it"
    return (
        f"no decision observes it; the decision for {later.month:%Y-%m} already observes "
        f"{later.observation:%Y-%m}"
    )


def reconcile_series(
    profile: SeriesProfile, catalog: Catalog, store: Path, context: RunContext
) -> SeriesReconciliation:
    """Compare a program's causal recomputation with the published series it reproduces.

    Decisions run from the month after the reference's first month through the month holding
    the context's cutoff. The window runs from the later of the first observed month and the
    first reference month to the last reference month. A window month matches when a decision
    observes it, the reference has a value, and every observing decision is within tolerance.
    Otherwise it is excluded with the profile's declared reason, excluded because no decision
    observes it, or mismatched.

    Args:
        profile (SeriesProfile): Program, reference dataset, tolerance, and declared exclusions.
        catalog (Catalog): Catalog that must approve the reference and every program dataset.
        store (Path): Dataset store directory.
        context (RunContext): Trusted run authority; its cutoff ends the decisions.

    Returns:
        (SeriesReconciliation) The window, its matched, excluded, and mismatched months, and the
            reference months released before any decision could observe them.

    Raises:
        ReconcileError: If the reference or a decision read is refused, the reference has no
            rows available by the cutoff, no decision can compute the program, or the
            reference ends before the first observed month.
        DatasetError: If a documented vintage is unreadable or malformed.
    """
    published = datasets.read(catalog, store, profile.reference, context)
    if isinstance(published, Refused):
        raise ReconcileError(f"Cannot read {profile.reference!r}: {published.message}")
    reference: dict[date, Fraction | None] = {}
    for row in published.rows:
        text = row.values["value"]
        reference[date.fromisoformat(row.key)] = None if text is None else Fraction(text)
    if not reference:
        cutoff = format_time(context.cutoff)
        raise ReconcileError(f"{profile.reference!r} has no rows available by {cutoff}")
    first, last = min(reference), max(reference)
    decisions = signals.decide(
        profile.program,
        catalog,
        store,
        context,
        _span(signals.shift(first, 1), date(context.cutoff.year, context.cutoff.month, 1)),
    )
    if isinstance(decisions, Refused):
        names = ", ".join(map(repr, sorted(profile.program.reach)))
        raise ReconcileError(f"Cannot read {names}: {decisions.message}")
    observed = defaultdict(set)
    for decision in decisions:
        if decision.observation is not None:
            observed[decision.observation].add(decision.value)
    if not observed:
        cutoff = format_time(context.cutoff)
        raise ReconcileError(f"No decision by {cutoff} can compute {profile.name}")
    start = max(min(observed), first)
    if start > last:
        raise ReconcileError(
            f"{profile.reference!r} ends at {last:%Y-%m}, before the first observed month "
            f"{start:%Y-%m}"
        )
    matched, excluded, mismatched = [], [], []
    for month in _span(start, last):
        values = observed.get(month, set())
        item = SeriesMonth(month, reference.get(month), tuple(sorted(values)))
        if (
            values
            and item.reference is not None
            and all(abs(value - item.reference) <= profile.tolerance for value in values)
        ):
            matched.append(item)
        elif month in profile.exclusions:
            excluded.append(replace(item, reason=profile.exclusions[month]))
        elif not values:
            excluded.append(replace(item, reason=_unobserved(month, item.reference, decisions)))
        else:
            mismatched.append(item)
    reads = (published.record, *(read for decision in decisions for read in decision.reads))
    return SeriesReconciliation(
        profile.name,
        context.cutoff,
        {read.dataset: read.vintage for read in reads},
        catalog.digest,
        start,
        last,
        tuple(matched),
        tuple(excluded),
        tuple(mismatched),
        tuple(month for month in reference if month < start),
    )
