from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import MAX_EMAX, MAX_PREC, MIN_EMIN, Context, Decimal
from typing import Literal
from uuid import UUID

from quant_research.packages import PackageRecord, TrialRecord, VerifiedPackage
from quant_research.replay import EVALUATION_PROFILE
from quant_research.report import single_line

type Mark = Literal["same", "different", "only in A", "only in B"]
type Relation = Literal["same lineage", "linked", "unrelated"]

_EXACT = Context(prec=MAX_PREC, Emax=MAX_EMAX, Emin=MIN_EMIN)
_METRICS = {
    EVALUATION_PROFILE: (
        "diagnostic",
        (
            "decisions",
            "signals",
            "pairs",
            "coefficient",
            "centered_cross",
            "signal_variance",
            "label_variance",
        ),
    )
}


class CompareError(ValueError):
    """Report a package that selects no trial, or a result artifact outside its profile."""


def _mark(a: object, b: object) -> Mark:
    if b is None:
        return "only in A"
    if a is None:
        return "only in B"
    return "same" if a == b else "different"


@dataclass(frozen=True)
class Identity:
    package: str
    trial_id: UUID
    lineage_id: UUID
    spec_id: str


@dataclass(frozen=True)
class Row:
    name: str
    a: str | None
    b: str | None

    @property
    def mark(self) -> Mark:
        return _mark(self.a, self.b)


@dataclass(frozen=True)
class Metric:
    name: str
    a: Decimal | None
    b: Decimal | None

    @property
    def mark(self) -> Mark:
        return _mark(self.a, self.b)

    @property
    def difference(self) -> Decimal | None:
        """Return B minus A without rounding when both sides are present."""
        if self.a is None or self.b is None:
            return None
        return _EXACT.subtract(self.b, self.a)


@dataclass(frozen=True)
class Comparison:
    """Pair the selected trials of two verified packages field by field."""

    a: Identity
    b: Identity
    specification: tuple[Row, ...]
    inputs: tuple[Row, ...]
    versions: tuple[Row, ...]
    outcome: tuple[Row, ...]
    metrics: tuple[Metric, ...]
    relation: Relation
    shared_trials: tuple[UUID, ...]

    @property
    def differences(self) -> tuple[Row | Metric, ...]:
        return tuple(row for _, rows in _sections(self) for row in rows if row.mark != "same")


def _sections(comparison: Comparison) -> tuple[tuple[str, tuple[Row | Metric, ...]], ...]:
    return (
        ("Specification", comparison.specification),
        ("Inputs", comparison.inputs),
        ("Versions", comparison.versions),
        ("Outcome", comparison.outcome),
        ("Metrics", comparison.metrics),
    )


def _identity(package: VerifiedPackage) -> Identity:
    trial = package.record.selected_trial
    return Identity(package.digest, trial.trial_id, trial.lineage_id, trial.specification.spec_id)


def _rows(trial: TrialRecord) -> dict[str, dict[str, str]]:
    spec = trial.specification
    environment = spec.environment
    return {
        "specification": {
            "question": spec.question,
            "signal profile": spec.signal_profile,
            "data version": spec.data_version,
        },
        "inputs": {
            "data contract": spec.data_contract_digest,
            "implementation": spec.implementation_digest,
            "environment lock": environment.lock_digest,
            **{f"input {number}": digest for number, digest in enumerate(spec.input_digests, 1)},
        },
        "versions": {
            "implementation version": spec.implementation_version,
            "evaluation profile": spec.evaluation_profile,
            "Python implementation": environment.implementation,
            "Python version": environment.python_version,
            "ABI": environment.abi,
            "platform": environment.platform,
            "uv version": environment.uv_version,
        },
        "outcome": _outcome(trial),
    }


def _outcome(trial: TrialRecord) -> dict[str, str]:
    if not trial.attempts:
        return {"attempts": "0"}
    outcome = trial.attempts[-1].outcome
    return {
        "attempts": str(len(trial.attempts)),
        "status": outcome.status,
        "detail": outcome.detail,
        **{f"limitation {number}": text for number, text in enumerate(outcome.limitations, 1)},
    }


def _metrics(package: VerifiedPackage) -> dict[str, Decimal]:
    trial = package.record.selected_trial
    table = _METRICS.get(trial.specification.evaluation_profile)
    if table is None or not trial.attempts:
        return {}
    role, fields = table
    names = trial.attempts[-1].outcome.result_artifacts
    artifact = next(
        (item for item in package.artifact_records if item.role == role and item.name in names),
        None,
    )
    if artifact is None:
        return {}
    source = f"Diagnostic {artifact.name!r} in {package.path}"
    content = package.read_artifact(artifact.name)
    try:
        values = json.loads(content)
    except (ValueError, RecursionError) as error:
        raise CompareError(f"{source} is not JSON: {error}") from error
    if not isinstance(values, dict):
        raise CompareError(f"{source} is not a JSON object")
    return {field: _number(source, field, values[field]) for field in fields if field in values}


def _number(source: str, field: str, value: object) -> Decimal:
    try:
        number = Decimal(value) if type(value) is int or isinstance(value, str) else None
    except ArithmeticError:
        number = None
    if number is None or not number.is_finite():
        raise CompareError(f"{source} has {field} {value!r}, not an integer or finite decimal")
    return number


def _ancestors(record: PackageRecord) -> set[UUID]:
    found = {record.trial_id}
    # Verified links name only earlier trials, so one reverse pass is transitive.
    for trial in reversed(record.trials):
        if trial.trial_id in found:
            found.update(link.trial_id for link in trial.links)
    return found


def _pair(kind: type[Row | Metric], a: dict, b: dict) -> tuple:
    return tuple(kind(name, a.get(name), b.get(name)) for name in a | b)


def compare_packages(a: VerifiedPackage, b: VerifiedPackage) -> Comparison:
    """Compare the selected trials of two verified development packages.

    Args:
        a (VerifiedPackage): Package shown as side A, opened with `open_package`.
        b (VerifiedPackage): Package shown as side B, opened with `open_package`.

    Returns:
        (Comparison) Paired specification, input, version, outcome, and metric rows, with the
            lineage relation between the two selected trials.

    Raises:
        CompareError: If either package selects no trial, or a diagnostic does not follow its
            declared evaluation profile.
    """
    for side, package in (("A", a), ("B", b)):
        if package.record.trial_id is None:
            raise CompareError(
                f"Package {side} at {package.path} selects no trial; "
                "compare needs a selected trial in each package"
            )
    rows_a, rows_b = _rows(a.record.selected_trial), _rows(b.record.selected_trial)
    shared = _ancestors(a.record) & _ancestors(b.record)
    same_lineage = a.record.selected_trial.lineage_id == b.record.selected_trial.lineage_id
    relation = "same lineage" if same_lineage else "linked" if shared else "unrelated"
    return Comparison(
        _identity(a),
        _identity(b),
        **{section: _pair(Row, values, rows_b[section]) for section, values in rows_a.items()},
        metrics=_pair(Metric, _metrics(a), _metrics(b)),
        relation=relation,
        shared_trials=tuple(sorted(shared, key=str)),
    )


def _row_lines(row: Row | Metric) -> list[str]:
    head = f"  {row.mark:<10} {row.name}"
    if row.mark == "same":
        return [f"{head}: {single_line(str(row.a))}"]
    details = [
        f"{side}: {single_line(str(value))}"
        for side, value in (("A", row.a), ("B", row.b))
        if value is not None
    ]
    if isinstance(row, Metric) and row.difference is not None:
        details.append(f"difference: {row.difference}")
    return [head, *(" " * 15 + detail for detail in details)]


def render_comparison(comparison: Comparison) -> str:
    """Render a comparison as deterministic text without filesystem paths.

    Args:
        comparison (Comparison): Result of `compare_packages`.

    Returns:
        (str) Text ending in one newline, with every value on a single line.
    """
    lines = [
        "Comparison of two verified development packages",
        "Outcome and metrics use the last supplied attempt of each selected trial. "
        "Metric differences are B minus A.",
        "",
    ]
    for side, identity in (("A", comparison.a), ("B", comparison.b)):
        lines += [
            side,
            f"  package: {identity.package}",
            f"  trial: {identity.trial_id}",
            f"  lineage: {identity.lineage_id}",
            f"  specification: {identity.spec_id}",
        ]
    lines += ["", f"Differences: {len(comparison.differences) or 'none'}"]
    for title, rows in _sections(comparison):
        lines += ["", title, *([line for row in rows for line in _row_lines(row)] or ["  none"])]
    shared = ", ".join(str(trial) for trial in comparison.shared_trials) or "none"
    lines += ["", "Lineage", f"  relation: {comparison.relation}", f"  shared trials: {shared}"]
    return "\n".join(lines) + "\n"
