from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import MAX_EMAX, MAX_PREC, MIN_EMIN, Context, Decimal
from typing import Literal
from uuid import UUID

from quant_research.evaluation import Invalid
from quant_research.packages import PackageRecord, TrialRecord, VerifiedPackage
from quant_research.reference import EVALUATION_ROLE, decode_evaluation
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
    """Report an unavailable selection, or a diagnostic outside its profile."""


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
    """Pair the compared trials of two verified packages field by field."""

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


def _trial(package: VerifiedPackage, requested: UUID | None, side: str) -> TrialRecord:
    if requested is None:
        if package.record.trial_id is None:
            raise CompareError(
                f"Package {side} at {package.path} selects no trial; "
                "compare needs a selected trial in each package"
            )
        return package.record.selected_trial
    for trial in package.record.trials:
        if trial.trial_id == requested:
            return trial
    raise CompareError(f"Package {side} at {package.path} has no trial {requested}")


def _identity(package: VerifiedPackage, trial: TrialRecord) -> Identity:
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


def _metrics(package: VerifiedPackage, trial: TrialRecord) -> dict[str, Decimal]:
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


def _pilot_evaluation(
    package: VerifiedPackage, trial: TrialRecord, requested: str | None, side: str
) -> tuple[dict[str, str], dict[str, Decimal]]:
    names = set(trial.attempts[-1].outcome.result_artifacts) if trial.attempts else set()
    eligible = sorted(
        item.name
        for item in package.artifact_records
        if item.role == EVALUATION_ROLE and item.name in names
    )
    choices = ", ".join(map(repr, eligible)) or "none"
    source = f"Package {side} at {package.path} trial {trial.trial_id}"
    if requested is not None:
        if requested not in eligible:
            raise CompareError(
                f"{source} has no eligible pilot-evaluation result artifact {requested!r} "
                f"in its last attempt; eligible artifacts: {choices}"
            )
        name = requested
    elif not eligible:
        return {}, {}
    elif len(eligible) > 1:
        raise CompareError(
            f"{source} has multiple pilot-evaluation result artifacts in its last attempt; "
            f"choose --evaluation-{side.lower()} NAME from: {choices}"
        )
    else:
        name = eligible[0]
    result = decode_evaluation(package.read_artifact(name))
    outcome = {
        "pilot artifact": name,
        "pilot kind": "invalid" if isinstance(result, Invalid) else "evaluation",
        "pilot procedure": result.profile.version,
        "pilot minimum effect": repr(result.profile.minimum_effect),
        "pilot subperiod count": str(result.profile.subperiods),
        "pilot alpha": repr(result.profile.alpha),
    }
    if isinstance(result, Invalid):
        return outcome | {"pilot reason": result.reason, "pilot invalid detail": result.detail}, {}
    outcome.update(
        {
            "pilot verdict": result.verdict,
            "pilot reason": result.reason,
            "pilot incremental interval": "absent" if result.incremental is None else "present",
            "pilot subperiod estimates by recorded position": repr(result.subperiods),
        }
    )
    metrics = {
        "pilot trial count": Decimal(result.trials),
        "pilot confidence": Decimal(repr(result.confidence)),
    }
    for label, interval in (("mean", result.mean), ("incremental", result.incremental)):
        if interval is not None:
            metrics.update(
                {
                    f"pilot {label} estimate": Decimal(repr(interval.estimate)),
                    f"pilot {label} standard error": Decimal(repr(interval.standard_error)),
                    f"pilot {label} lower": Decimal(repr(interval.lower)),
                    f"pilot {label} upper": Decimal(repr(interval.upper)),
                }
            )
    return outcome, metrics


def _ancestors(record: PackageRecord, root: UUID) -> set[UUID]:
    found = {root}
    # Verified links name only earlier trials, so one reverse pass is transitive.
    for trial in reversed(record.trials):
        if trial.trial_id in found:
            found.update(link.trial_id for link in trial.links)
    return found


def _pair(kind: type[Row | Metric], a: dict, b: dict) -> tuple:
    return tuple(kind(name, a.get(name), b.get(name)) for name in a | b)


def compare_packages(
    a: VerifiedPackage,
    b: VerifiedPackage,
    *,
    trial_a: UUID | None = None,
    trial_b: UUID | None = None,
    evaluation_a: str | None = None,
    evaluation_b: str | None = None,
) -> Comparison:
    """Compare one supplied trial from each verified development package.

    Args:
        a (VerifiedPackage): Package shown as side A, opened with `open_package`.
        b (VerifiedPackage): Package shown as side B, opened with `open_package`.
        trial_a (UUID, optional): Trial in A to compare, or its recorded selection.
        trial_b (UUID, optional): Trial in B to compare, or its recorded selection.
        evaluation_a (str, optional): Pilot artifact in A's trial's last attempt, or its sole one.
        evaluation_b (str, optional): Pilot artifact in B's trial's last attempt, or its sole one.

    Returns:
        (Comparison) Paired specification, input, version, outcome, and metric rows, with the
            lineage relation between the two compared trials.

    Raises:
        CompareError: If a requested trial or pilot artifact is absent, an omitted side selects
            no trial, pilot artifacts are ambiguous, or a diagnostic is outside its profile.
        PackageError: If a selected pilot artifact fails verified reading or typed decoding.
    """
    compared_a = _trial(a, trial_a, "A")
    compared_b = _trial(b, trial_b, "B")
    pilot_outcome_a, pilot_metrics_a = _pilot_evaluation(a, compared_a, evaluation_a, "A")
    pilot_outcome_b, pilot_metrics_b = _pilot_evaluation(b, compared_b, evaluation_b, "B")
    rows_a, rows_b = _rows(compared_a), _rows(compared_b)
    rows_a["outcome"].update(pilot_outcome_a)
    rows_b["outcome"].update(pilot_outcome_b)
    shared = _ancestors(a.record, compared_a.trial_id) & _ancestors(b.record, compared_b.trial_id)
    same_lineage = compared_a.lineage_id == compared_b.lineage_id
    relation = "same lineage" if same_lineage else "linked" if shared else "unrelated"
    return Comparison(
        _identity(a, compared_a),
        _identity(b, compared_b),
        **{section: _pair(Row, values, rows_b[section]) for section, values in rows_a.items()},
        metrics=_pair(
            Metric,
            _metrics(a, compared_a) | pilot_metrics_a,
            _metrics(b, compared_b) | pilot_metrics_b,
        ),
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
        "Outcome and metrics use the last supplied attempt of each compared trial. "
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


def render_comparison_json(comparison: Comparison) -> str:
    """Render schema 1 JSON with exact decimal strings and one final newline.

    Args:
        comparison (Comparison): Result of `compare_packages`.

    Returns:
        (str) JSON preserving row order, raw text, and computed comparison values.
    """
    metrics = []
    for metric in comparison.metrics:
        difference = metric.difference
        metrics.append(
            {
                "name": metric.name,
                "a": str(metric.a) if metric.a is not None else None,
                "b": str(metric.b) if metric.b is not None else None,
                "mark": metric.mark,
                "difference": str(difference) if difference is not None else None,
            }
        )
    payload = {
        "schema_version": 1,
        **{
            side: {
                "package": identity.package,
                "trial_id": str(identity.trial_id),
                "lineage_id": str(identity.lineage_id),
                "spec_id": identity.spec_id,
            }
            for side, identity in (("a", comparison.a), ("b", comparison.b))
        },
        **{
            section: [{"name": row.name, "a": row.a, "b": row.b, "mark": row.mark} for row in rows]
            for section, rows in (
                ("specification", comparison.specification),
                ("inputs", comparison.inputs),
                ("versions", comparison.versions),
                ("outcome", comparison.outcome),
            )
        },
        "metrics": metrics,
        "difference_count": len(comparison.differences),
        "relation": comparison.relation,
        "shared_trials": [str(trial) for trial in comparison.shared_trials],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
