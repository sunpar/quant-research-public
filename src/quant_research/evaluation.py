from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from quant_research.packages import PackageRecord

MINIMUM_OBSERVATIONS = 30
VERSION = "pilot-evaluation/v1"

type Verdict = Literal["supported", "null", "inconclusive"]
type Reason = Literal["above_minimum", "below_minimum", "spans_minimum", "unstable_subperiods"]

_VERDICTS: dict[Reason, Verdict] = {
    "above_minimum": "supported",
    "below_minimum": "null",
    "spans_minimum": "inconclusive",
    "unstable_subperiods": "inconclusive",
}


@dataclass(frozen=True)
class Profile:
    """Hold the procedure version and the parameters declared before evaluation."""

    minimum_effect: float
    subperiods: int
    alpha: float
    version: str = field(default=VERSION, init=False)

    def __post_init__(self) -> None:
        if not (
            all(
                isinstance(value, int | float) and not isinstance(value, bool)
                for value in (self.minimum_effect, self.subperiods, self.alpha)
            )
            and isinstance(self.subperiods, int)
            and math.isfinite(self.minimum_effect)
            and self.minimum_effect > 0
            and self.subperiods >= 2
            and 0 < self.alpha < 1
        ):
            raise ValueError(
                "Need numbers other than booleans, with finite minimum_effect > 0, integer "
                f"subperiods >= 2 and 0 < alpha < 1, got {self}"
            )
        object.__setattr__(self, "minimum_effect", float(self.minimum_effect))
        object.__setattr__(self, "alpha", float(self.alpha))


@dataclass(frozen=True)
class Interval:
    """Bound an estimate at the evaluation's Bonferroni-adjusted confidence."""

    estimate: float
    standard_error: float
    lower: float
    upper: float


@dataclass(frozen=True)
class Evaluation:
    """Report the verdict on the baseline intercept if one is supplied, else on the mean."""

    reason: Reason
    trials: int
    confidence: float
    mean: Interval
    incremental: Interval | None
    subperiods: tuple[float, ...]
    profile: Profile

    @property
    def verdict(self) -> Verdict:
        return _VERDICTS[self.reason]


@dataclass(frozen=True)
class Invalid:
    reason: Literal["length_mismatch", "nonfinite_value", "too_few_observations", "zero_variance"]
    detail: str
    profile: Profile


def _effect(returns: Sequence[float], baseline: Sequence[float] | None) -> tuple[float, float]:
    if baseline is None:
        return statistics.fmean(returns), 0.0
    slope, intercept = statistics.linear_regression(baseline, returns)
    return intercept, slope


def _t_quantile(probability: float, freedom: int) -> float:
    # Abramowitz and Stegun 26.7.5 expansion, since the stdlib has no t quantile.
    x = statistics.NormalDist().inv_cdf(probability)
    return (
        x
        + (x**3 + x) / 4 / freedom
        + (5 * x**5 + 16 * x**3 + 3 * x) / 96 / freedom**2
        + (3 * x**7 + 19 * x**5 + 17 * x**3 - 15 * x) / 384 / freedom**3
        + (79 * x**9 + 776 * x**7 + 1482 * x**5 - 1920 * x**3 - 945 * x) / 92160 / freedom**4
    )


def _interval(
    returns: Sequence[float], baseline: Sequence[float] | None, probability: float
) -> Interval:
    estimate, slope = _effect(returns, baseline)
    count = len(returns)
    if baseline is None:
        freedom = count - 1
        error = statistics.stdev(returns) / math.sqrt(count)
    else:
        freedom = count - 2
        squares = math.fsum(
            (value - estimate - slope * base) ** 2
            for value, base in zip(returns, baseline, strict=True)
        )
        center = statistics.fmean(baseline)
        spread = math.fsum((base - center) ** 2 for base in baseline)
        error = math.sqrt(squares / freedom) * math.sqrt(1 / count + center**2 / spread)
    margin = _t_quantile(probability, freedom) * error
    return Interval(estimate, error, estimate - margin, estimate + margin)


def _reason(tested: Interval, minimum_effect: float, estimates: tuple[float, ...]) -> Reason:
    if tested.upper < minimum_effect:
        return "below_minimum"
    if tested.lower <= minimum_effect:
        return "spans_minimum"
    if any(estimate <= 0 for estimate in estimates):
        return "unstable_subperiods"
    return "above_minimum"


def evaluate(
    returns: Sequence[float],
    history: PackageRecord,
    baseline: Sequence[float] | None = None,
    *,
    minimum_effect: float,
    subperiods: int = 2,
    alpha: float = 0.05,
    adjustment_trials: int | None = None,
) -> Evaluation | Invalid:
    """Test whether the selected candidate's returns clear a declared minimum effect.

    Args:
        returns (Sequence[float]): Equally spaced, non-overlapping period returns, oldest first.
        history (PackageRecord): Supplied measurements, counted unless adjustment_trials is set.
        baseline (Sequence[float] | None, optional): Comparison signal returns on the same periods.
        minimum_effect (float): Smallest per-period return worth having, declared in advance.
        subperiods (int, optional): Contiguous parts that must each show a positive effect.
        alpha (float, optional): Family-wise error rate across the supplied trials.
        adjustment_trials (int | None, optional): Declared candidate family size, otherwise the
            recorded trial count. Repeated measurements need not consume another candidate.

    Returns:
        (Evaluation | Invalid) Adjusted intervals with a verdict, or why the data cannot be tested.

    Raises:
        ValueError: If a parameter is a boolean or not a number, minimum_effect is not positive
            and finite, subperiods is not an integer of at least 2, alpha is outside (0, 1), or
            adjustment_trials is not a positive integer.
    """
    profile = Profile(minimum_effect, subperiods, alpha)
    if adjustment_trials is not None and (
        type(adjustment_trials) is not int or adjustment_trials < 1
    ):
        raise ValueError("adjustment_trials must be a positive integer")
    count = len(returns)
    if baseline is not None and len(baseline) != count:
        return Invalid(
            "length_mismatch", f"Baseline has {len(baseline)} values for {count} returns", profile
        )
    if not all(math.isfinite(value) for value in (*returns, *(baseline or ()))):
        return Invalid(
            "nonfinite_value", "Returns and baseline must not contain NaN or infinity", profile
        )
    needed = max(MINIMUM_OBSERVATIONS, 2 * subperiods)
    if count < needed:
        return Invalid(
            "too_few_observations", f"Need at least {needed} returns, got {count}", profile
        )
    bounds = [count * index // subperiods for index in range(subperiods + 1)]
    parts = [
        (returns[start:end], None if baseline is None else baseline[start:end])
        for start, end in pairwise(bounds)
    ]
    if statistics.variance(returns) == 0 or any(
        base is not None and statistics.variance(base) == 0 for _, base in parts
    ):
        return Invalid(
            "zero_variance", "Returns, or the baseline within a subperiod, are constant", profile
        )
    trials = len(history.trials) if adjustment_trials is None else adjustment_trials
    probability = 1 - alpha / (2 * trials)
    mean = _interval(returns, None, probability)
    incremental = None if baseline is None else _interval(returns, baseline, probability)
    tested = mean if incremental is None else incremental
    estimates = tuple(_effect(part, base)[0] for part, base in parts)
    return Evaluation(
        _reason(tested, minimum_effect, estimates),
        trials,
        1 - alpha / trials,
        mean,
        incremental,
        estimates,
        profile,
    )
