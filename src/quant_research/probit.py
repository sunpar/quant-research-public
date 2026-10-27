from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Literal

from quant_research import datasets, public_data
from quant_research.access import RunContext
from quant_research.datasets import ReadRecord, Refused
from quant_research.locators import Kind
from quant_research.records import format_time
from quant_research.reproduction import (
    Choice,
    Locator,
    LocatorPin,
    OpenQuestion,
    Specification,
    Target,
    Tolerance,
)

_ITERATIONS = 100
_HALVINGS = 60
# The fit takes its last step, so each fitted index lands within about the square of this distance.
_STEP = 1e-6
_SINGULAR = 1e-12
_ROOT_2 = math.sqrt(2)
_DENSITY = 1 / math.sqrt(2 * math.pi)


class ProbitError(ValueError):
    """Report data a probit cannot estimate, a failed fit, or a paper-case sample it cannot read."""


@dataclass(frozen=True)
class Probit:
    """A probit maximum-likelihood fit.

    coefficients start with the intercept, then one per regressor column. covariance is
    H^-1 S H^-1, with H minus the observed Hessian and S the Bartlett-weighted long-run
    covariance of the scores, without degrees-of-freedom scaling.
    """

    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    log_likelihood: float
    constant_log_likelihood: float
    observations: int

    @property
    def pseudo_r2(self) -> float:
        """Return Estrella's measure 1 - (ln Lu / ln Lc) ** (-(2 / n) ln Lc)."""
        constant = self.constant_log_likelihood
        return 1 - (self.log_likelihood / constant) ** (-2 / self.observations * constant)

    @property
    def t_statistics(self) -> tuple[float, ...]:
        """Return each coefficient over the square root of its covariance diagonal entry."""
        return tuple(
            coefficient / math.sqrt(self.covariance[index][index])
            for index, coefficient in enumerate(self.coefficients)
        )


def fit(outcomes: Sequence[int], regressors: Sequence[Sequence[float]], *, lags: int) -> Probit:
    """Fit P(y = 1) = Phi(a0 + a . x) by maximum likelihood with Newey-West covariance.

    Newton's method with the observed Hessian starts at intercept Phi^-1(mean outcome) and zero
    slopes, halves a step that does not raise the log-likelihood, and stops after a step that
    moves no observation's fitted index more than 1e-6, so the regressors' units do not change
    where it stops.

    Args:
        outcomes (Sequence[int]): Outcome of each observation, 0 or 1, in time order.
        regressors (Sequence[Sequence[float]]): Regressor row of each observation, all of one
            width, which may be zero. The fit adds the intercept.
        lags (int): Newey-West lag L, with Bartlett weights 1 - j / (L + 1).

    Returns:
        (Probit) The estimates, their covariance, and both log-likelihoods.

    Raises:
        ProbitError: If the lengths differ, the rows are ragged, an outcome is not 0 or 1, every
            outcome is equal, a regressor is not finite, there are fewer observations than
            parameters, lags is negative, the fit does not converge in 100 iterations, the
            Hessian is singular, or a fitted probability reaches 0 or 1.
    """
    if len(outcomes) != len(regressors):
        raise ProbitError("Outcomes and regressor rows differ in number")
    if len({len(row) for row in regressors}) > 1:
        raise ProbitError("Regressor rows differ in width")
    if any(outcome not in (0, 1) for outcome in outcomes):
        raise ProbitError("An outcome is not 0 or 1")
    if len(set(outcomes)) != 2:
        raise ProbitError("Every outcome is equal")
    design = [(1.0, *row) for row in regressors]
    if not all(math.isfinite(value) for row in design for value in row):
        raise ProbitError("A regressor is not finite")
    if len(design) < len(design[0]):
        raise ProbitError("There are fewer observations than parameters")
    if lags < 0:
        raise ProbitError("Lags must be nonnegative")
    ones = sum(outcomes)
    share = ones / len(outcomes)
    coefficients = [NormalDist().inv_cdf(share), *[0.0] * (len(design[0]) - 1)]
    current = _log_likelihood(outcomes, design, coefficients)
    for _ in range(_ITERATIONS):
        scores, information = _derivatives(outcomes, design, coefficients)
        gradient = [math.fsum(column) for column in zip(*scores, strict=True)]
        step = [
            math.fsum(entry * slope for entry, slope in zip(row, gradient, strict=True))
            for row in _inverse(information)
        ]
        if max(abs(_index(step, row)) for row in design) <= _STEP:
            coefficients = [value + move for value, move in zip(coefficients, step, strict=True)]
            break
        coefficients, current = _ascend(outcomes, design, coefficients, step, current)
    else:
        raise ProbitError(f"The fit does not converge in {_ITERATIONS} iterations")
    for row in design:
        index = _index(coefficients, row)
        if not (0 < _probability(1, index) < 1 and 0 < _probability(0, index) < 1):
            raise ProbitError("A fitted probability reaches 0 or 1")
    scores, information = _derivatives(outcomes, design, coefficients)
    inverse = _inverse(information)
    return Probit(
        tuple(coefficients),
        tuple(map(tuple, _product(_product(inverse, _long_run(scores, lags)), inverse))),
        _log_likelihood(outcomes, design, coefficients),
        ones * math.log(share) + (len(outcomes) - ones) * math.log(1 - share),
        len(outcomes),
    )


def _index(coefficients: Sequence[float], row: Sequence[float]) -> float:
    return math.fsum(value * entry for value, entry in zip(coefficients, row, strict=True))


def _probability(outcome: int, index: float) -> float:
    """Return Phi(index) for outcome 1 and 1 - Phi(index) for 0, each from its own tail."""
    return 0.5 * math.erfc((-index if outcome else index) / _ROOT_2)


def _log_likelihood(
    outcomes: Sequence[int], design: list[tuple[float, ...]], coefficients: Sequence[float]
) -> float:
    """Return the log-likelihood, or -inf when a probability underflows to 0."""
    probabilities = [
        _probability(outcome, _index(coefficients, row))
        for outcome, row in zip(outcomes, design, strict=True)
    ]
    return math.fsum(map(math.log, probabilities)) if all(probabilities) else -math.inf


def _derivatives(
    outcomes: Sequence[int], design: list[tuple[float, ...]], coefficients: Sequence[float]
) -> tuple[list[list[float]], list[list[float]]]:
    """Return each observation's score lambda x and minus the observed Hessian.

    lambda is phi / Phi for outcome 1 and -phi / (1 - Phi) for 0, and its derivative in the
    index is -lambda (lambda + index) for both, which gives the Hessian weights.
    """
    scores, weights = [], []
    for outcome, row in zip(outcomes, design, strict=True):
        index = _index(coefficients, row)
        density = _DENSITY * math.exp(-index * index / 2)
        ratio = (1 if outcome else -1) * density / _probability(outcome, index)
        scores.append([ratio * value for value in row])
        weights.append(ratio * (ratio + index))
    size = len(design[0])
    information = [
        [
            math.fsum(weight * row[i] * row[j] for weight, row in zip(weights, design, strict=True))
            for j in range(size)
        ]
        for i in range(size)
    ]
    return scores, information


def _ascend(
    outcomes: Sequence[int],
    design: list[tuple[float, ...]],
    coefficients: list[float],
    step: list[float],
    current: float,
) -> tuple[list[float], float]:
    """Return the first of the step's halvings that raises the log-likelihood, and its value."""
    scale = 1.0
    for _ in range(_HALVINGS):
        trial = [old + scale * move for old, move in zip(coefficients, step, strict=True)]
        if (value := _log_likelihood(outcomes, design, trial)) > current:
            return trial, value
        scale /= 2
    raise ProbitError("No fraction of the Newton step raises the log-likelihood")


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Invert a symmetric positive definite matrix by Gauss-Jordan elimination.

    A pivot at most 1e-12 of its own diagonal entry means the column is numerically a
    combination of the earlier ones, so the matrix counts as singular.
    """
    size = len(matrix)
    rows = [[*row, *(float(i == j) for j in range(size))] for i, row in enumerate(matrix)]
    for k in range(size):
        pivot = rows[k][k]
        if not pivot > _SINGULAR * matrix[k][k]:
            raise ProbitError("The Hessian is singular")
        rows[k] = [value / pivot for value in rows[k]]
        for i in range(size):
            if i != k:
                factor = rows[i][k]
                rows[i] = [
                    value - factor * other for value, other in zip(rows[i], rows[k], strict=True)
                ]
    return [row[size:] for row in rows]


def _product(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            math.fsum(a * b for a, b in zip(row, column, strict=True))
            for column in zip(*right, strict=True)
        ]
        for row in left
    ]


def _long_run(scores: list[list[float]], lags: int) -> list[list[float]]:
    """Return Gamma_0 + sum over j = 1..lags of (1 - j / (lags + 1)) (Gamma_j + Gamma_j')."""
    size = len(scores[0])

    def gamma(lag: int) -> list[list[float]]:
        return [
            [
                math.fsum(
                    later[i] * earlier[j]
                    for later, earlier in zip(scores[lag:], scores, strict=False)
                )
                for j in range(size)
            ]
            for i in range(size)
        ]

    total = gamma(0)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        cross = gamma(lag)
        total = [
            [value + weight * (cross[i][j] + cross[j][i]) for j, value in enumerate(row)]
            for i, row in enumerate(total)
        ]
    return total


CUTOFF = datetime(2026, 10, 22, tzinfo=UTC)
VINTAGES = {
    "alfred.gs10": "1939516d506ba3dc3a277e9941ddc470d884d589c04c19a6507ce5a01c059ecf",
    "alfred.tb3ms": "db1e34bafe4cc80d13792990150e6c27455e82fb456f20c72319e17434bce117",
    "alfred.usrecq": "6fa55811cdba98ce573ea0c6ced5349775973ee5f8056872be4863941dea3706",
}
_LOCATIONS = "6a49b4eb7b1aab159a3b26c1a797cfb5257454696bf89fc46a156f37f4530a2f"
_CHUNKS = "ec99feefb2b4093b54d55da7395e2d2070a10a45f37126c7ae786043ef27fde8"
_QUARTERS = 145
# Table 2's SPREAD row: horizon k -> (pseudo R2, t-statistic).
_TABLE_2_SPREAD = {
    1: (0.071, -2.71),
    2: (0.211, -4.21),
    3: (0.271, -4.71),
    4: (0.296, -4.57),
    5: (0.256, -3.87),
    6: (0.149, -4.13),
    7: (0.078, -3.02),
    8: (0.031, -1.63),
}
_INPUTS = ("model", "sample", "spread", "bill", "recession")


def _at(kind: Literal["passage"] | Kind, start: int, end: int) -> Locator:
    """Name a span of the conversion; a passage uses the chunk generation in the same form."""
    generation = _CHUNKS if kind == "passage" else _LOCATIONS
    return Locator(kind, f"{generation}:{kind}:{start}-{end}")


_FORECAST = _at("equation", 10223, 10264)
_LIKELIHOOD = _at("equation", 10431, 10525)
_CODING = _at("equation", 10626, 10760)
_MODEL = _at("equation", 19385, 19431)
_TABLE_2 = _at("table", 19452, 20940)
_TABLE_1 = _at("table", 15867, 17101)
_T_METHOD = _at("claim", 12480, 12692)
_PSEUDO_R2_SOURCE = _at("passage", 11197, 11962)
_NBER_DATES = _at("passage", 15326, 15756)
_SAMPLE = _at("passage", 17211, 18025)
_SPREAD = _at("passage", 71067, 71137)
_BILL = _at("passage", 71139, 71196)
_AVERAGES = _at("passage", 73243, 73827)
_DATA_SOURCE = _at("passage", 73844, 74268)
_FOOTNOTE_3 = _at("passage", 75235, 76011)
_ESTRELLA_1995 = _at("passage", 84229, 84393)
_ESTRELLA_9716 = Locator(
    "passage",
    "frbny-research-paper-9716.pdf@"
    "bea3994336c69d6a6099a3d2816c622f36ab5a6c6b80f1a28fccd50990b62053:page-8",
)

SPECIFICATION = Specification(
    paper=(
        "Arturo Estrella and Frederic S. Mishkin, Predicting U.S. Recessions: Financial "
        "Variables as Leading Indicators, Federal Reserve Bank of New York Research Paper 9609, "
        "May 1996, the working paper of the 1998 Review of Economics and Statistics article; "
        "Table 2, SPREAD row"
    ),
    data=(
        ", ".join(f"{name} vintage {vintage}" for name, vintage in VINTAGES.items())
        + f", each read through {datasets.ADAPTER} at cutoff {format_time(CUTOFF)}"
    ),
    basis="adapted",
    choices=(
        Choice(
            "model",
            "A probit P(R_{t+k} = 1) = Phi(a0 + a1 SPREAD_t), fit by maximum likelihood, one "
            "equation for each horizon k = 1 to 8.",
            (_FORECAST, _LIKELIHOOD, _MODEL, _TABLE_2),
        ),
        Choice(
            "sample",
            "Quarters 1959Q1 to 1995Q1; each equation pairs SPREAD_t with R_{t+k} for t from "
            "1959Q1 through 1995Q1 minus k quarters, so n = 145 - k.",
            (_SAMPLE,),
        ),
        Choice(
            "spread",
            "SPREAD_t is the mean over the quarter's three months of GS10 minus the "
            "bond-equivalent TB3MS yield, in percentage points, each month the latest ALFRED "
            "revision at the cutoff.",
            (_TABLE_1, _SPREAD, _AVERAGES),
        ),
        Choice(
            "bill",
            "Each monthly TB3MS discount rate d, in percent, becomes the bond-equivalent yield "
            "100 * 365 (d / 100) / (360 - 91 (d / 100)), a fixed 91-day maturity applied to the "
            "monthly average.",
            (_BILL, _DATA_SOURCE),
        ),
        Choice(
            "recession",
            "R_t is USRECQ_t, which is 1 from the quarter after an NBER peak through the trough.",
            (_CODING, _NBER_DATES),
        ),
        Choice(
            "pseudo_r2",
            "Estrella's measure 1 - (ln Lu / ln Lc) ^ (-(2 / n) ln Lc), with Lc the "
            "constant-only likelihood, in natural logs.",
            (_PSEUDO_R2_SOURCE, _ESTRELLA_1995, _ESTRELLA_9716),
        ),
        Choice(
            "t_statistic",
            "a1 over its Newey-West standard error, with Bartlett weights, lag k - 1, and the "
            "sandwich H^-1 S H^-1 of the probit scores with the observed Hessian.",
            (_T_METHOD, _FOOTNOTE_3),
            "newey_west_lag",
        ),
    ),
    questions=(
        OpenQuestion(
            "newey_west_lag",
            "missing",
            "The paper names the Newey-West adjustment of the probit first-order conditions but "
            "not its lag, kernel, or Hessian form, and Estrella and Rodrigues' FRBNY Staff "
            "Report 39 compares several such estimators without saying which this paper used.",
            (_T_METHOD, _FOOTNOTE_3),
        ),
    ),
    targets=tuple(
        target
        for k, (pseudo_r2, t_stat) in _TABLE_2_SPREAD.items()
        for target in (
            Target(
                f"spread.k{k}.pseudo_r2",
                pseudo_r2,
                Tolerance(0.0005),
                (_TABLE_2,),
                depends_on=(*_INPUTS, "pseudo_r2"),
            ),
            Target(
                f"spread.k{k}.t_stat",
                t_stat,
                Tolerance(0.005),
                (_TABLE_2,),
                depends_on=(*_INPUTS, "t_statistic"),
            ),
        )
    ),
    pin=LocatorPin(_LOCATIONS, "09013031e8c0d9337b7c7298a777b887ceaa0a70b6f3f42b63fa0d517e88bcb9"),
)


@dataclass(frozen=True)
class Observed:
    """One observation per target of SPECIFICATION and the adapter's records of the reads."""

    observations: dict[str, float]
    reads: tuple[ReadRecord, ...]


def observe(root: Path, context: RunContext) -> Observed:
    """Fit Table 2's SPREAD equations on the paper case's series.

    The reads go through the dataset adapter, and their vintages are not checked here, so the
    caller freezes SPECIFICATION first and checks each read's vintage against VINTAGES after.

    Args:
        root (Path): Public data root whose catalog approves the three series.
        context (RunContext): Trusted run authority, at CUTOFF for the live run.

    Returns:
        (Observed) The pseudo R2 and slope t-statistic of each horizon and the three reads.

    Raises:
        ProbitError: If a read is refused, a month of GS10 or TB3MS or a quarter of USRECQ from
            1959-01 through 1995-03 is missing or not a number, or a fit fails.
    """
    catalog = public_data.load_catalog(root)
    series, reads = {}, []
    for name in VINTAGES:
        result = datasets.read(catalog, root / "datasets", name, context)
        if isinstance(result, Refused):
            raise ProbitError(f"Cannot read {name!r}: {result.message}")
        series[name] = {row.key: row.values.get("value") for row in result.rows}
        reads.append(result.record)
    months = [date(1959 + month // 12, month % 12 + 1, 1) for month in range(3 * _QUARTERS)]
    monthly = []
    for month in months:
        discount = _number(series, "alfred.tb3ms", month) / 100
        bill = 100 * 365 * discount / (360 - 91 * discount)
        monthly.append(_number(series, "alfred.gs10", month) - bill)
    spreads = [math.fsum(monthly[start : start + 3]) / 3 for start in range(0, len(monthly), 3)]
    recessions = [_number(series, "alfred.usrecq", quarter) for quarter in months[::3]]
    observations = {}
    for k in _TABLE_2_SPREAD:
        probit = fit(recessions[k:], [[spread] for spread in spreads[:-k]], lags=k - 1)
        observations[f"spread.k{k}.pseudo_r2"] = probit.pseudo_r2
        observations[f"spread.k{k}.t_stat"] = probit.t_statistics[1]
    return Observed(observations, tuple(reads))


def _number(series: dict[str, dict[str, object]], name: str, day: date) -> float:
    try:
        return float(series[name][day.isoformat()])
    except (KeyError, TypeError, ValueError) as error:
        raise ProbitError(f"{name!r} has no number for {day.isoformat()}") from error
