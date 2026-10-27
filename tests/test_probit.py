from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import NormalDist

import pytest

from quant_research import probit
from quant_research.access import issue_context
from quant_research.probit import SPECIFICATION, VINTAGES, ProbitError, fit, observe
from quant_research.public_data import approve, ingest, pull, validate
from quant_research.reproduction import freeze

NORMAL = NormalDist()
NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
CONTEXT = issue_context(clearances=(), cutoff=NOW, generation="probit/v1", providers=())
X = [-1.4, -0.9, -0.3, 0.2, 0.5, 0.8, 1.1, 1.7, 2.3, -2.0, 0.0, 1.4]
Y = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
PRINTED = {
    **{
        f"spread.k{k}.pseudo_r2": value
        for k, value in enumerate((0.071, 0.211, 0.271, 0.296, 0.256, 0.149, 0.078, 0.031), 1)
    },
    **{
        f"spread.k{k}.t_stat": value
        for k, value in enumerate((-2.71, -4.21, -4.71, -4.57, -3.87, -4.13, -3.02, -1.63), 1)
    },
}
SAMPLE = [date(1959 + month // 12, month % 12 + 1, 1) for month in range(435)]
SYNTHETIC = random.Random(22)
# The series run past both ends of the sample, so a wrong window reads other values.
MONTHS = [date(1957 + month // 12, month % 12 + 1, 1) for month in range(41 * 12)]
BONDS = {month: f"{SYNTHETIC.uniform(3, 9):.2f}" for month in MONTHS}
BILLS = {month: f"{SYNTHETIC.uniform(1, 9):.2f}" for month in MONTHS}
RECESSIONS = {quarter: str(int(SYNTHETIC.random() < 0.3)) for quarter in MONTHS[::3]}


def alfred(values: dict[date, str]) -> bytes:
    observations = [
        {
            "date": day.isoformat(),
            "realtime_start": "2020-01-02",
            "realtime_end": "9999-12-31",
            "value": value,
        }
        for day, value in values.items()
    ]
    response = {
        "count": len(observations),
        "observations": observations,
        "realtime_end": "9999-12-31",
        "realtime_start": "1776-07-04",
    }
    return json.dumps(response).encode()


def paper_root(root: Path, *, bonds=BONDS, bills=BILLS, recessions=RECESSIONS, approved=VINTAGES):
    for name, values in zip(VINTAGES, (bonds, bills, recessions), strict=True):
        data = alfred(values)
        pull(root, name, now=NOW, api_key="k", fetch=lambda url, data=data: data)
        ingest(root, name, now=NOW)
        if name in approved:
            validate(root, name, actor="researcher", evidence=b"notes", summary="checked", now=NOW)
            approve(root, name, actor="researcher", now=NOW)
    return root


def log_likelihood(a: float, b: float) -> float:
    return sum(
        math.log(NORMAL.cdf(a + b * x) if y else 1 - NORMAL.cdf(a + b * x))
        for x, y in zip(X, Y, strict=True)
    )


def test_a_saturated_probit_matches_the_group_shares():
    """Ten observations at x = 0 with 2 ones and ten at x = 1 with 7 ones."""
    result = fit([1] * 2 + [0] * 8 + [1] * 7 + [0] * 3, [[0.0]] * 10 + [[1.0]] * 10, lags=0)
    low, high = NORMAL.inv_cdf(0.2), NORMAL.inv_cdf(0.7)
    unrestricted = sum(10 * (p * math.log(p) + (1 - p) * math.log(1 - p)) for p in (0.2, 0.7))
    constant = 9 * math.log(9 / 20) + 11 * math.log(11 / 20)
    # At a saturated MLE the sandwich is the inverse information, and each group's index has
    # the delta-method variance p(1 - p) / (n phi(Phi^-1(p))^2).
    low_variance, high_variance = (
        p * (1 - p) / (10 * NORMAL.pdf(NORMAL.inv_cdf(p)) ** 2) for p in (0.2, 0.7)
    )
    assert result.coefficients == pytest.approx((low, high - low), abs=1e-12)
    assert result.log_likelihood == pytest.approx(unrestricted, rel=1e-12)
    assert result.constant_log_likelihood == pytest.approx(constant, rel=1e-12)
    assert result.pseudo_r2 == pytest.approx(
        1 - (unrestricted / constant) ** (-(2 / 20) * constant), rel=1e-12
    )
    assert result.covariance[0] == pytest.approx((low_variance, -low_variance), rel=1e-12)
    assert result.covariance[1] == pytest.approx(
        (-low_variance, low_variance + high_variance), rel=1e-12
    )
    assert result.t_statistics[1] == pytest.approx(
        (high - low) / math.sqrt(low_variance + high_variance), rel=1e-12
    )
    assert result.observations == 20


@pytest.mark.parametrize(("lags", "variance"), [(0, math.pi / 8), (1, math.pi / 32)])
def test_constant_only_covariance_follows_the_scores_by_hand(lags, variance):
    """Outcomes 1, 0, 1, 0 give intercept 0, scores +-2 phi(0), and H = 16 phi(0)^2.

    Gamma_0 = 16 phi(0)^2 and Gamma_1 = -12 phi(0)^2, so S is 16 phi(0)^2 at lag 0 and
    16 phi(0)^2 + (1/2)(2)(-12 phi(0)^2) = 4 phi(0)^2 at lag 1, and S / H^2 with
    phi(0)^2 = 1 / (2 pi) gives pi / 8 and pi / 32.
    """
    result = fit([1, 0, 1, 0], [[]] * 4, lags=lags)
    assert result.coefficients == pytest.approx((0.0,), abs=1e-15)
    assert result.covariance[0][0] == pytest.approx(variance, rel=1e-12)
    assert result.pseudo_r2 == pytest.approx(0.0, abs=1e-15)


def test_the_fit_is_a_maximum_where_the_score_vanishes():
    result = fit(Y, [[x] for x in X], lags=1)
    a, b = result.coefficients
    ratios = [
        NORMAL.pdf(a + b * x) / (NORMAL.cdf(a + b * x) if y else NORMAL.cdf(a + b * x) - 1)
        for x, y in zip(X, Y, strict=True)
    ]
    assert sum(ratios) == pytest.approx(0.0, abs=1e-9)
    assert sum(ratio * x for ratio, x in zip(ratios, X, strict=True)) == pytest.approx(
        0.0, abs=1e-9
    )
    assert result.log_likelihood == pytest.approx(log_likelihood(a, b), rel=1e-12)
    for da, db in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
        assert log_likelihood(a + da, b + db) < log_likelihood(a, b)


@pytest.mark.parametrize("scale", [1e6, 1e-10])
def test_the_fit_does_not_depend_on_the_regressor_units(scale):
    """Rescaling the regressor divides the slope by the scale and leaves the maximum unchanged."""
    outcomes, values = [0, 0, 1, 0, 1, 1], [-3, -2, -1, 1, 2, 3]
    unscaled = fit(outcomes, [[x] for x in values], lags=0)
    scaled = fit(outcomes, [[x * scale] for x in values], lags=0)
    a, b = unscaled.coefficients
    ratios = [
        NORMAL.pdf(a + b * x) / (NORMAL.cdf(a + b * x) if y else NORMAL.cdf(a + b * x) - 1)
        for x, y in zip(values, outcomes, strict=True)
    ]
    assert sum(ratios) == pytest.approx(0.0, abs=1e-9)
    assert sum(ratio * x for ratio, x in zip(ratios, values, strict=True)) == pytest.approx(
        0.0, abs=1e-9
    )
    assert scaled.log_likelihood == pytest.approx(unscaled.log_likelihood, rel=1e-9)
    assert scaled.coefficients[1] * scale == pytest.approx(b, rel=1e-9)


@pytest.mark.parametrize("lags", [0, 2])
def test_the_covariance_is_the_bartlett_sandwich_with_the_observed_hessian(lags):
    """H sums lambda (lambda + z) x x', and S adds weighted score autocovariances to Gamma_0."""
    result = fit(Y, [[x] for x in X], lags=lags)
    a, b = result.coefficients
    scores, weights = [], []
    for x, y in zip(X, Y, strict=True):
        z = a + b * x
        ratio = NORMAL.pdf(z) / (NORMAL.cdf(z) if y else NORMAL.cdf(z) - 1)
        scores.append((ratio, ratio * x))
        weights.append(ratio * (ratio + z))
    h00, h01, h11 = (
        sum(w * x**power for w, x in zip(weights, X, strict=True)) for power in (0, 1, 2)
    )
    determinant = h00 * h11 - h01**2
    inverse = ((h11 / determinant, -h01 / determinant), (-h01 / determinant, h00 / determinant))

    def long_run(i, j):
        total = sum(g[i] * g[j] for g in scores)
        for lag in range(1, lags + 1):
            total += (1 - lag / (lags + 1)) * sum(
                scores[t][i] * scores[t - lag][j] + scores[t - lag][i] * scores[t][j]
                for t in range(lag, len(scores))
            )
        return total

    for i in range(2):
        assert result.covariance[i] == pytest.approx(
            [
                sum(
                    inverse[i][m] * long_run(m, n) * inverse[n][j]
                    for m in range(2)
                    for n in range(2)
                )
                for j in range(2)
            ],
            rel=1e-9,
        )


@pytest.mark.parametrize(
    ("outcomes", "regressors", "lags"),
    [
        pytest.param([0, 1, 1], [[0.1], [0.4]], 0, id="mismatched-lengths"),
        pytest.param([0, 1, 1], [[0.1], [0.4, 0.2], [0.3]], 0, id="ragged-rows"),
        pytest.param([0, 2, 1, 0], [[0.1], [0.4], [0.2], [0.9]], 0, id="outcome-not-binary"),
        pytest.param([1, 1, 1], [[0.1], [0.4], [0.2]], 0, id="all-outcomes-equal"),
        pytest.param([], [], 0, id="no-observations"),
        pytest.param([0, 1, 1, 0], [[0.1], [math.nan], [0.2], [0.9]], 0, id="nan-regressor"),
        pytest.param([0, 1, 1, 0], [[0.1], [math.inf], [0.2], [0.9]], 0, id="inf-regressor"),
        pytest.param([0, 1], [[0.1, 0.3], [0.4, 0.2]], 0, id="fewer-observations-than-parameters"),
        pytest.param([0, 1, 1, 0], [[0.1], [0.4], [0.2], [0.9]], -1, id="negative-lags"),
        pytest.param(Y, [[x, 3 * x] for x in X], 0, id="collinear-regressors"),
        pytest.param(Y, [[1.0] for _ in X], 0, id="regressor-equal-to-the-intercept"),
        pytest.param([0, 0, 1, 1], [[0.0], [0.0], [1.0], [1.0]], 0, id="perfect-separation"),
        pytest.param([0, 1, 0, 1, 1, 1], [[0.0]] * 4 + [[1.0]] * 2, 0, id="quasi-separation"),
        pytest.param(
            [0, 1, 0, 1, 1, 0, 0, 1, 1],
            [[-1.0], [-0.5], [0.0], [0.5], [1.0], [-0.8], [0.3], [0.7], [50.0]],
            0,
            id="tail-underflow",
        ),
    ],
)
def test_fit_refuses_data_it_cannot_estimate(outcomes, regressors, lags):
    with pytest.raises(ProbitError):
        fit(outcomes, regressors, lags=lags)


def test_a_fit_stopped_before_convergence_refuses(monkeypatch):
    monkeypatch.setattr(probit, "_ITERATIONS", 1)
    with pytest.raises(ProbitError):
        fit(Y, [[x] for x in X], lags=0)


def test_observe_fits_each_horizon_on_the_quarterly_series(tmp_path):
    observed = observe(paper_root(tmp_path), CONTEXT)
    rate = [float(BILLS[month]) / 100 for month in SAMPLE]
    monthly = [
        float(BONDS[month]) - 100 * 365 * d / (360 - 91 * d)
        for month, d in zip(SAMPLE, rate, strict=True)
    ]
    spreads = [sum(monthly[3 * quarter : 3 * quarter + 3]) / 3 for quarter in range(145)]
    recessions = [int(RECESSIONS[month]) for month in SAMPLE[::3]]
    assert set(observed.observations) == {target.name for target in SPECIFICATION.targets}
    assert [read.dataset for read in observed.reads] == list(VINTAGES)
    for k in range(1, 9):
        pairs = [(spreads[t], recessions[t + k]) for t in range(145 - k)]
        expected = fit([r for _, r in pairs], [[s] for s, _ in pairs], lags=k - 1)
        assert observed.observations[f"spread.k{k}.pseudo_r2"] == pytest.approx(
            expected.pseudo_r2, rel=1e-9
        )
        assert observed.observations[f"spread.k{k}.t_stat"] == pytest.approx(
            expected.t_statistics[1], rel=1e-9
        )


def test_observe_reads_only_the_sample_months(tmp_path):
    window = {
        name: {day: value for day, value in values.items() if day in SAMPLE}
        for name, values in (("bonds", BONDS), ("bills", BILLS), ("recessions", RECESSIONS))
    }
    trimmed = observe(paper_root(tmp_path / "window", **window), CONTEXT)
    assert trimmed.observations == observe(paper_root(tmp_path / "full"), CONTEXT).observations


@pytest.mark.parametrize(
    "damage",
    ["missing-bond-month", "null-bill-month", "missing-last-quarter", "unapproved-recessions"],
)
def test_observe_refuses_a_sample_it_cannot_read_whole(tmp_path, damage):
    bonds, bills, recessions, approved = dict(BONDS), dict(BILLS), dict(RECESSIONS), VINTAGES
    match damage:
        case "missing-bond-month":
            del bonds[date(1977, 5, 1)]
        case "null-bill-month":
            bills[date(1990, 2, 1)] = "."
        case "missing-last-quarter":
            del recessions[date(1995, 1, 1)]
        case "unapproved-recessions":
            approved = ("alfred.gs10", "alfred.tb3ms")
    root = paper_root(tmp_path, bonds=bonds, bills=bills, recessions=recessions, approved=approved)
    with pytest.raises(ProbitError):
        observe(root, CONTEXT)


def test_the_printed_values_classify_by_the_frozen_tolerances_and_question():
    reproduction = (
        freeze(replace(SPECIFICATION, pin=None))
        .record("printed", PRINTED)
        .record("inside", {**PRINTED, "spread.k4.pseudo_r2": 0.2964})
        .record("outside", {**PRINTED, "spread.k4.pseudo_r2": 0.2966})
    )
    printed, inside, outside = (
        {verdict.target: (verdict.outcome, verdict.questions) for verdict in result.verdicts}
        for result in reproduction.classify()
    )
    assert printed == {
        name: ("reproduced_on_adapted_data", ())
        if name.endswith("pseudo_r2")
        else ("inconclusive", ("newey_west_lag",))
        for name in PRINTED
    }
    assert inside == printed
    assert outside == {**printed, "spread.k4.pseudo_r2": ("not_reproduced_on_adapted_data", ())}
    assert all(
        f"{name} vintage {vintage}" in SPECIFICATION.data for name, vintage in VINTAGES.items()
    )
