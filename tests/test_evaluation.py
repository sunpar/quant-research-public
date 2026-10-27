from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from quant_research.evaluation import Evaluation, Invalid, Profile, evaluate
from quant_research.packages import (
    EnvironmentRecord,
    ExecutionAttempt,
    ExposureEvent,
    LineageLink,
    Outcome,
    PackageRecord,
    Specification,
    TrialRecord,
)
from quant_research.records import canonical_bytes

TIME = datetime(2026, 10, 21, 12, tzinfo=UTC)
SEED = 0
NEGLIGIBLE_EFFECT = 1e-9
QUANTILE_TOLERANCE = 1e-6
BASELINE_PATTERN = (1, 1, -1, -1)
NOISE_PATTERN = (1, -1, 1, -1)
EXACT_T_QUANTILE = {
    (0.975, 39): 2.022690920036759,
    (0.995, 39): 2.707913183517661,
    (0.9975, 39): 2.9756087577929935,
    (0.975, 46): 2.012895598919431,
    (0.975, 47): 2.0117405137297673,
    (0.975, 249): 1.969536867640347,
    (0.999875, 249): 3.715946989183938,
}
SPEC = Specification(
    "Does the candidate clear its minimum effect?",
    "hand-authored/v1",
    "pilot-evaluation/v1",
    "synthetic/v1",
    "test-authoring/v1",
    ("0" * 64,),
    "0" * 64,
    "0" * 64,
    EnvironmentRecord("cpython", "3.12.13", "cpython-312", "test-platform", "uv test", "0" * 64),
)


def attempt(run_id: UUID, predecessor: UUID | None, order: int) -> ExecutionAttempt:
    started = TIME + timedelta(minutes=order)
    outcome = Outcome("successful", "Supplied outcome")
    return ExecutionAttempt(
        uuid4(), run_id, predecessor, started, started + timedelta(seconds=1), outcome
    )


def history(trials: int, *, retries: int = 0, repeats: int = 0) -> PackageRecord:
    records = []
    for _ in range(trials):
        attempts = [attempt(uuid4(), None, 0)]
        for _ in range(retries):
            attempts.append(attempt(attempts[-1].run_id, attempts[-1].id, len(attempts)))
        for _ in range(repeats):
            attempts.append(attempt(uuid4(), None, len(attempts)))
        records.append(TrialRecord(uuid4(), uuid4(), SPEC, tuple(attempts)))
    return PackageRecord(records[-1].trial_id, tuple(records))


def alternating(mean: float, swing: float, length: int) -> list[float]:
    return [mean + swing * (-1) ** index for index in range(length)]


def test_an_empty_history_keeps_the_existing_invalid_input_result():
    result = evaluate([], PackageRecord(None, ()), minimum_effect=0.001)
    assert isinstance(result, Invalid)
    assert result.reason == "too_few_observations"


@pytest.mark.parametrize("recorded, declared", [(1, 5), (5, 2)])
def test_a_declared_candidate_family_is_independent_of_measurement_count(recorded, declared):
    result = evaluate(
        alternating(0.01, 0.01, 40),
        history(recorded),
        minimum_effect=0.005,
        adjustment_trials=declared,
    )
    assert isinstance(result, Evaluation)
    assert result.trials == declared
    assert result.confidence == 1 - 0.05 / declared
    assert result.profile.alpha == 0.05


@pytest.mark.parametrize("count", [True, False, 0, -1, 1.0, "2"])
def test_an_explicit_adjustment_needs_a_positive_integer(count):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate(
            alternating(0.01, 0.01, 40), history(1), minimum_effect=0.005, adjustment_trials=count
        )


def blocks(center: float, swing: float, pattern: tuple[int, ...], length: int) -> list[float]:
    return [center + swing * pattern[index % len(pattern)] for index in range(length)]


def orthogonal_mean_error(length: int, *swings: float) -> float:
    return math.sqrt(sum(swing * swing for swing in swings) / (length - 1))


def intercept_error(length: int, center: float, swing: float, noise: float) -> float:
    residual = noise * math.sqrt(length / (length - 2))
    return residual * math.sqrt(1 / length + center**2 / (length * swing**2))


def binomial_margin(rate: float, count: int) -> float:
    return 3 * math.sqrt(rate * (1 - rate) / count)


def assert_interval(actual, estimate: float, error: float, quantile: float) -> None:
    width = quantile * error
    assert actual.estimate == pytest.approx(estimate, rel=1e-9)
    assert actual.standard_error == pytest.approx(error, rel=1e-9)
    assert actual.lower == pytest.approx(estimate - width, abs=QUANTILE_TOLERANCE * width)
    assert actual.upper == pytest.approx(estimate + width, abs=QUANTILE_TOLERANCE * width)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "mean, swing, minimum_effect, reason, verdict",
    [
        pytest.param(0.01, 0.01, 0.005, "above_minimum", "supported", id="lower above minimum"),
        pytest.param(-0.01, 0.01, 0.005, "below_minimum", "null", id="significantly negative"),
        pytest.param(0.01, 0.01, 0.02, "below_minimum", "null", id="inside zero and minimum"),
        pytest.param(0.002, 0.01, 0.01, "below_minimum", "null", id="spans zero below minimum"),
        pytest.param(
            0.002, 0.01, 0.003, "spans_minimum", "inconclusive", id="spans zero and minimum"
        ),
        pytest.param(0.01, 0.01, 0.008, "spans_minimum", "inconclusive", id="lower under minimum"),
    ],
)
def test_mean_interval_and_verdict_follow_the_closed_form(
    mean, swing, minimum_effect, reason, verdict
):
    result = evaluate(alternating(mean, swing, 40), history(1), minimum_effect=minimum_effect)
    assert isinstance(result, Evaluation)
    assert (result.reason, result.verdict) == (reason, verdict)
    assert result.trials == 1
    assert result.confidence == pytest.approx(0.95)
    assert_interval(
        result.mean, mean, orthogonal_mean_error(40, swing), EXACT_T_QUANTILE[0.975, 39]
    )
    assert result.incremental is None
    assert result.subperiods == pytest.approx((mean, mean), rel=1e-9)


def test_bonferroni_widens_the_interval_with_the_trial_count():
    returns = alternating(0.01, 0.025, 40)
    alone = evaluate(returns, history(1), minimum_effect=NEGLIGIBLE_EFFECT)
    disclosed = evaluate(returns, history(10), minimum_effect=NEGLIGIBLE_EFFECT)
    assert (alone.trials, disclosed.trials) == (1, 10)
    assert alone.confidence == pytest.approx(0.95)
    assert disclosed.confidence == pytest.approx(0.995)
    assert_interval(alone.mean, 0.01, orthogonal_mean_error(40, 0.025), EXACT_T_QUANTILE[0.975, 39])
    assert_interval(
        disclosed.mean, 0.01, orthogonal_mean_error(40, 0.025), EXACT_T_QUANTILE[0.9975, 39]
    )
    assert (alone.reason, alone.verdict) == ("above_minimum", "supported")
    assert (disclosed.reason, disclosed.verdict) == ("spans_minimum", "inconclusive")


def test_attempts_and_runs_inside_a_trial_do_not_count_as_trials():
    returns = alternating(0.01, 0.01, 40)
    retried = evaluate(returns, history(1, retries=2, repeats=2), minimum_effect=NEGLIGIBLE_EFFECT)
    five = evaluate(returns, history(5), minimum_effect=NEGLIGIBLE_EFFECT)
    assert (retried.trials, five.trials) == (1, 5)
    assert retried.confidence == pytest.approx(0.95)
    assert five.confidence == pytest.approx(0.99)
    assert_interval(
        retried.mean, 0.01, orthogonal_mean_error(40, 0.01), EXACT_T_QUANTILE[0.975, 39]
    )
    assert_interval(five.mean, 0.01, orthogonal_mean_error(40, 0.01), EXACT_T_QUANTILE[0.995, 39])


def test_every_supplied_trial_counts_whatever_its_lineage_outcome_or_exposures():
    record = history(4)
    first = record.trials[0]
    link = LineageLink(first.trial_id, SPEC.spec_id, "variant", "Next parameter")
    failure = Outcome("failed", "Supplied execution failure")
    trials = [first] + [
        replace(trial, lineage_id=first.lineage_id, links=(link,)) for trial in record.trials[1:]
    ]
    trials[1] = replace(
        trials[1], attempts=tuple(replace(item, outcome=failure) for item in trials[1].attempts)
    )
    exposures = tuple(
        ExposureEvent(
            uuid4(), "researcher", TIME, "0" * 64, "synthetic/v1", TIME, TIME, "result", "Prior"
        )
        for _ in range(3)
    )
    shared = replace(record, trials=tuple(trials), exposures=exposures)
    result = evaluate(alternating(0.01, 0.01, 40), shared, minimum_effect=NEGLIGIBLE_EFFECT)
    assert result.trials == 4
    assert result.confidence == pytest.approx(1 - 0.05 / 4)


@pytest.mark.smoke
def test_lucky_noise_winner_is_supported_alone_and_inconclusive_with_its_history():
    rng = random.Random(SEED)
    candidates = [[rng.gauss(0.0, 1.0) for _ in range(250)] for _ in range(200)]
    winner = max(candidates, key=statistics.fmean)
    estimate = statistics.fmean(winner)
    error = statistics.stdev(winner) / math.sqrt(250)
    halves = (statistics.fmean(winner[:125]), statistics.fmean(winner[125:]))
    disclosed = evaluate(winner, history(200), minimum_effect=NEGLIGIBLE_EFFECT)
    assert isinstance(disclosed, Evaluation)
    assert disclosed.trials == 200
    assert disclosed.confidence == pytest.approx(1 - 0.05 / 200)
    assert_interval(disclosed.mean, estimate, error, EXACT_T_QUANTILE[0.999875, 249])
    naive_lower = (
        disclosed.mean.estimate - EXACT_T_QUANTILE[0.975, 249] * disclosed.mean.standard_error
    )
    assert naive_lower > 0 >= disclosed.mean.lower
    assert disclosed.subperiods == pytest.approx(halves, rel=1e-9)
    assert (disclosed.reason, disclosed.verdict) == ("spans_minimum", "inconclusive")
    alone = evaluate(winner, history(1), minimum_effect=NEGLIGIBLE_EFFECT)
    assert_interval(alone.mean, estimate, error, EXACT_T_QUANTILE[0.975, 249])
    assert (alone.reason, alone.verdict) == ("above_minimum", "supported")


def test_selection_from_twenty_noise_trials_is_calibrated_under_repeated_search():
    rng = random.Random(SEED)
    alone, disclosed = history(1), history(20)
    naive = adjusted = 0
    for _ in range(400):
        winner = max(
            ([rng.gauss(0.0, 1.0) for _ in range(100)] for _ in range(20)), key=statistics.fmean
        )
        naive += evaluate(winner, alone, minimum_effect=NEGLIGIBLE_EFFECT).verdict == "supported"
        adjusted += (
            evaluate(winner, disclosed, minimum_effect=NEGLIGIBLE_EFFECT).verdict == "supported"
        )
    naive_rate = 1 - (1 - 0.025) ** 20
    adjusted_rate = 20 * 0.05 / (2 * 20)
    assert naive / 400 >= naive_rate - binomial_margin(naive_rate, 400)
    assert adjusted / 400 <= adjusted_rate + binomial_margin(adjusted_rate, 400)


@pytest.mark.parametrize(
    "intercept, reason, verdict",
    [
        pytest.param(0.0, "below_minimum", "null", id="redundant"),
        pytest.param(0.01, "above_minimum", "supported", id="additive"),
    ],
)
def test_incremental_value_is_the_exact_intercept_over_the_baseline(intercept, reason, verdict):
    assert sum(BASELINE_PATTERN) == sum(NOISE_PATTERN) == 0
    assert math.sumprod(BASELINE_PATTERN, NOISE_PATTERN) == 0
    baseline = blocks(0.02, 0.03, BASELINE_PATTERN, 48)
    noise = blocks(0.0, 0.01, NOISE_PATTERN, 48)
    returns = [intercept + b + e for b, e in zip(baseline, noise, strict=True)]
    result = evaluate(returns, history(1), baseline, minimum_effect=0.005, subperiods=4)
    assert isinstance(result, Evaluation)
    assert (result.reason, result.verdict) == (reason, verdict)
    assert_interval(
        result.mean,
        intercept + 0.02,
        orthogonal_mean_error(48, 0.03, 0.01),
        EXACT_T_QUANTILE[0.975, 47],
    )
    assert result.mean.lower > 0.005
    assert_interval(
        result.incremental,
        intercept,
        intercept_error(48, 0.02, 0.03, 0.01),
        EXACT_T_QUANTILE[0.975, 46],
    )
    assert result.subperiods == pytest.approx((intercept,) * 4, rel=1e-9)


def test_scaled_copy_of_the_baseline_has_no_incremental_value():
    rng = random.Random(1)
    baseline = [rng.gauss(0.0005, 0.01) for _ in range(250)]
    returns = [1.1161 * value for value in baseline]
    result = evaluate(returns, history(1), baseline, minimum_effect=NEGLIGIBLE_EFFECT)
    assert result.incremental.estimate == pytest.approx(0.0, abs=1e-15)
    assert (result.reason, result.verdict) == ("below_minimum", "null")


def test_opposite_sign_halves_make_a_significant_mean_inconclusive():
    returns = alternating(0.03, 0.01, 20) + alternating(-0.01, 0.01, 20)
    result = evaluate(returns, history(1), minimum_effect=NEGLIGIBLE_EFFECT)
    assert (result.reason, result.verdict) == ("unstable_subperiods", "inconclusive")
    assert result.subperiods == pytest.approx((0.03, -0.01), rel=1e-9)
    assert_interval(
        result.mean, 0.01, orthogonal_mean_error(40, 0.02, 0.01), EXACT_T_QUANTILE[0.975, 39]
    )
    assert result.mean.lower > 0


def test_subperiods_need_a_positive_sign_not_the_minimum_effect():
    returns = alternating(0.004, 0.001, 20) + alternating(0.02, 0.001, 20)
    result = evaluate(returns, history(1), minimum_effect=0.005)
    assert result.subperiods == pytest.approx((0.004, 0.02), rel=1e-9)
    assert_interval(
        result.mean, 0.012, orthogonal_mean_error(40, 0.008, 0.001), EXACT_T_QUANTILE[0.975, 39]
    )
    assert (result.reason, result.verdict) == ("above_minimum", "supported")


def test_near_equal_subperiods_use_floor_bounds():
    result = evaluate(
        alternating(0.01, 0.01, 42), history(1), minimum_effect=NEGLIGIBLE_EFFECT, subperiods=4
    )
    assert result.subperiods == pytest.approx(
        (0.01, 0.01 + 0.01 / 11, 0.01, 0.01 - 0.01 / 11), rel=1e-9
    )


def test_thirty_observations_with_two_per_subperiod_are_accepted():
    result = evaluate(
        alternating(0.01, 0.01, 30), history(1), minimum_effect=NEGLIGIBLE_EFFECT, subperiods=15
    )
    assert isinstance(result, Evaluation)
    assert result.subperiods == pytest.approx((0.01,) * 15, rel=1e-9)


@pytest.mark.parametrize("length, kind", [(40, Evaluation), (29, Invalid)])
def test_results_record_the_procedure_version_and_profile_through_the_artifact_path(length, kind):
    result = evaluate(
        alternating(0.01, 0.01, length), history(1), minimum_effect=0.005, subperiods=4, alpha=0.01
    )
    assert isinstance(result, kind)
    assert result.profile == Profile(0.005, 4, 0.01)
    saved = canonical_bytes(asdict(result))
    assert canonical_bytes(json.loads(saved)) == saved
    assert json.loads(saved)["profile"] == {
        "minimum_effect": 0.005,
        "subperiods": 4,
        "alpha": 0.01,
        "version": "pilot-evaluation/v1",
    }


def test_equal_numbers_record_identical_results():
    returns, record = alternating(0.01, 0.01, 40), history(1)
    integral = evaluate(returns, record, minimum_effect=1, subperiods=2, alpha=0.05)
    real = evaluate(returns, record, minimum_effect=1.0, subperiods=2, alpha=0.05)
    assert canonical_bytes(asdict(integral)) == canonical_bytes(asdict(real))


@pytest.mark.parametrize(
    "returns, baseline, subperiods, reason",
    [
        pytest.param(
            alternating(0.01, 0.01, 40),
            alternating(0.0, 0.01, 39),
            2,
            "length_mismatch",
            id="shorter baseline",
        ),
        pytest.param(
            [*alternating(0.01, 0.01, 29), math.nan],
            alternating(0.0, 0.01, 28),
            2,
            "length_mismatch",
            id="length before nonfinite",
        ),
        pytest.param(
            [math.nan, *alternating(0.01, 0.01, 39)], None, 2, "nonfinite_value", id="nan return"
        ),
        pytest.param(
            [math.inf, *alternating(0.01, 0.01, 39)],
            None,
            2,
            "nonfinite_value",
            id="infinite return",
        ),
        pytest.param(
            alternating(0.01, 0.01, 40),
            [math.nan, *alternating(0.0, 0.01, 39)],
            2,
            "nonfinite_value",
            id="nan baseline",
        ),
        pytest.param(
            alternating(0.01, 0.01, 40),
            [-math.inf, *alternating(0.0, 0.01, 39)],
            2,
            "nonfinite_value",
            id="negative infinite baseline",
        ),
        pytest.param(
            [math.nan, *alternating(0.01, 0.01, 28)],
            None,
            2,
            "nonfinite_value",
            id="nonfinite before too few",
        ),
        pytest.param(
            alternating(0.01, 0.01, 29), None, 2, "too_few_observations", id="29 observations"
        ),
        pytest.param(
            alternating(0.01, 0.01, 30),
            None,
            16,
            "too_few_observations",
            id="16 subperiods of 30",
        ),
        pytest.param([0.01] * 40, None, 2, "zero_variance", id="constant returns"),
        pytest.param(
            alternating(0.01, 0.01, 40), [0.01] * 40, 2, "zero_variance", id="constant baseline"
        ),
        pytest.param(
            alternating(0.01, 0.01, 40),
            [0.01] * 20 + alternating(0.01, 0.01, 20),
            2,
            "zero_variance",
            id="baseline constant in one subperiod",
        ),
    ],
)
def test_data_problems_return_invalid_by_reason(returns, baseline, subperiods, reason):
    result = evaluate(
        returns, history(1), baseline, minimum_effect=NEGLIGIBLE_EFFECT, subperiods=subperiods
    )
    assert isinstance(result, Invalid)
    assert result.reason == reason


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param({"alpha": 0.0}, id="alpha zero"),
        pytest.param({"alpha": 1.0}, id="alpha one"),
        pytest.param({"subperiods": 1}, id="one subperiod"),
        pytest.param({"minimum_effect": 0.0}, id="zero minimum"),
        pytest.param({"minimum_effect": -0.001}, id="negative minimum"),
        pytest.param({"minimum_effect": math.nan}, id="nan minimum"),
        pytest.param({"minimum_effect": math.inf}, id="infinite minimum"),
        pytest.param({"minimum_effect": True}, id="boolean minimum"),
        pytest.param({"alpha": True}, id="boolean alpha"),
        pytest.param({"subperiods": True}, id="boolean subperiods"),
        pytest.param({"subperiods": 2.0}, id="float subperiods"),
        pytest.param({"subperiods": 2.5}, id="fractional subperiods"),
    ],
)
def test_invalid_profile_parameters_raise_value_error(profile):
    declared = {"minimum_effect": NEGLIGIBLE_EFFECT, "subperiods": 2, "alpha": 0.05, **profile}
    with pytest.raises(ValueError):
        evaluate(alternating(0.01, 0.01, 40), history(1), **declared)
    with pytest.raises(ValueError):
        Profile(**declared)
