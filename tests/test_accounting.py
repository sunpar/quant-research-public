from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from decimal import Context, Decimal, localcontext
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from quant_research.accounting import (
    Budget,
    CostLine,
    Decision,
    InexactUsage,
    Ledger,
    LedgerError,
    Refusal,
    Stall,
    Usage,
    Work,
    dump_ledger,
    load_ledger,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_PROBE = """
import sys
sys.path.append(sys.argv[1])
from decimal import Decimal
from uuid import uuid4
from quant_research.accounting import Budget, Ledger, LedgerError, Usage, Work
campaign = uuid4()
ledger = Ledger({campaign: Budget()})
work = Work(campaign, uuid4(), uuid4(), uuid4())
print(ledger.reserve(work, Usage(cost_usd=Decimal("0.10"))).admitted)
ledger.charge(uuid4(), work.attempt_id, "researcher", Usage(cost_usd=Decimal("0.20")))
print(ledger.report()[0].usage.cost_usd)
huge = Usage(cost_usd=Decimal("9E+999999"))
print(ledger.reserve(Work(campaign, uuid4(), uuid4(), uuid4()), huge).admitted)
try:
    ledger.reserve(Work(campaign, uuid4(), uuid4(), uuid4()), huge)
except LedgerError:
    print(len(ledger.events))
"""


@pytest.mark.smoke
def test_repeated_cent_charges_report_exact_dollars():
    campaign = uuid4()
    ledger = Ledger({campaign: Budget()})
    work = Work(campaign, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(work, Usage()).admitted
    for _ in range(3):
        ledger.charge(uuid4(), work.attempt_id, "researcher", Usage(cost_usd=Decimal("0.10")))
    assert ledger.report() == (
        CostLine(campaign, "researcher", "open", Usage(cost_usd=Decimal("0.30"))),
    )


def test_spend_stays_exact_under_a_coarse_caller_context():
    campaign, trial = uuid4(), uuid4()
    ledger = Ledger({campaign: Budget(cost_usd=Decimal("1000.35"))})
    first = Work(campaign, trial, uuid4(), uuid4())
    second = Work(campaign, trial, uuid4(), uuid4())
    with localcontext(Context(prec=3)):
        assert ledger.reserve(first, Usage(cost_usd=Decimal("1000.00"))).admitted
        assert ledger.reserve(second, Usage(cost_usd=Decimal("0.36"))).refusals == (
            Refusal("cost_usd", Decimal("1000.00"), Decimal("0.36"), Decimal("1000.35")),
        )
        assert ledger.reserve(second, Usage(cost_usd=Decimal("0.35"))).admitted
        for amount in ("1000.00", "0.10", "0.25"):
            ledger.charge(uuid4(), first.attempt_id, "researcher", Usage(cost_usd=Decimal(amount)))
        assert ledger.report() == (
            CostLine(campaign, "researcher", "open", Usage(cost_usd=Decimal("1000.35"))),
        )
    with pytest.raises(LedgerError):
        Usage(cost_usd=Decimal("9E+999999")) + Usage(cost_usd=Decimal("9E+999999"))


@pytest.mark.parametrize("setting", ["clamp = 1", "rounding = decimal.ROUND_DOWN"])
def test_ledger_ignores_the_default_decimal_context_at_import(setting):
    script = f"import decimal\ndecimal.DefaultContext.{setting}\n{DEFAULT_CONTEXT_PROBE}"
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", script, str(ROOT / "src")],
        capture_output=True,
        text=True,
    )
    assert (result.returncode, result.stdout) == (0, "True\n0.20\nTrue\n3\n"), result.stderr


def test_charge_that_cannot_be_summed_records_nothing():
    campaign = uuid4()
    ledger = Ledger({campaign: Budget(cost_usd=Decimal("1.00"))})
    work = Work(campaign, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(work, Usage(cost_usd=Decimal("0.50"))).admitted
    huge = Usage(cost_usd=Decimal("9E+999999"))
    ledger.charge(uuid4(), work.attempt_id, "researcher", huge)
    with pytest.raises(LedgerError):
        ledger.charge(uuid4(), work.attempt_id, "researcher", huge)
    assert len(ledger.events) == 2
    assert ledger.report() == (CostLine(campaign, "researcher", "open", huge),)


def test_released_estimates_do_not_block_a_later_charge():
    campaign = uuid4()
    ledger = Ledger({campaign: Budget()})
    huge = Usage(cost_usd=Decimal("9E+999999"))
    released, open_ = (Work(campaign, uuid4(), uuid4(), uuid4()) for _ in range(2))
    assert ledger.reserve(released, huge).admitted
    ledger.release(released.attempt_id, "null")
    assert ledger.reserve(open_, huge).admitted
    ledger.charge(uuid4(), open_.attempt_id, "researcher", Usage())
    assert ledger.report() == (CostLine(campaign, "researcher", "open", Usage()),)


def test_charge_within_an_open_estimate_counts_once_toward_later_sums():
    campaign = uuid4()
    ledger = Ledger({campaign: Budget()})
    huge = Usage(cost_usd=Decimal("9E+999999"))
    work = Work(campaign, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(work, huge).admitted
    ledger.charge(uuid4(), work.attempt_id, "researcher", huge)
    assert ledger.reserve(Work(campaign, uuid4(), uuid4(), uuid4()), Usage()).admitted
    with pytest.raises(LedgerError):
        ledger.charge(uuid4(), work.attempt_id, "researcher", huge)
    assert len(ledger.events) == 3
    assert ledger.report() == (CostLine(campaign, "researcher", "open", huge),)


def test_reservation_that_cannot_be_summed_records_nothing():
    capped, unlimited = uuid4(), uuid4()
    huge = Decimal("9E+999999")
    ledger = Ledger({capped: Budget(cost_usd=huge), unlimited: Budget()})
    for campaign in (capped, unlimited):
        work = Work(campaign, uuid4(), uuid4(), uuid4())
        assert ledger.reserve(work, Usage(cost_usd=huge)).admitted
    work = Work(capped, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(work, Usage(cost_usd=huge)).refusals == (
        Refusal("cost_usd", huge, huge, huge),
    )
    with pytest.raises(LedgerError):
        ledger.reserve(Work(unlimited, uuid4(), uuid4(), uuid4()), Usage(cost_usd=huge))
    assert len(ledger.events) == 2


def test_a_cap_refusal_comes_before_an_inexact_sum_of_an_undeclared_resource():
    campaign = uuid4()
    huge = Usage(model_calls=1, cost_usd=Decimal("9E+999999"))
    ledger = Ledger({campaign: Budget(model_calls=1)})
    assert ledger.reserve(Work(campaign, uuid4(), uuid4(), uuid4()), huge).admitted
    work = Work(campaign, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(work, huge).refusals == (Refusal("model_calls", 1, 1, 1),)
    with pytest.raises(LedgerError, match="exact"):
        ledger.reserve(work, replace(huge, model_calls=0))
    assert len(ledger.events) == 1


@pytest.mark.parametrize(
    "resource, cap, held, exact, over",
    [
        ("compute_seconds", Decimal("10.5"), Decimal("4.25"), Decimal("6.25"), Decimal("6.26")),
        ("wall_seconds", Decimal("1200"), Decimal("600.5"), Decimal("599.5"), Decimal("600")),
        ("model_calls", 5, 3, 2, 3),
        ("input_tokens", 1000, 600, 400, 401),
        ("output_tokens", 200, 150, 50, 51),
        ("cost_usd", Decimal("2.00"), Decimal("1.25"), Decimal("0.75"), Decimal("0.76")),
    ],
)
def test_each_declared_resource_admits_work_up_to_its_cap(resource, cap, held, exact, over):
    campaign, trial = uuid4(), uuid4()
    ledger = Ledger({campaign: Budget(**{resource: cap})})
    large = Usage(
        compute_seconds=Decimal("1e9"),
        wall_seconds=Decimal("1e9"),
        model_calls=10**9,
        input_tokens=10**9,
        output_tokens=10**9,
        cost_usd=Decimal("1e9"),
    )
    first = Work(campaign, trial, uuid4(), uuid4())
    second = Work(campaign, trial, uuid4(), uuid4())
    assert ledger.reserve(first, replace(large, **{resource: held})).admitted
    assert ledger.reserve(second, replace(large, **{resource: over})).refusals == (
        Refusal(resource, held, over, cap),
    )
    assert ledger.reserve(second, replace(large, **{resource: exact})).admitted


def test_refusal_lists_every_exceeded_limit_in_budget_field_order():
    campaign, trial, run = uuid4(), uuid4(), uuid4()
    budget = Budget(model_calls=1, output_tokens=10, cost_usd=Decimal("1.00"), failures_per_run=1)
    ledger = Ledger({campaign: budget})
    failed = Work(campaign, trial, run, uuid4())
    assert ledger.reserve(failed, Usage()).admitted
    ledger.release(failed.attempt_id, "failed")
    estimate = Usage(model_calls=2, output_tokens=10, cost_usd=Decimal("1.01"))
    assert ledger.reserve(Work(campaign, trial, run, uuid4()), estimate).refusals == (
        Refusal("model_calls", 0, 2, 1),
        Refusal("cost_usd", Decimal(0), Decimal("1.01"), Decimal("1.00")),
        Refusal("failures_per_run", 1, 1, 1),
    )


def test_repeated_requests_record_once_and_conflicting_repeats_record_nothing():
    campaign = uuid4()
    ledger = Ledger({campaign: Budget(cost_usd=Decimal("1.00"))})
    work = Work(campaign, uuid4(), uuid4(), uuid4())
    estimate = Usage(cost_usd=Decimal("0.60"))
    charge = uuid4()
    assert ledger.reserve(work, estimate).admitted
    assert ledger.reserve(work, estimate) == Decision(())
    assert len(ledger.events) == 1
    for _ in range(2):
        ledger.charge(charge, work.attempt_id, "researcher", Usage(cost_usd=Decimal("0.20")))
    assert len(ledger.events) == 2
    for _ in range(2):
        ledger.release(work.attempt_id, "successful")
    assert len(ledger.events) == 3
    with pytest.raises(LedgerError):
        ledger.reserve(work, Usage(cost_usd=Decimal("0.50")))
    with pytest.raises(LedgerError):
        ledger.charge(charge, work.attempt_id, "researcher", Usage(cost_usd=Decimal("0.30")))
    with pytest.raises(LedgerError):
        ledger.release(work.attempt_id, "failed")
    assert len(ledger.events) == 3


def test_release_returns_unused_reservation_to_the_campaign():
    campaign, trial = uuid4(), uuid4()
    ledger = Ledger({campaign: Budget(cost_usd=Decimal("1.00"))})
    first = Work(campaign, trial, uuid4(), uuid4())
    second = Work(campaign, trial, uuid4(), uuid4())
    estimate = Usage(cost_usd=Decimal("0.70"))
    held = (Refusal("cost_usd", Decimal("1.00"), Decimal("0.70"), Decimal("1.00")),)
    assert ledger.reserve(first, Usage(cost_usd=Decimal("1.00"))).admitted
    assert ledger.reserve(second, estimate).refusals == held
    ledger.charge(uuid4(), first.attempt_id, "researcher", Usage(cost_usd=Decimal("0.30")))
    assert ledger.reserve(second, estimate).refusals == held
    ledger.release(first.attempt_id, "successful")
    assert ledger.reserve(second, estimate).admitted


def test_an_observed_token_limit_refuses_only_once_the_known_sum_passes_it():
    campaign, trial = uuid4(), uuid4()
    ledger = Ledger({campaign: Budget(output_tokens=100)})
    first, second, third = (Work(campaign, trial, uuid4(), uuid4()) for _ in range(3))
    assert ledger.reserve(first, Usage()).admitted
    ledger.charge(uuid4(), first.attempt_id, "critic", Usage(output_tokens=100))
    unreported = Usage(output_tokens=30, unreported=frozenset({"output_tokens"}))
    ledger.charge(uuid4(), first.attempt_id, "critic", unreported)
    ledger.release(first.attempt_id, "null")
    assert ledger.reserve(second, Usage()).refusals == (Refusal("output_tokens", 130, 0, 100),)
    capped = Ledger({campaign: Budget(output_tokens=100)})
    assert capped.reserve(first, Usage()).admitted
    capped.charge(uuid4(), first.attempt_id, "critic", Usage(output_tokens=100))
    capped.release(first.attempt_id, "null")
    assert capped.reserve(third, Usage()).admitted


def test_unreported_usage_sums_its_known_part_and_counts_the_charges_that_left_it_out():
    campaign = uuid4()
    ledger = Ledger({campaign: Budget(cost_usd=Decimal("1.00"))})
    work = Work(campaign, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(work, Usage(model_calls=3)).admitted
    sealed = Usage(
        model_calls=1, unreported=frozenset({"input_tokens", "output_tokens", "cost_usd"})
    )
    subscription = Usage(
        model_calls=1, input_tokens=40, unreported=frozenset({"input_tokens", "output_tokens"})
    )
    metered = Usage(model_calls=1, input_tokens=10, output_tokens=5, cost_usd=Decimal("0.30"))
    for usage in (sealed, subscription, metered):
        ledger.charge(uuid4(), work.attempt_id, "proposer", usage)
    ledger.release(work.attempt_id, "null")
    known = Usage(
        model_calls=3,
        input_tokens=50,
        output_tokens=5,
        cost_usd=Decimal("0.30"),
        unreported=frozenset({"input_tokens", "output_tokens", "cost_usd"}),
    )
    assert ledger.report() == (
        CostLine(
            campaign,
            "proposer",
            "null",
            known,
            {"input_tokens": 2, "output_tokens": 2, "cost_usd": 1},
        ),
    )
    later = Work(campaign, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(later, Usage(cost_usd=Decimal("0.71"))).refusals == (
        Refusal("cost_usd", Decimal("0.30"), Decimal("0.71"), Decimal("1.00")),
    )
    assert ledger.reserve(later, Usage(cost_usd=Decimal("0.70"))).admitted


def test_open_attempt_charged_beyond_its_estimate_holds_its_spend_within_its_campaign():
    campaign, other, trial = uuid4(), uuid4(), uuid4()
    budget = Budget(cost_usd=Decimal("1.00"))
    ledger = Ledger({campaign: budget, other: budget})
    first = Work(campaign, trial, uuid4(), uuid4())
    assert ledger.reserve(first, Usage(cost_usd=Decimal("0.40"))).admitted
    ledger.charge(uuid4(), first.attempt_id, "researcher", Usage(cost_usd=Decimal("0.90")))
    second = Work(campaign, trial, uuid4(), uuid4())
    assert ledger.reserve(second, Usage(cost_usd=Decimal("0.20"))).refusals == (
        Refusal("cost_usd", Decimal("0.90"), Decimal("0.20"), Decimal("1.00")),
    )
    elsewhere = Work(other, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(elsewhere, Usage(cost_usd=Decimal("1.00"))).admitted


@pytest.mark.smoke
def test_failed_attempts_stall_their_run_but_not_a_repeat_of_the_trial():
    campaign, unlimited, trial, run = uuid4(), uuid4(), uuid4(), uuid4()
    ledger = Ledger({campaign: Budget(failures_per_run=2), unlimited: Budget()})
    for outcome in ("null", "failed", "failed"):
        work = Work(campaign, trial, run, uuid4())
        assert ledger.reserve(work, Usage()).admitted
        ledger.release(work.attempt_id, outcome)
    assert ledger.reserve(Work(campaign, trial, run, uuid4()), Usage()).refusals == (
        Refusal("failures_per_run", 2, 1, 2),
    )
    assert ledger.stalls() == (Stall(campaign, trial, run, 2),)
    repeat = Work(campaign, trial, uuid4(), uuid4())
    elsewhere = Work(unlimited, uuid4(), uuid4(), uuid4())
    for work in (repeat, elsewhere):
        assert ledger.reserve(work, Usage()).admitted
        ledger.release(work.attempt_id, "failed")
    assert ledger.stalls() == (Stall(campaign, trial, run, 2),)


@pytest.mark.parametrize(
    "outcome, refusals, stalls",
    [
        ("successful", 0, 0),
        ("null", 0, 0),
        ("rejected", 0, 0),
        ("inconclusive", 0, 0),
        ("failed", 1, 1),
    ],
)
def test_open_retries_hold_failure_slots_until_released_without_failing(outcome, refusals, stalls):
    campaign, trial, run = uuid4(), uuid4(), uuid4()
    ledger = Ledger({campaign: Budget(failures_per_run=2)})
    failed, retry, overlapping = (Work(campaign, trial, run, uuid4()) for _ in range(3))
    full = Refusal("failures_per_run", 2, 1, 2)
    assert ledger.reserve(failed, Usage()).admitted
    ledger.release(failed.attempt_id, "failed")
    assert ledger.reserve(retry, Usage()).admitted
    assert ledger.reserve(overlapping, Usage()).refusals == (full,)
    ledger.release(retry.attempt_id, outcome)
    assert ledger.reserve(overlapping, Usage()).refusals == (full,) * refusals
    assert ledger.stalls() == (Stall(campaign, trial, run, 2),) * stalls


def test_report_attributes_spend_to_failed_null_successful_and_open_work():
    first, second = uuid4(), uuid4()
    ledger = Ledger({first: Budget(), second: Budget()})
    trial, run = uuid4(), uuid4()
    attempts = [
        (
            Work(first, trial, run, uuid4()),
            "failed",
            [("researcher", Usage(model_calls=2, cost_usd=Decimal("0.25")))],
        ),
        (
            Work(first, trial, run, uuid4()),
            "failed",
            [("researcher", Usage(cost_usd=Decimal("0.05")))],
        ),
        (
            Work(first, trial, run, uuid4()),
            "null",
            [("critic", Usage(input_tokens=500, cost_usd=Decimal("0.10")))],
        ),
        (
            Work(first, trial, uuid4(), uuid4()),
            "successful",
            [
                ("researcher", Usage(cost_usd=Decimal("0.40"))),
                ("critic", Usage(output_tokens=80, cost_usd=Decimal("0.05"))),
            ],
        ),
        (Work(first, trial, uuid4(), uuid4()), "rejected", []),
        (
            Work(second, uuid4(), uuid4(), uuid4()),
            None,
            [("researcher", Usage(compute_seconds=Decimal("12.5"), cost_usd=Decimal("1.00")))],
        ),
        (Work(second, uuid4(), uuid4(), uuid4()), None, []),
    ]
    for work, outcome, charges in attempts:
        assert ledger.reserve(work, Usage(cost_usd=Decimal("5.00"))).admitted
        for role, usage in charges:
            ledger.charge(uuid4(), work.attempt_id, role, usage)
        if outcome is not None:
            ledger.release(work.attempt_id, outcome)
    expected = [
        CostLine(first, "critic", "null", Usage(input_tokens=500, cost_usd=Decimal("0.10"))),
        CostLine(first, "critic", "successful", Usage(output_tokens=80, cost_usd=Decimal("0.05"))),
        CostLine(first, "researcher", "failed", Usage(model_calls=2, cost_usd=Decimal("0.30"))),
        CostLine(first, "researcher", "successful", Usage(cost_usd=Decimal("0.40"))),
        CostLine(
            second,
            "researcher",
            "open",
            Usage(compute_seconds=Decimal("12.5"), cost_usd=Decimal("1.00")),
        ),
    ]
    report = ledger.report()
    assert report == tuple(
        sorted(expected, key=lambda line: (str(line.campaign_id), line.role, line.outcome))
    )
    assert sum((line.usage for line in report), Usage()) == Usage(
        compute_seconds=Decimal("12.5"),
        model_calls=2,
        input_tokens=500,
        output_tokens=80,
        cost_usd=Decimal("1.85"),
    )


def test_late_charge_after_release_counts_under_its_outcome_and_against_later_work():
    campaign, trial = uuid4(), uuid4()
    ledger = Ledger({campaign: Budget(cost_usd=Decimal("1.00"))})
    first = Work(campaign, trial, uuid4(), uuid4())
    assert ledger.reserve(first, Usage(cost_usd=Decimal("0.50"))).admitted
    ledger.charge(uuid4(), first.attempt_id, "researcher", Usage(cost_usd=Decimal("0.20")))
    ledger.release(first.attempt_id, "null")
    ledger.charge(uuid4(), first.attempt_id, "researcher", Usage(cost_usd=Decimal("0.95")))
    assert ledger.report() == (
        CostLine(campaign, "researcher", "null", Usage(cost_usd=Decimal("1.15"))),
    )
    second = Work(campaign, trial, uuid4(), uuid4())
    assert ledger.reserve(second, Usage(cost_usd=Decimal("0.10"))).refusals == (
        Refusal("cost_usd", Decimal("1.15"), Decimal("0.10"), Decimal("1.00")),
    )


@pytest.mark.parametrize(
    "record, values",
    [
        (Usage, {"cost_usd": 0.1}),
        (Usage, {"cost_usd": 1}),
        (Usage, {"model_calls": True}),
        (Usage, {"output_tokens": 2.0}),
        (Usage, {"input_tokens": -1}),
        (Usage, {"compute_seconds": Decimal("-0.5")}),
        (Usage, {"compute_seconds": Decimal("NaN")}),
        (Usage, {"cost_usd": Decimal("-Infinity")}),
        (Usage, {"cost_usd": Decimal("1E-1000000")}),
        (Usage, {"compute_seconds": Decimal("10E+999999")}),
        (Usage, {"wall_seconds": 1.5}),
        (Usage, {"unreported": frozenset({"model_calls"})}),
        (Usage, {"unreported": {"cost_usd"}}),
        (Usage, {"unreported": ("cost_usd",)}),
        (Budget, {"compute_seconds": 1.5}),
        (Budget, {"model_calls": False}),
        (Budget, {"failures_per_run": 2.0}),
        (Budget, {"failures_per_run": -1}),
        (Budget, {"cost_usd": Decimal("sNaN")}),
        (Budget, {"compute_seconds": Decimal("1E-999999999999999999")}),
        (
            Work,
            {
                "campaign_id": uuid4(),
                "trial_id": uuid4(),
                "run_id": uuid4(),
                "attempt_id": UUID(int=0),
            },
        ),
    ],
)
def test_records_reject_floats_bools_negatives_unbounded_values_and_nil_ids(record, values):
    with pytest.raises(LedgerError):
        record(**values)


@pytest.mark.parametrize(
    "case",
    [
        "unknown-campaign",
        "run-under-another-trial",
        "trial-under-another-campaign",
        "trial-reuses-run-id",
        "attempt-reuses-campaign-id",
        "attempt-reuses-charge-id",
        "charge-reuses-run-id",
        "nil-charge-id",
        "unreserved-charge",
        "unreserved-release",
        "blank-role",
        "nul-role",
        "unknown-outcome",
    ],
)
def test_invalid_ledger_requests_raise_and_record_nothing(case):
    campaign, other, charge = uuid4(), uuid4(), uuid4()
    ledger = Ledger({campaign: Budget(), other: Budget()})
    work = Work(campaign, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(work, Usage()).admitted
    ledger.charge(charge, work.attempt_id, "researcher", Usage())
    with pytest.raises(LedgerError):
        match case:
            case "unknown-campaign":
                ledger.reserve(Work(uuid4(), uuid4(), uuid4(), uuid4()), Usage())
            case "run-under-another-trial":
                ledger.reserve(Work(campaign, uuid4(), work.run_id, uuid4()), Usage())
            case "trial-under-another-campaign":
                ledger.reserve(Work(other, work.trial_id, uuid4(), uuid4()), Usage())
            case "trial-reuses-run-id":
                ledger.reserve(Work(campaign, work.run_id, uuid4(), uuid4()), Usage())
            case "attempt-reuses-campaign-id":
                ledger.reserve(Work(campaign, work.trial_id, work.run_id, other), Usage())
            case "attempt-reuses-charge-id":
                ledger.reserve(Work(campaign, work.trial_id, work.run_id, charge), Usage())
            case "charge-reuses-run-id":
                ledger.charge(work.run_id, work.attempt_id, "researcher", Usage())
            case "nil-charge-id":
                ledger.charge(UUID(int=0), work.attempt_id, "researcher", Usage())
            case "unreserved-charge":
                ledger.charge(uuid4(), uuid4(), "researcher", Usage())
            case "unreserved-release":
                ledger.release(uuid4(), "failed")
            case "blank-role":
                ledger.charge(uuid4(), work.attempt_id, " ", Usage())
            case "nul-role":
                ledger.charge(uuid4(), work.attempt_id, "critic\x00", Usage())
            case "unknown-outcome":
                ledger.release(work.attempt_id, "confirmed")
    assert len(ledger.events) == 2


def test_refused_or_failed_requests_leave_later_requests_unaffected():
    capped, unlimited = uuid4(), uuid4()
    huge = Decimal("9E+999999")
    ledger = Ledger({capped: Budget(model_calls=1), unlimited: Budget()})
    held = Work(capped, uuid4(), uuid4(), uuid4())
    assert ledger.reserve(held, Usage(model_calls=1, cost_usd=huge)).admitted
    ledger.charge(uuid4(), held.attempt_id, "researcher", Usage(cost_usd=huge))
    refused, clash, overflow = (Work(capped, uuid4(), uuid4(), uuid4()) for _ in range(3))
    assert not ledger.reserve(refused, Usage(model_calls=1)).admitted
    with pytest.raises(LedgerError, match="another kind or parent") as clashed:
        ledger.reserve(replace(clash, run_id=held.trial_id), Usage())
    assert not isinstance(clashed.value, InexactUsage)
    with pytest.raises(InexactUsage, match="exact"):
        ledger.reserve(overflow, Usage(cost_usd=huge))
    with pytest.raises(InexactUsage, match="exact"):
        ledger.charge(uuid4(), held.attempt_id, "researcher", Usage(cost_usd=huge))
    assert ledger.report() == (CostLine(capped, "researcher", "open", Usage(cost_usd=huge)),)
    for work in (refused, clash, overflow):
        assert ledger.reserve(replace(work, campaign_id=unlimited), Usage()).admitted
    assert len(ledger.events) == 5


def test_a_thousand_retries_of_one_run_record_within_seconds():
    attempts = 1000
    campaign, trial, run = uuid4(), uuid4(), uuid4()
    budget = Budget(compute_seconds=Decimal(attempts), failures_per_run=attempts)
    ledger = Ledger({campaign: budget})
    cost = Usage(compute_seconds=Decimal(1))
    start = time.process_time()
    for _ in range(attempts):
        work = Work(campaign, trial, run, uuid4())
        assert ledger.reserve(work, cost).admitted
        ledger.charge(uuid4(), work.attempt_id, "researcher", cost)
        ledger.release(work.attempt_id, "failed")
    report = ledger.report()
    # A worker restart replays every attempt inside one 10-second Temporal workflow task.
    # CPU time ignores how busy the shared test machine is.
    assert time.process_time() - start < 3
    assert report == (
        CostLine(campaign, "researcher", "failed", Usage(compute_seconds=Decimal(attempts))),
    )


def filled_ledger(budgets):
    ledger = Ledger(budgets)
    campaign = next(iter(budgets))
    first, second = (Work(campaign, uuid4(), uuid4(), uuid4()) for _ in range(2))
    assert ledger.reserve(first, Usage(cost_usd=Decimal("0.70"))).admitted
    ledger.charge(uuid4(), first.attempt_id, "researcher", Usage(cost_usd=Decimal("0.20")))
    ledger.release(first.attempt_id, "failed")
    assert ledger.reserve(second, Usage(wall_seconds=Decimal("600"), model_calls=1)).admitted
    sealed = Usage(model_calls=1, unreported=frozenset({"input_tokens", "output_tokens"}))
    ledger.charge(uuid4(), second.attempt_id, "proposer", sealed)
    return ledger


def test_a_dumped_ledger_reloads_with_the_same_events_report_and_bytes():
    campaign, idle = uuid4(), uuid4()
    budgets = {campaign: Budget(cost_usd=Decimal("1.00"), failures_per_run=2), idle: Budget()}
    ledger = filled_ledger(budgets)
    content = dump_ledger(ledger)
    with localcontext(Context(capitals=0, prec=3)):
        reloaded = load_ledger(content)
        assert dump_ledger(reloaded) == content
    assert reloaded.events == ledger.events
    assert reloaded.report() == ledger.report()
    stored = json.loads(content)
    assert stored["budgets"][str(campaign)] == {
        "compute_seconds": None,
        "wall_seconds": None,
        "model_calls": None,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": "1.00",
        "failures_per_run": 2,
    }
    assert stored["events"][-1]["charge"]["usage"]["unreported"] == [
        "input_tokens",
        "output_tokens",
    ]
    later = Work(campaign, uuid4(), uuid4(), uuid4())
    assert reloaded.reserve(later, Usage(cost_usd=Decimal("0.81"))).refusals == (
        Refusal("cost_usd", Decimal("0.20"), Decimal("0.81"), Decimal("1.00")),
    )


def _damage(content, change):
    value = json.loads(content)
    change(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.mark.parametrize(
    "case, message",
    [
        ("noncanonical-bytes", "not canonical"),
        ("schema-version", "schema version"),
        ("charge-before-reservation", "has no reservation"),
        ("repeated-event", "refused or repeats"),
        ("estimate-over-budget", "refused or repeats"),
        ("conflicting-release", "already released"),
        ("noncanonical-decimal", "Invalid decimal"),
        ("unbounded-exponent", "between 1E-999999 and 1E\\+999999"),
        ("noncanonical-uuid", "Invalid canonical charge id UUID"),
        ("float-count", "finite nonnegative int"),
        ("unknown-campaign", "Unknown campaign"),
        ("lowered-budget", "refused or repeats"),
        ("unsorted-unreported", "sorted order"),
        ("unreportable-field", "unreported must be"),
    ],
)
def test_damaged_ledger_bytes_fail_to_load(case, message):
    campaign = uuid4()
    content = dump_ledger(filled_ledger({campaign: Budget(cost_usd=Decimal("1.00"))}))
    match case:
        case "noncanonical-bytes":
            content = content.replace(b",", b", ", 1)
        case "schema-version":
            content = _damage(content, lambda value: value.update(schema_version=2))
        case "charge-before-reservation":
            content = _damage(
                content, lambda value: value["events"].insert(0, value["events"].pop(1))
            )
        case "repeated-event":
            content = _damage(content, lambda value: value["events"].append(value["events"][1]))
        case "estimate-over-budget":
            content = _damage(
                content,
                lambda value: value["events"][0]["reserve"]["estimate"].update(cost_usd="1.01"),
            )
        case "conflicting-release":
            release = {"release": json.loads(content)["events"][2]["release"] | {"outcome": "null"}}
            content = _damage(content, lambda value: value["events"].append(release))
        case "noncanonical-decimal":
            content = _damage(
                content,
                lambda value: value["events"][0]["reserve"]["estimate"].update(cost_usd="7E-1"),
            )
        case "unbounded-exponent":
            content = _damage(
                content,
                lambda value: value["events"][0]["reserve"]["estimate"].update(
                    cost_usd="1E-999999999999999999"
                ),
            )
        case "noncanonical-uuid":
            content = _damage(
                content,
                lambda value: value["events"][1]["charge"].update(
                    id=value["events"][1]["charge"]["id"].upper()
                ),
            )
        case "float-count":
            content = _damage(
                content, lambda value: value["events"][1]["charge"]["usage"].update(model_calls=1.0)
            )
        case "unknown-campaign":
            content = _damage(
                content,
                lambda value: value.update(budgets={str(uuid4()): value["budgets"].popitem()[1]}),
            )
        case "lowered-budget":
            content = _damage(
                content, lambda value: value["budgets"][str(campaign)].update(cost_usd="0.69")
            )
        case "unsorted-unreported":
            content = _damage(
                content,
                lambda value: value["events"][-1]["charge"]["usage"].update(
                    unreported=["output_tokens", "input_tokens"]
                ),
            )
        case "unreportable-field":
            content = _damage(
                content,
                lambda value: value["events"][-1]["charge"]["usage"].update(
                    unreported=["model_calls"]
                ),
            )
    with pytest.raises(LedgerError, match=message):
        load_ledger(content)


def test_dumped_event_prefix_stays_identical_after_ledger_advancement():
    campaign = uuid4()
    ledger = Ledger({campaign: Budget(model_calls=100)})
    empty = dump_ledger(ledger)
    work = Work(campaign, uuid4(), uuid4(), uuid4())
    ledger.reserve(work, Usage(model_calls=2))
    reserved = dump_ledger(ledger)
    ledger.charge(uuid4(), work.attempt_id, "researcher", Usage(model_calls=1))
    charged = dump_ledger(ledger)
    ledger.release(work.attempt_id, "successful")
    for count, content in enumerate((empty, reserved, charged, dump_ledger(ledger))):
        assert dump_ledger(ledger, event_count=count) == content
        assert dump_ledger(load_ledger(content)) == content
    assert dump_ledger(ledger, event_count=len(ledger.events)) == dump_ledger(ledger)


@pytest.mark.parametrize("count", [-1, 1, True, 0.0, "0"])
def test_dump_ledger_refuses_an_unavailable_or_invalid_event_prefix(count):
    with pytest.raises(LedgerError, match="available prefix"):
        dump_ledger(Ledger({uuid4(): Budget()}), event_count=count)
