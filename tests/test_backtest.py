from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import UTC, date, datetime
from fractions import Fraction
from pathlib import Path

import pytest

from quant_research import backtest, datasets
from quant_research.access import PERIOD_MAPPING, DatasetPeriods, Interval, issue_context
from quant_research.backtest import TOLERANCE, BacktestError, Factors, Holding, Period, Sample
from quant_research.datasets import Refused
from quant_research.public_data import approve, ingest, load_catalog, pull, validate
from quant_research.signals import shift

# numba's default cache writes into site-packages, which the prepared environment fingerprints.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "quant-research-numba"))

# A real 12-month excerpt, July 1989 to June 1990. Percent text per month (Mkt-RF, RF):
# 1989-07 7.18 0.70 | 1989-08 1.43 0.74 | 1989-09 -0.76 0.65 | 1989-10 -3.65 0.68
# 1989-11 1.04 0.69 | 1989-12 1.18 0.61 | 1990-01 -7.80 0.57 | 1990-02 1.12 0.57
# 1990-03 1.83 0.64 | 1990-04 -3.36 0.69 | 1990-05 8.43 0.68 | 1990-06 -1.09 0.63
FRENCH = (Path(__file__).parent / "fixtures" / "public_data" / "french-ff3.zip").read_bytes()
DATASET = "french.ff3_monthly"
NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
CONTEXT = issue_context(clearances=(), cutoff=NOW, generation="backtest/v1", providers=())
ENGINES = [
    pytest.param(backtest.plain, id="plain"),
    pytest.param(backtest.vectorbt, id="vectorbt"),
]
M, B = Holding.MARKET, Holding.BILLS


def ingested(root: Path) -> Path:
    pull(root, DATASET, now=NOW, api_key=None, fetch=lambda url: FRENCH)
    ingest(root, DATASET, now=NOW)
    return root


@pytest.fixture
def root(tmp_path):
    ingested(tmp_path)
    validate(tmp_path, DATASET, actor="researcher", evidence=b"notes", summary="checked", now=NOW)
    approve(tmp_path, DATASET, actor="researcher", now=NOW)
    return tmp_path


@pytest.fixture
def market(root):
    return backtest.read(load_catalog(root), root / "datasets", CONTEXT)


def allocation(start: str, *holdings: Holding) -> dict[date, Holding]:
    first = date.fromisoformat(f"{start}-01")
    return {shift(first, offset): holding for offset, holding in enumerate(holdings)}


def assert_returns(engine, sample: Sample, expected: tuple[Fraction, ...]) -> None:
    returns = engine(sample)
    if engine is backtest.plain:
        assert returns == expected
    else:
        assert (
            max(abs(got - want) for got, want in zip(returns, expected, strict=True)) <= TOLERANCE
        )


def test_read_converts_percent_text_to_exact_decimals(root, market):
    july, september = date(1989, 7, 1), date(1989, 9, 1)
    assert sorted(market.factors) == [shift(july, offset) for offset in range(12)]
    assert market.factors[july] == Factors(Fraction(7, 1000), Fraction(359, 5000))  # 0.70%, 7.18%
    assert market.factors[september].mkt_rf == Fraction(-19, 2500)  # -0.76% = -76/10000
    adapter = datasets.read(load_catalog(root), root / "datasets", DATASET, CONTEXT)
    assert market.record == adapter.record


def test_read_returns_the_adapter_refusal(tmp_path):
    root = ingested(tmp_path)
    assert backtest.read(load_catalog(root), root / "datasets", CONTEXT) == Refused("documented")


@pytest.mark.parametrize("engine", ENGINES)
def test_first_month_carries_no_initial_fee(market, engine):
    # 1989-07 in the market: RF + Mkt-RF = 0.70% + 7.18% = 788/10000 = 197/2500.
    assert_returns(engine, market.sample(allocation("1989-07", M)), (Fraction(197, 2500),))
    # 1989-07 in T-bills: RF = 0.70% = 7/1000.
    assert_returns(engine, market.sample(allocation("1989-07", B)), (Fraction(7, 1000),))


@pytest.mark.parametrize("engine", ENGINES)
def test_switch_in_a_middle_month_costs_ten_basis_points_of_the_starting_value(market, engine):
    expected = (
        # 1989-07 bills, initial: RF = 70/10000 = 7/1000.
        Fraction(7, 1000),
        # 1989-08 into the market: (999/1000)(1 + 74/10000 + 143/10000) - 1
        #   = 999 × 10217 / 10^7 - 1 = 10206783/10^7 - 1 = 206783/10^7.
        # The additive form gives 217/10000 - 1/1000 = 207/10000, 2.17e-5 away.
        Fraction(206783, 10_000_000),
        # 1989-09 stays in the market: 65/10000 - 76/10000 = -11/10000.
        Fraction(-11, 10_000),
    )
    assert_returns(engine, market.sample(allocation("1989-07", B, M, M)), expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_switch_in_the_last_month_is_charged(market, engine):
    into_bills = (
        # 1990-04 market, initial: 69/10000 - 336/10000 = -267/10000.
        Fraction(-267, 10_000),
        # 1990-05 market: 68/10000 + 843/10000 = 911/10000.
        Fraction(911, 10_000),
        # 1990-06 into bills: (999/1000)(1 + 63/10000) - 1
        #   = 999 × 10063 / 10^7 - 1 = 10052937/10^7 - 1 = 52937/10^7.
        Fraction(52937, 10_000_000),
    )
    assert_returns(engine, market.sample(allocation("1990-04", M, M, B)), into_bills)
    into_market = (
        # 1990-04 bills, initial: 69/10000.
        Fraction(69, 10_000),
        # 1990-05 bills: 68/10000 = 17/2500.
        Fraction(17, 2500),
        # 1990-06 into the market: (999/1000)(1 + 63/10000 - 109/10000) - 1
        #   = 999 × 9954 / 10^7 - 1 = 9944046/10^7 - 1 = -55954/10^7 = -27977/5000000.
        Fraction(-27977, 5_000_000),
    )
    assert_returns(engine, market.sample(allocation("1990-04", B, B, M)), into_market)


@pytest.mark.parametrize("engine", ENGINES)
def test_switches_in_consecutive_months_each_cost(market, engine):
    expected = (
        # 1989-07 market, initial: 70/10000 + 718/10000 = 788/10000 = 197/2500.
        Fraction(197, 2500),
        # 1989-08 into bills: (999/1000)(10074/10000) - 1 = 10063926/10^7 - 1 = 31963/5000000.
        Fraction(31963, 5_000_000),
        # 1989-09 into the market: (999/1000)(1 + 65/10000 - 76/10000) - 1
        #   = 999 × 9989 / 10^7 - 1 = 9979011/10^7 - 1 = -20989/10^7.
        Fraction(-20989, 10_000_000),
    )
    assert_returns(engine, market.sample(allocation("1989-07", M, B, M)), expected)


def test_buy_and_hold_is_the_market_total_return(market):
    sample = market.sample(dict.fromkeys(market.factors, M))
    expected = tuple(period.factors.rf + period.factors.mkt_rf for period in sample.periods)
    assert backtest.plain(sample) == expected


@pytest.mark.parametrize(
    "holdings",
    [
        pytest.param({}, id="empty"),
        pytest.param(allocation("1989-07", M) | allocation("1989-09", M), id="skips 1989-08"),
        pytest.param(allocation("1990-06", M, M), id="past the factors"),
        pytest.param(allocation("1989-06", B, M), id="before the factors"),
        pytest.param({date(1989, 7, 1): 1}, id="a weight"),
        pytest.param({date(1989, 7, 1): None}, id="unresolved previous holding"),
        pytest.param({date(1989, 7, 15): M}, id="mid-month"),
    ],
)
def test_sample_refuses_an_allocation_the_factors_cannot_back_test(market, holdings):
    with pytest.raises(BacktestError):
        market.sample(holdings)


@pytest.mark.parametrize(
    "period",
    [
        Period(date(1989, 7, 15), M, Factors(Fraction(7, 1000), Fraction(359, 5000))),
        Period(date(1989, 7, 1), M, Factors(0.007, 0.0718)),
    ],
    ids=["mid-month", "float factors"],
)
def test_sample_refuses_a_period_engines_cannot_trust(period):
    with pytest.raises(BacktestError):
        Sample((period,))


def test_compare_agrees_with_vectorbt_by_default(market):
    sample = market.sample(allocation("1989-07", M, M, B, B, B, M, M, M, M, M, M, B))
    agreement = backtest.compare(sample)
    assert agreement.agreed
    assert agreement.reference == backtest.plain(sample)
    assert agreement.candidate == backtest.vectorbt(sample)


def test_compare_names_the_earliest_worst_month_of_a_disagreeing_engine(market):
    sample = market.sample(allocation("1989-07", M, B, B, M))

    def off(sample):
        first, second, third, fourth = backtest.plain(sample)
        return (
            first,
            second + Fraction(1, 10**12),
            third + Fraction(1, 10**11),
            fourth - Fraction(1, 10**11),
        )

    agreement = backtest.compare(sample, off)
    assert (agreement.month, agreement.difference) == (date(1989, 9, 1), Fraction(1, 10**11))
    assert not agreement.agreed


def test_compare_fails_an_engine_that_charges_the_additive_cost(market):
    sample = market.sample(allocation("1989-07", M, B, B, M))

    def additive(sample):
        returns, previous = [], sample.periods[0].holding
        for period in sample.periods:
            earned = period.factors.rf + period.holding.value * period.factors.mkt_rf
            returns.append(earned - (Fraction(1, 1000) if period.holding is not previous else 0))
            previous = period.holding
        return tuple(returns)

    # (1 - c)(1 + r) - 1 = r - c - c × r, so the additive r - c misses by c × |r| in a switch month.
    # 1989-08 into bills: r = 74/10000, gap 74/10^7.
    # 1989-10 into the market: r = 68/10000 - 365/10000 = -297/10000, gap 297/10^7, the worst.
    agreement = backtest.compare(sample, additive)
    assert (agreement.month, agreement.difference) == (date(1989, 10, 1), Fraction(297, 10**7))
    assert not agreement.agreed


def test_compare_refuses_an_engine_that_drops_a_month(market):
    sample = market.sample(allocation("1989-07", M, B))
    with pytest.raises(BacktestError):
        backtest.compare(sample, lambda sample: backtest.plain(sample)[:-1])


@pytest.mark.smoke
def test_module_and_plain_engine_run_without_site_packages():
    script = (
        f"import sys; sys.path.append({str(Path(__file__).parents[1] / 'src')!r})\n"
        "from datetime import date\n"
        "from fractions import Fraction\n"
        "from quant_research.backtest import Factors, Holding, Period, Sample, plain\n"
        "factors = Factors(Fraction(7, 1000), Fraction(359, 5000))\n"
        "sample = Sample((Period(date(1989, 7, 1), Holding.MARKET, factors),))\n"
        "assert plain(sample) == (Fraction(197, 2500),)\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_period_permission_does_not_widen_to_supply_an_allocated_factor_month(root):
    october = datetime(1989, 10, 1, tzinfo=UTC)
    context = issue_context(
        clearances=(),
        cutoff=NOW,
        generation="backtest/v1",
        providers=(),
        scope=(DATASET,),
        dataset_periods=(
            DatasetPeriods(
                DATASET, "a" * 64, "b" * 64, PERIOD_MAPPING, (Interval(october, october),)
            ),
        ),
    )
    market = backtest.read(load_catalog(root), root / "datasets", context)
    assert market.record.supplied == (("1989-10-01", 1),)
    assert tuple(market.factors) == (date(1989, 10, 1),)
    with pytest.raises(BacktestError, match="No factors for allocated month 1989-09-01"):
        market.sample({date(1989, 9, 1): Holding.MARKET})
