from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from enum import Enum
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from quant_research import datasets
from quant_research.access import RunContext
from quant_research.catalog import Catalog
from quant_research.datasets import ReadRecord, Refused
from quant_research.signals import shift

FACTORS = "french.ff3_monthly"
COST = Fraction(1, 1000)
TOLERANCE = Fraction(1, 10**12)


class BacktestError(ValueError):
    """Report a sample the engines cannot back test, or an engine that broke its contract."""


class Holding(Enum):
    """What the portfolio holds through one month. The value is w in RF + w × (Mkt-RF)."""

    BILLS = 0
    MARKET = 1


@dataclass(frozen=True)
class Factors:
    """One month's T-bill return and market excess return as exact decimals, not percent."""

    rf: Fraction
    mkt_rf: Fraction


@dataclass(frozen=True)
class Period:
    """One month of a sample: what the portfolio held and what the factors paid."""

    month: date
    holding: Holding
    factors: Factors


@dataclass(frozen=True)
class Sample:
    """Consecutive months, oldest first, each with its holding and factors.

    The first period holds the initial allocation and costs nothing. Every later period whose
    holding differs from the period before is a switch, charged COST in that month, including
    the last month. Construction checks these invariants once, and engines trust them.
    """

    periods: tuple[Period, ...]

    def __post_init__(self) -> None:
        if not self.periods:
            raise BacktestError("A sample needs at least one month")
        for period in self.periods:
            if period.month.day != 1:
                raise BacktestError(f"{period.month} is not the first of a month")
            if not isinstance(period.holding, Holding):
                raise BacktestError(f"{period.month} holds {period.holding!r}, not a Holding")
            factors = period.factors
            if not isinstance(factors.rf, Fraction) or not isinstance(factors.mkt_rf, Fraction):
                raise BacktestError(f"{period.month} factors are not exact Fractions")
        for earlier, later in pairwise(self.periods):
            if later.month != shift(earlier.month, 1):
                raise BacktestError(f"{later.month} does not follow {earlier.month}")


@dataclass(frozen=True)
class Market:
    """The approved Ken French monthly factors by month, and the read that supplied them.

    read builds it. record carries the vintage digest that a package names as its input.
    """

    factors: Mapping[date, Factors]
    record: ReadRecord

    def sample(self, allocation: Mapping[date, Holding]) -> Sample:
        """Pair each allocated month with its factors.

        The allocation's months define the sample. The earliest holds the initial allocation
        and the latest is the last month.

        Args:
            allocation (Mapping[date, Holding]): Holding per month, keyed by the first of the month.

        Returns:
            (Sample) The allocated months, oldest first, with their holdings and factors.

        Raises:
            BacktestError: If the allocation is empty, skips a month, holds a month the factors
                lack, or holds something other than a Holding.
        """
        months = sorted(allocation)
        missing = [month for month in months if month not in self.factors]
        if missing:
            raise BacktestError(f"No factors for allocated month {missing[0]}")
        return Sample(
            tuple(Period(month, allocation[month], self.factors[month]) for month in months)
        )


def read(catalog: Catalog, store: Path, context: RunContext) -> Market | Refused:
    """Read the approved Ken French monthly factors through the dataset adapter.

    Percent text becomes an exact decimal, so "2.89" is Fraction(289, 10000). No value passes
    through float.

    Args:
        catalog (Catalog): Catalog that must approve french.ff3_monthly.
        store (Path): Dataset store directory.
        context (RunContext): Trusted run authority. Its cutoff bounds the months supplied.

    Returns:
        (Market | Refused) Every supplied month's factors with the read record, or the refusal.

    Raises:
        DatasetError: If the documented vintage is unreadable or malformed.
    """
    supplied = datasets.read(catalog, store, FACTORS, context)
    if isinstance(supplied, Refused):
        return supplied
    factors = {
        date.fromisoformat(row.key): Factors(
            Fraction(row.values["rf"]) / 100, Fraction(row.values["mkt_rf"]) / 100
        )
        for row in supplied.rows
    }
    return Market(factors, supplied.record)


# One return per sample month, oldest first. A float engine returns each float's exact value,
# which Fraction(float) gives without rounding, so TOLERANCE is the only allowance.
type Engine = Callable[[Sample], tuple[Fraction, ...]]


def plain(sample: Sample) -> tuple[Fraction, ...]:
    """Compute each month's return exactly. This is the reference engine.

    A month that keeps the previous month's holding returns RF + w × (Mkt-RF). A switch month
    pays COST on the portfolio value at its start, and the remainder earns the month's return,
    so it returns (1 − COST) × (1 + RF + w × (Mkt-RF)) − 1.

    Args:
        sample (Sample): Months to back test.

    Returns:
        (tuple[Fraction, ...]) One exact return per month, oldest first.
    """
    returns, previous = [], sample.periods[0].holding
    for period in sample.periods:
        kept = 1 if period.holding is previous else 1 - COST
        growth = 1 + period.factors.rf + period.holding.value * period.factors.mkt_rf
        returns.append(kept * growth - 1)
        previous = period.holding
    return tuple(returns)


def vectorbt(sample: Sample) -> tuple[Fraction, ...]:
    """Compute each month's return with vectorbt 1.2.0, the stand-in for the firm backtester.

    vectorbt's cash earns nothing, so T-bills are a second asset beside the market. Orders fill
    at the previous close, so a fee lands inside the month the allocation starts.

    Args:
        sample (Sample): Months to back test.

    Returns:
        (tuple[Fraction, ...]) The exact value of each float return vectorbt computes.
    """
    import numpy as np
    import pandas as pd
    import vectorbt as vbt

    market = np.array(
        [float(period.factors.rf + period.factors.mkt_rf) for period in sample.periods]
    )
    bills = np.array([float(period.factors.rf) for period in sample.periods])
    close = pd.DataFrame({"market": np.cumprod(1 + market), "bills": np.cumprod(1 + bills)})
    w = np.array([period.holding.value for period in sample.periods], dtype=float)
    weights = np.column_stack((w, 1 - w))
    switch = np.r_[False, w[1:] != w[:-1]]
    trade = switch.copy()
    trade[0] = True
    # Only the sold leg, whose target drops to 0, pays, so COST falls once on the starting value.
    fees = np.where(switch[:, None] & (weights == 0), float(COST), 0.0)
    portfolio = vbt.Portfolio.from_orders(
        close=close,
        size=np.where(trade[:, None], weights, np.nan),
        size_type="targetpercent",
        price=close.shift(1).fillna(1.0),
        fees=fees,
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
        init_cash=1.0,
    )
    return tuple(Fraction(value) for value in portfolio.returns().tolist())


@dataclass(frozen=True)
class Agreement:
    """Both engines' returns over one sample and their largest absolute monthly difference.

    month is the earliest month with that difference. difference is exact, because every engine
    returns rationals.
    """

    reference: tuple[Fraction, ...]
    candidate: tuple[Fraction, ...]
    month: date
    difference: Fraction

    @property
    def agreed(self) -> bool:
        return self.difference <= TOLERANCE


def compare(sample: Sample, engine: Engine = vectorbt) -> Agreement:
    """Run the plain reference and another engine over one sample and find their worst month.

    Args:
        sample (Sample): Months to back test.
        engine (Engine, optional): Engine under test, such as vectorbt or the firm backtester.

    Returns:
        (Agreement) Both series, the worst month, and its exact absolute difference.

    Raises:
        BacktestError: If the engine returns a different number of months than the sample.
    """
    reference, candidate = plain(sample), engine(sample)
    if len(candidate) != len(reference):
        raise BacktestError(
            f"The engine returned {len(candidate)} months for {len(reference)} sample months"
        )
    differences = [abs(mine - theirs) for mine, theirs in zip(reference, candidate, strict=True)]
    index = max(range(len(differences)), key=differences.__getitem__)
    return Agreement(reference, candidate, sample.periods[index].month, differences[index])
