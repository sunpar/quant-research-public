# Trailing Spread Persistence in a Synthetic Exchange

Synthetic test fixture. The paper, its authors, market, data, and results are invented. Each labeled block is a locatable passage, equation, or table.

## 1. Data

[p1] The sample contains 40 synthetic assets traded monthly from month 1 through month 120.

## 2. Method

[p2] At each month end, rank assets by their trailing spread and form five equal-weighted portfolios. The long-short portfolio buys the top quintile and sells the bottom quintile.

[eq1] s(i,t) = (1/12) × sum over k = 2..13 of r(i,t-k)

[p3] The trailing spread skips the most recent month, as Equation 1 shows, to avoid short-term reversal.

[p4] We report Newey-West t-statistics. Net returns deduct trading costs.

## 3. Results

[p5] The long-short portfolio turns over 31% of its top-quintile positions each month.

[table1] Long-short gross returns, months 13 to 120.

| Statistic | Value |
| --- | --- |
| Mean monthly return | 0.0050 |
| Newey-West t-statistic | 2.40 |

[table2] Long-short net returns and turnover, months 13 to 120.

| Statistic | Value |
| --- | --- |
| Mean monthly net return | 0.0038 |
| Top-quintile monthly turnover | 0.13 |

[p6] Subperiod results appear in Appendix B.

## Appendix B. Subperiods

[appendix-b] Available to subscribers only.
