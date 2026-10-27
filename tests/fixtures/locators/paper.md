# Synthetic momentum earns 1% per month

We study the desk`s synthetic momentum signal on simulated venues. We find that the top decile earns 1.2% per month, e.g. in calm regimes.

The model is $r_t = \alpha + \beta m_t$ and we estimate \(\beta = 0.8\) on 40 venues. Trading costs of $5 and $10 per order, or $5-$10 in US $ terms, are not math, and an escaped \$3 fee per unit ($q$) is literal.

$$
r_t = \alpha + \beta m_t + \varepsilon_t
$$

The estimator is
\[ \hat\beta = (X'X)^{-1} X'y \]
and the Sharpe ratio is
\begin{equation}
SR = \mu / \sigma
\end{equation}

| Decile | Return | t-stat |
| :----- | -----: | -----: |
| 1      | 0.4%   | 1.1    |
| 10     | 1.6%   | 3.2    |

![Cumulative returns](figures/cumulative.png "Synthetic signal")
Figure 1: Cumulative returns of the top decile, which outperforms by 3%.

The spread is statistically significant (see Fig. 2 and Eq. 1). The premium is $2\%$ per month. The sample is synthetic.

- Results suggest the signal survives costs. Turnover averages 35% a year.
  Costs take 150 basis points.
> We document a 2% drift in the synthetic book.

Run `make table COST=$x$ RATE=9%` to rebuild the table.

```text
| not | a table |
| --- | ------- |

We find $x$ grows 9% here.
```

Is the effect robust? We report a t-statistic of 2.4!
