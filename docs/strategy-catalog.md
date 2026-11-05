# Research strategy catalog

[strategy-catalog.csv](./strategy-catalog.csv) contains 200 individually named
OHLCV setup hypotheses, their entry descriptions, compatible bar styles,
feature needs, source references, and a high-level exit template. `I` means
intraday bars; `S` means swing bars. `IS` setups are eligible for either style.
The catalog includes 1-minute through 1-hour intraday data and daily, 2-day,
and weekly swing data where those bars are available.

## Executable signal templates

The CLI recognizes each row as `catalog_001` through `catalog_200`. These are
research candidates, not 200 proven or independently validated strategies.
For an initial executable baseline, the signal engine maps the candidates into
nine causal OHLCV templates: trend, breakout, reversal, volatility, volume,
gap, intraday, calendar/trend, and candle confirmation. The catalog ID also
limits eligible bar styles and restricts a one-direction hypothesis when its
description explicitly identifies long or short exposure.

This mapping is deliberately disclosed: entries sharing a template can produce
the same signals, and template thresholds are general-purpose defaults rather
than a faithful parameterization of every paper, named pattern, or market
convention in the catalog. The catalog's written hypothesis is the research
specification; independently translating and validating each candidate is
necessary before treating its backtest as distinct evidence. Stops and targets
use the common ATR-based signal levels, while the backtester applies next-bar
fills, fees, slippage, and portfolio risk controls.

The technical feature set includes causal 5/20-bar returns, candle location and
shape, stochastic %K/%D, OBV, directional indicators, and existing moving
averages, RSI, MACD, ATR, ADX, Bollinger Bands, relative volume, and VWAP proxy.
VWAP is derived from OHLCV typical price and volume, not trade-level prints.
Intraday time-of-day volume normalization, point-in-time fundamentals, news,
order-book data, and cross-sectional peer rankings are not available in every
input, so those catalog hypotheses remain approximations.

## Run signals and benchmarks

Generate one catalog setup:

```powershell
python main.py generate-signals `
  --input "data\stocks\minute\RELIANCE\5m.csv.gz" `
  --style intraday `
  --strategy catalog_121
```

Generate all setups compatible with that style:

```powershell
python main.py generate-signals `
  --input "data\stocks\minute\RELIANCE\5m.csv.gz" `
  --style intraday `
  --strategy catalog-all
```

Use the same `--strategy catalog-all` option with `backtest` to produce one
costed result directory per eligible setup plus a comparison CSV. This can be
computationally expensive; start with one symbol and timeframe. Existing `all`
continues to mean only the original four core strategies.

Train one model suite for one candidate only when explicitly ready; no training
is started by catalog signal generation or backtesting:

```powershell
python main.py train-models `
  --input "data\stocks\minute\RELIANCE\5m.csv.gz" `
  --style intraday `
  --strategy catalog_121
```

Training still requires the optional dependencies and explicit command
execution. See [machine-learning.md](./machine-learning.md) for checkpoints,
pause/resume, and the other model stages.

## Research sources

The references are evidence for broad topics and research methods. They do not
establish that every catalog row or chosen implementation is profitable.

| Key | Source |
|---|---|
| S1 | Jegadeesh & Titman, “Returns to Buying Winners and Selling Losers” ([DOI](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)) |
| S2 | Moskowitz, Ooi & Pedersen, “Time Series Momentum” ([DOI](https://doi.org/10.1016/j.jfineco.2011.11.003)) |
| S3 | Brock, Lakonishok & LeBaron, “Simple Technical Trading Rules and the Stochastic Properties of Stock Returns” ([DOI](https://doi.org/10.1111/j.1540-6261.1992.tb04681.x)) |
| S4 | Lo, Mamaysky & Wang, “Foundations of Technical Analysis” ([NBER](https://www.nber.org/papers/w7613)) |
| S5 | Blume, Easley & O’Hara, “Market Statistics and Technical Analysis: The Role of Volume” ([DOI](https://doi.org/10.1111/j.1540-6261.1994.tb04424.x)) |
| S6 | Jegadeesh, “Evidence of Predictable Behavior of Security Returns” ([DOI](https://doi.org/10.1111/j.1540-6261.1990.tb05110.x)) |
| S7 | De Bondt & Thaler, “Does the Stock Market Overreact?” ([DOI](https://doi.org/10.1111/j.1540-6261.1985.tb05004.x)) |
| S8 | Gao et al., “Market Intraday Momentum” ([DOI](https://doi.org/10.1016/j.jfineco.2017.11.009)) |
| S9 | Marshall, Young & Rose, “Candlestick Technical Trading Strategies: Can They Create Value for Investors?” ([DOI](https://doi.org/10.1016/j.jbankfin.2005.08.001)) |
| S10 | Park & Irwin, “What Do We Know About the Profitability of Technical Analysis?” ([DOI](https://doi.org/10.1111/j.1467-6419.2007.00519.x)) |
| S11 | George & Hwang, “The 52-Week High and Momentum Investing” ([DOI](https://doi.org/10.1111/j.1540-6261.2004.00695.x)) |
| S12 | Fidelity, [Technical Indicator Guide](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/technical-indicator-guide) |
| S13 | Fama & French, “Business Conditions and Expected Returns on Stocks and Bonds” ([DOI](https://doi.org/10.1086/260910)) |
| S14 | Ariel, “A Monthly Effect in Stock Returns” ([DOI](https://doi.org/10.1111/j.1540-6261.1987.tb02321.x)) |
| S15 | Parkinson, “The Extreme Value Method for Estimating the Variance of the Rate of Return” ([DOI](https://doi.org/10.1086/296071)) |

Candlestick patterns, pivots, and other multi-bar structures must be confirmed
without future-bar leakage. OHLCV backtests also cannot resolve intrabar stop
versus target ordering, queue position, or actual bid/ask execution. Results
need out-of-sample and multiple-testing controls, survivorship and corporate
action handling, and realistic costs; even a positive backtest is not evidence
of live profitability.
