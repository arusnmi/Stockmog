# Backtesting

The Phase 4 backtester reuses the Phase 3 signal rules and simulates close
signals at the next available candle's open. It supports one or more OHLCV CSV
or CSV.GZ inputs in a portfolio-wide chronological run. `--strategy all` runs
each strategy independently against the same starting capital so the results
can be compared without one strategy consuming another's portfolio allocation.

## Execution assumptions

- Entry signals are known only after the signal candle closes. Entries fill at
  the next candle's open, with adverse configured slippage.
- Stops and targets are derived from the signal's risk distance and adjusted
  around the actual simulated entry fill. A gap through a stop/target exits at
  the worse/better opening price, respectively.
- If both stop and target fall within one candle's range, the stop is assumed
  to occur first. OHLC bars cannot establish intrabar order, so this is the
  conservative assumption.
- Intraday positions are closed at the last available candle of the session.
  A signal on the last candle is not carried into the next session. Swing
  positions can cross sessions and are closed at the final available candle
  when the dataset ends.
- There is no leverage: whole-share quantity is limited by risk budget and
  available cash. Only one position per symbol is allowed at a time.

Execution uses configurable per-side basis points for commission, other fees,
and slippage. Defaults are 3 bps commission, 7 bps other fees, and 5 bps
slippage per side. Gross P&L is calculated from unadjusted market prices;
slippage and fees are reported separately and both included in total
transaction costs and net P&L. These are illustrative research assumptions,
not a quote for an exchange or broker tariff. Replace them with verified
broker- and product-specific charges before interpreting results. Run
configuration values are included in each `summary.json`.

## Portfolio controls

Defaults follow the plan's initial simulation limits:

- Initial capital: Rs 10,00,000.
- Risk per trade: 0.25% of current realized equity, including estimated
  round-trip costs when sizing.
- Maximum open risk: 1% of equity, including estimated round-trip costs.
- Maximum daily loss: 1%; open positions are marked to market, liquidated at
  the available close when breached, and new entries are blocked for the day.
- Maximum drawdown circuit breaker: 5%; it blocks new entries after the
  marked-to-market equity falls from its peak by that amount.
- Maximum positions: 3 intraday or 5 swing, overridable from the CLI.
- Sector exposure and correlation controls are not yet modeled.

These are simulation settings, not investment advice or live-trading
authorization. A stop gap can exceed the planned risk. Short sales are
simulated symmetrically and do not yet model borrow fees, availability, or
Indian-market product-specific restrictions.

## Run and inspect

Run one symbol:

```powershell
python main.py backtest `
  --input "data\stocks\minute\RELIANCE\5m.csv.gz" `
  --style intraday `
  --strategy all
```

The same command accepts any generated intraday timeframe, including 1m and
20m. Run intervals separately for meaningful comparisons:

```powershell
python main.py backtest `
  --input "data\stocks\minute\RELIANCE\1m.csv.gz" `
  --style intraday `
  --strategy all

python main.py backtest `
  --input "data\stocks\minute\RELIANCE\20m.csv.gz" `
  --style intraday `
  --strategy all
```

Add inputs with repeated `--input` options to simulate symbols together:

```powershell
python main.py backtest `
  --input "data\stocks\minute\RELIANCE\5m.csv.gz" `
  --input "data\stocks\minute\TCS\5m.csv.gz" `
  --style intraday `
  --strategy box_breakout
```

Outputs are written to `data\backtests\<style>`. Each strategy has its own
directory containing `trades.csv`, `equity.csv`, `yearly.csv`, and
`summary.json`; `comparison.csv` summarizes the independent strategy runs.
Override defaults with `--initial-capital`, `--risk-per-trade`,
`--max-open-risk`, `--max-daily-loss`, `--max-drawdown`, `--max-positions`,
`--commission-bps`, `--other-fees-bps`, and `--slippage-bps`. Risk fractions
are decimal values, so `0.0025` means 0.25%.

Reports include gross and net P&L, fees, slippage, total transaction costs,
return, win rate, average win/loss, expectancy, profit factor, maximum
drawdown, Sharpe and Sortino ratios, turnover, and annual marked-to-market
return/drawdown plus realized trade summaries. Ratios are descriptive
historical statistics, not estimates of future performance. The tool does not
perform walk-forward validation, model parameter selection, sector/correlation
analysis, or benchmark comparison.
