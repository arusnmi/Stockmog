# Trading strategies

Phase 3 adds four independent, rule-based signal generators. They use the
causal features in `features.feature_engineering`; they do not place orders,
size positions, or claim profitability. Signals are saved as CSV records for
review and later backtesting.

## Rules

All strategies use the completed candle's close as the entry reference, an
initial stop one ATR (ATR 14) from that reference, and a target two times the
initial risk away (2R). These are signal levels, not simulated fills. The
backtesting phase must model next-bar fills, gaps, slippage, transaction costs,
and which exit level was reached first.

Repeated bars satisfying the same setup emit one signal until the setup turns
off. Intraday setups can trigger again in a new session. Indicator warm-up
periods or missing required features do not generate signals.

| Strategy | Long condition | Short condition |
|---|---|---|
| Box breakout | Close exceeds the prior 12-bar intraday or 10-bar swing box high by at least 0.1 ATR; relative volume >= 1.5 intraday or 1.3 swing | Close breaks the corresponding box low by the same distance and volume threshold |
| Momentum | RSI 14 >= 60, MACD histogram > 0, and relative volume meets the box-breakout threshold | RSI 14 <= 40, MACD histogram < 0, and relative volume meets that threshold |
| Trend following | EMA 9 > EMA 20 > EMA 50, close > EMA 20, and ADX 14 >= 20; intraday also requires close > VWAP | Reverse EMA and close relationships, with ADX 14 >= 20; intraday also requires close < VWAP |
| Mean reversion (experimental) | Close <= lower Bollinger Band, RSI 14 <= 30, and ADX 14 < 25 | Close >= upper Bollinger Band, RSI 14 >= 70, and ADX 14 < 25 |

These are initial testable rules, not optimized settings. In particular,
mean-reversion signals are experimental and should not be used without
independent out-of-sample evaluation.

## Generate and review signals

Generate one strategy's signals from an OHLCV CSV or compressed CSV:

```powershell
python main.py generate-signals `
  --input "data\stocks\minute\RELIANCE\5m.csv.gz" `
  --style intraday `
  --strategy box_breakout
```

Use `--strategy all` to evaluate all four independently in one run. For daily
candles, use `--style swing` and an appropriate daily input, for example
`data\stocks\day\RELIANCE\1d.csv.gz`. The default output is
`data\signals\<style>\<input>_<strategy>_signals.csv`; override it with
`--output`.

Each record contains the symbol, signal timestamp, strategy, style, direction,
entry reference, stop, target, risk per unit, reward/risk ratio, and a short
rule explanation. Empty results still produce a CSV with the review columns.
Signals are not executable orders. Use the Phase 4 [backtester](./backtesting.md)
to simulate next-bar fills, costs, and portfolio risk limits.

The research expansion catalog contains 200 additional setup hypotheses.
See [strategy-catalog.md](./strategy-catalog.md) for executable family-template
approximations, sources, timeframe eligibility, and the `catalog-all` signal
and backtest commands. The original `--strategy all` still selects only the
four core rules described above.
