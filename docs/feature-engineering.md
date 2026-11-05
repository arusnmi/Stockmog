# Feature engineering

Phase 2 reads normalized OHLCV CSVs, computes features with lazy Polars
expressions, and writes the result to CSV. No SQL database is required or
configured by the project.

## Generate features

```powershell
python main.py features `
  --input "data\raw\NIFTY 50\NIFTY 50_5m.csv" `
  --style intraday `
  --market-index "data\raw\NIFTY 50\NIFTY 50_5m.csv" `
  --volatility-index "data\raw\INDIA VIX\INDIA VIX_5m.csv"

python main.py features `
  --input "data\raw\NIFTY 50\NIFTY 50_1d.csv" `
  --style swing
```

By default, outputs are saved under `data\features\intraday` or
`data\features\swing`. Override the output path with `--output`. The box
lookback defaults to 12 bars for intraday and 10 bars for swing; the relative
volume lookback defaults to 20 bars. Intraday opening range defaults to the
first three available candles in each session. These can be overridden with
`--box-window`, `--relative-volume-window`, and `--opening-range-bars`.

Optional market-index and volatility-index CSVs add their close and one-bar
return as context. Context joins use only strictly earlier timestamps.

## Feature set

- Candle returns (1/5/20 bars), gap percentage, close location, body, wicks, and
  range.
- EMA 9/20/50/200, RSI 7/14, MACD and signal/histogram.
- ATR 14 and ATR percentage, Bollinger Bands/width/%B, ADX 14 and directional
  indicators, stochastic %K/%D, and OBV with its 20-bar average.
- Prior-bar box high/low, width relative to ATR, and breakout distances.
- Relative volume using a prior-bars-only baseline.
- Intraday session VWAP and opening range, published only once the opening
  range has closed.
- Prior completed day's high/low and normalized distances.
- Optional lagged benchmark and volatility-index context.

Features use the current completed candle and past data only. Box and relative
volume baselines exclude the current candle; daily levels use the previous
completed session; opening-range levels are unavailable until their candles
have closed. Indicator warm-up periods are blank, not backfilled. If volume is
missing or zero, volume-dependent features remain blank.

The Kaggle NIFTY 50/INDIA VIX files currently have a volume column containing
only zeros. Consequently, VWAP and relative-volume columns are blank for these
particular files. Do not treat those features as meaningful until a data source
with actual traded volume is used.

Training code can consume the stored CSV lazily:

```python
from datetime import datetime

import polars as pl

from data_pipeline.csv_dataset import scan_training_csv

features = scan_training_csv(
    r"data\features\intraday\NIFTY 50_5m_features.csv"
)
training_window = features.filter(
    pl.col("timestamp") >= datetime(2024, 1, 1)
).collect()
```

## Prepare per-stock Hugging Face data

The `prepare-stock-data` command downloads the Parquet shards from
`Saintforest/indian-stock-market-minute-data`, processes one stock at a time,
and writes compressed CSVs under `data\stocks`. It does not use a SQL database.
The minute split creates `1m`, `5m`, `15m`, `20m`, `30m`, `1h`, and `1d` candles. Both the
minute and daily splits also create `2d` candles (each combines two consecutive
NSE trading sessions) and `trading_week` candles. A trading week is labelled
with its Monday date and includes the actual observed sessions only; the
`trading_days` column records how many sessions it contains, and
`expected_trading_days` and `is_partial` are calculated using the holiday
calendar. Weekends and dates in the configured holiday CSV are excluded.
Intraday bars are anchored to the 09:15 Asia/Kolkata market open; short final
session bars are retained and marked in `is_partial`. All outputs include
OHLCV, RSI 14, MACD, EMA 9/20/50/200, SMA 20/50/200, ATR 14, ADX 14, VWAP,
and Bollinger Bands.

Before preparing data, fill `data\calendar\nse_holidays.csv` with official
full-day NSE market-closure dates covering the years in your selected dataset.
It must have a `date` column in `YYYY-MM-DD` format; `name` is optional:

```csv
date,name
2024-01-26,Republic Day
2024-03-08,Mahashivratri
```

The checked-in file is an empty template, and the command rejects it until at
least one holiday date is added. Keep this calendar complete and current:
unlisted holiday dates are treated as working weekdays when calculating
expected trading days. Saturdays and Sundays are excluded automatically.

Daily-source outputs are clipped to the dataset-wide first and last calendar
dates present in the minute source, so `1d`, `2d`, and `trading_week` files
cover the same date range as the minute dataset. Running `--split both` learns
that range while processing minute shards. Running `--split day` alone must
download and scan all minute shards first to discover the range, then downloads
the daily shards; it does not generate minute candles during that scan.

Install dependencies and prepare a small symbol subset first:

```powershell
python -m pip install -r requirements.txt
python main.py prepare-stock-data --split minute --symbols RELIANCE,TCS,INFY
```

The minute split creates both `1m.csv.gz` and `20m.csv.gz` for each selected
symbol. Once generated, either file can be passed independently to
`train-models` or `backtest`; see [machine-learning.md](./machine-learning.md)
and [backtesting.md](./backtesting.md). The historical model-training command
is separate and is not run by data preparation.

To prepare both the minute and daily datasets for every symbol, run:

```powershell
python main.py prepare-stock-data --split both
```

The minute Parquet source is about 10.5 GB before download, and the full
transformation may take hours and requires additional disk space for generated
CSV outputs. Shards are downloaded and removed one at a time by default. Add
`--keep-shards` to retain them; use `--output-root`, `--download-dir`, or
`--holidays-csv` to choose other locations. Choose `--split minute` or
`--split day` to prepare only one dataset split. `--symbols` limits which
stocks are written, but the source shards still need to be downloaded because
each shard contains many symbols. The generated `2d` candles pair market-open
sessions, so intervening weekends and listed holidays do not break a pair. For
example, to inspect an output:

```powershell
python main.py inspect-csv --file "data\stocks\minute\RELIANCE\5m.csv.gz"
```
