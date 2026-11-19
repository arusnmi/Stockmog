# Machine learning: Phases 5–7

The model pipeline is optional. No training runs automatically. The `train-models`
command creates cost-aware outcome labels from rule-based strategy candidates,
builds causal features and 60-bar sequences, then makes chronological
train/validation/test splits. Examples whose forward outcome crosses a split
boundary are purged. Feature scaling is fitted on training examples only.

The four core strategies and the 200 research-catalog IDs can be selected
individually for signal generation and model preparation. Catalog setups are
initial family-template approximations, as documented in
[strategy-catalog.md](./strategy-catalog.md); one setup is trained per
`train-models` invocation. Training is never started automatically.

Labels are based on simulated next-bar entry, stop/target priority, the configured
forward horizon, and fees/slippage. Net returns below -0.1R are labeled loss,
above +0.1R profitable, and the interval between them near break-even. Intraday
signals do not cross sessions. The default horizon is 12 bars intraday and 10
bars for swing strategies.

## Install training dependencies

Use the project's supported Python environment and install the optional model
packages only when ready:

```powershell
python -m pip install -r requirements.txt -r requirements-training.txt
```

For a CUDA-enabled Windows PyTorch build, install the GPU-specific requirements
instead:

```powershell
python -m pip install -r requirements.txt -r requirements-training-gpu.txt
```

The GPU requirements pin a PyTorch CUDA wheel tested with the project's Python
3.14 environment and current NVIDIA driver. The trainer uses CUDA for XGBoost
when the installed XGBoost build includes CUDA support. PyTorch uses CUDA for
the LSTM when its CUDA build detects the GPU. On Windows, cuDNN is disabled for
LSTM training because it caused a native crash at process exit in the tested
environment; CUDA remains enabled. Check the startup line
`Training devices: XGBoost=..., LSTM=...` before launching a long batch.

The model suite trains XGBoost, Random Forest, Extra Trees, and regularized
Logistic Regression, followed by a small LSTM. XGBoost uses CUDA when available;
the other tree/classical estimators remain CPU-based. The LSTM uses CUDA when
PyTorch detects an available GPU, otherwise it runs on CPU. `--models all`
trains each estimator in turn and calibrates a validation-weighted ensemble.
LSTM progress includes process/system RAM and CUDA memory usage when a GPU is
active.

Each classifier predicts loss, near-break-even, and profitable outcomes. The
LSTM predicts those outcome probabilities plus expected net return in R and a
learned return uncertainty. Regression targets are clipped to [-5R, 5R] for
stability. The hybrid blend weights are selected by validation log loss.
By default the signal filter accepts candidates when their predicted profitable
outcome probability is at least 0.22, regardless of expected return or
uncertainty. This experimental threshold came from one symbol's validation
period; it is not evidence that the rule improves accuracy or profitability
across symbols.

## Retrospective Backtrader comparison in Colab

`compare-hybrid-models` evaluates the completed saved hybrid models without
training or modifying their weights or existing score files. For every recorded
input it compares the experimental 0.22 filter, the old 0.45 filter (positive
expected return and uncertainty <= 0.95), and no trading. This uses the
models' recorded test boundaries, which have already been inspected; the
results are exploratory, **not an independent forward test**. Original training
event sampling also depended on the total future event count. Only completed
models and their recorded inputs are included; skipped or failed model jobs
without a completed retry are not evaluated.

Copy the current project **including uncommitted changes**, the two completed
model batch directories and the original stock files to your own Drive. Preserve
`data/stocks/minute/<symbol>/<interval>.csv.gz` and
`data/stocks/day/<symbol>/<interval>.csv.gz`. The full stock dataset may need
hundreds of GB; check available Drive storage before copying it. File size is
checked against the saved training records, but this alone does not establish
that a copied file has identical contents.

In Colab, mount Drive and set the working directory to your copied project
(adjust the example path):

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Stockmog
```

In this checkout `requirements.txt` contains extra `hf` and `0` entries, so
the following installs the actual base dependencies explicitly instead of
using that file. Keep the same package versions when resuming:

```python
%pip install "numpy>=1.26,<3" "pandas>=2.2,<3" "polars>=1.0,<2" "yfinance>=0.2,<1" "huggingface_hub>=0.26,<2" "pyarrow>=17" -r requirements-training.txt "backtrader==1.9.78.123"
```

Run a small pilot first. Output must be a new directory outside the data and
model directories; use persistent Drive storage. Change these paths if your
copy is elsewhere:

```python
!python main.py compare-hybrid-models \
  --model-roots data/models/batch-runs/all-timeframes-pooled-canonical data/models/batch-runs/failed-timeframes-retry \
  --stock-data-dir data/stocks \
  --output /content/drive/MyDrive/Stockmog-results/comparison-pilot \
  --intervals 15m --symbols 20MICRONS
```

For the full comparison use a **different, new** output directory and omit
`--intervals` and `--symbols`:

```python
!python main.py compare-hybrid-models \
  --model-roots data/models/batch-runs/all-timeframes-pooled-canonical data/models/batch-runs/failed-timeframes-retry \
  --stock-data-dir data/stocks \
  --output /content/drive/MyDrive/Stockmog-results/comparison-full
```

Restart an interrupted run with the identical command and `--resume`. The
command checks saved input sizes and checkpoint
size/modification times, model SHA-256 hashes, relevant project Python source
hashes, package versions and configuration; a mismatch stops the run. Keep the
same paths on resume. A missing input, changed model or invalid simulation is
an error, not a successful result. Runs checkpoint after each symbol/model
episode; check that `comparison-run.json` and `comparison-summary.json` both
say `complete`. Each run is single-process; do not start concurrent copies.
Pilot timing does not guarantee full-run duration, Colab availability or cost.

Each symbol/model episode starts independently with INR 50,000 and targets 10%
equity exposure. Commission, other fees and slippage come from the original
batch configuration (the current batches record 3, 7 and 5 basis points per
side). Entries and exits fill at the next bar open, with horizon/session exits,
not the training-label stop/target rules. Shorts are intraday only. An intraday
order crossing a date fails explicitly; missing bars can therefore interrupt a
run. A 50% peak drawdown triggers next-open liquidation, not a guaranteed loss
limit. To avoid unfillable end-of-data orders, new entries are disallowed on
the last two bars and an existing final position is marked to market. Closed-trade
win rate excludes any open final position.

The summary groups P/L and closed-trade counts by timeframe and strategy, with
per-symbol results in `symbols/`. Maximum drawdown is reported per episode.
Totals sum **rounded** P/L from independent accounts; percentages use their
combined starting capital. They do not model a shared-capital portfolio. Fewer
than two test bars is recorded as insufficient rather than silently treated
as a successful backtest.

Do not tune thresholds, position sizing, exits or drawdown limits on these
already-inspected tests. Those follow-up steps need a validation-only selection
procedure and genuinely later untouched data; retraining, if warranted, is a
separate user-initiated step.

## Nifty 50 price-model research

The new daily-price regression components, research-only workflow commands,
holdout evaluation and per-stock graphical reports are implemented.
The production workflow is **not implemented**.
`modeling.nifty_universe` provides historical membership parsing
from the [CC0 point-in-time membership dataset](https://github.com/aditya-jha/nse-historical-membership).
Its source revision is pinned; lookups are limited to April 2018 through
October 13, 2026. Historical symbol renames and corporate actions must be
verified against local files before using them for training. The verified
2023-07-13 to 2024-09-29 `LTM` membership is mapped to local `LTIM`; the
mapping is rejected outside that interval.

### Data inventory and causal preparation

The local candidate catalog inventories historical members and Nifty-named
index/fund series, their available timeframe files, and actual minute-derived
and independent daily-source coverage. These are overlapping market proxies, not 138 independent
economic or global factors. A file's presence does not establish complete
intraday coverage. Missing-session checks use minute-derived RELIANCE sessions,
not an authoritative exchange calendar; for example, that reference omits
2022-10-24 although the local RELIANCE daily source contains that date.
The configured local source is
[Saintforest/indian-stock-market-minute-data](https://huggingface.co/datasets/Saintforest/indian-stock-market-minute-data);
its card declares MIT, but the revision used to build the local files was not
recorded. Verify data rights before redistribution.

```powershell
python -m modeling.nifty_preparation catalog --output "data\external\new_nifty_catalog"
python -m modeling.nifty_preparation prepare --output "data\external\new_nifty_prepared" `
  --stocks RELIANCE LTM --factors NIFTY
```

Omit `--stocks` and `--factors` to select all historical members and local
candidates. Use `--stocks` or `--factors` with no values to select none of
that kind, and `--timeframes` to select a subset. The default is all nine
timeframes. Prepared stock and factor files are kept separate, in Parquet
chunks, not expanded into a large stock-by-factor matrix. The preparation
records missing inputs but cannot repair gaps within an existing file.
Run `python -m modeling.nifty_preparation coverage --output <run-directory>`
after preparation to create `coverage.json`, which distinguishes missing
files from series with no eligible rows and reports prepared row counts.

The separately stored daily source includes some dates absent from minute-derived
files, including special evening sessions. Prepare it separately, without
inventing intraday data:

```powershell
python -m modeling.nifty_preparation prepare --output "data\external\new_daily_source" `
  --candle-source day --timeframes 1d 2d trading_week
```

For these daily-source bars, `available_at` is conservatively the following
midnight IST, to avoid assuming that special sessions close at 15:30. This is
not an audited vendor publication time. Two-day bars combine consecutive
observed sessions, not necessarily consecutive official sessions; the second
session's following midnight is their availability time. Short weeks are
marked partial and weekly bars become available the next Monday. Daily-source
multisession input must fit in `--chunk-rows`; increase it if preparation reports
that it is too small. Do not combine these bars with minute-derived aggregates
without accounting for their different session coverage.

Output includes UTC `available_at`, not just the bar's start label. Intraday
bars become available at their scheduled end (capped at 15:30 IST); minute-derived daily
bars at 15:30 IST; two-session bars at the second session close; and weekly
bars conservatively at the following Monday midnight IST. Incomplete
two-session bars are excluded. Historical stock membership must cover the
entire aggregate candle. The `join_available` helper uses a backward-only
join with a required maximum age; missing/stale values stay missing. A decision
at exactly `available_at` assumes zero publication and processing latency.
These rules assume standard NSE session hours and the local pipeline's bar
conventions; unusual sessions require separate verification.

Use the existing `training-control pause|resume|stop|status` commands with the
catalog or preparation output directory. Cataloging checks the controls between
series; preparation checks between input chunks and jobs. After a stop or
interruption, rerun the identical command with `--resume`; already written
chunks are checksum-verified and inputs/settings must match. Input identity uses
file sizes and modification times, not content hashes for the large candle
files, so it cannot detect deliberately preserved timestamps and sizes.
Do not run two workers against the same output directory.

HDFC daily data was recovered from the pinned
[olo-labs NSE release](https://github.com/olo-labs/olo-ai-training-data-nse).
The recovered file has 378 unique sessions covering HDFC's local membership
period, after removing 97 identical duplicate rows. Its dates match the local
daily RELIANCE reference; this is not an authoritative exchange-calendar check.
The release code is Apache-2.0, but its underlying market-data rights need
independent verification. To reproduce the download and validation:

```powershell
python -m data_pipeline.nifty_hdfc_daily --output "data\external\new_hdfc_daily_source"
```

The collector validates a pinned archive checksum, rejects conflicting duplicates
and missing reference sessions, and will not overwrite an unrelated destination.
It supports the same controls and `--resume`.

The public [HDFC intraday archive](https://github.com/gsidhu/nse-intraday-data)
can be collected separately:

```powershell
python -m data_pipeline.nifty_hdfc_archive --run-dir "data\external\hdfc_archive"
```

It has missing sessions and no data in that archive for March-July 2023; a
December file also contained the wrong date and was quarantined. Its README
states MIT, but no separate license file
was found. Raw timestamps and data rights still require verification, so these
files are **not** merged into the prepared dataset. The collector supports the
same controls and `--resume`.

An exploratory daily correlation audit can be run with:

```powershell
python -m modeling.factor_research `
  --membership "data\external\nse_index_membership_0786cd915974.csv" `
  --stocks "data\stocks" `
  --output "data\external\new_factor_audit" `
  --train-end 2025-01-01
```

The output directory must not already exist for the first run. Preparation
checkpoints each completed daily factor and stock file. To pause, resume or stop
from another terminal, use the same output directory:

```powershell
python main.py training-control pause --run-dir "data\external\new_factor_audit"
python main.py training-control status --run-dir "data\external\new_factor_audit"
python main.py training-control resume --run-dir "data\external\new_factor_audit"
python main.py training-control stop --run-dir "data\external\new_factor_audit"
```

Pause and stop take effect after the current daily input file or stock is
processed; they do not interrupt a file read in progress. A stopped or interrupted
run can be restarted with the original audit command plus `--resume`; matching
completed checkpoints are reused. Resume refuses changed input-file sizes or
modification times, membership contents, or settings. File size and modification
time alone do not prove stock-file contents are unchanged.

The audit screens local Nifty-named market indices and funds, not a complete set
of independent global factors.
It correlates each factor's same-session return with the stock's next-session
return using only sessions and labels strictly before the training cutoff,
requires both stock dates to fall within the same membership interval, and
discards returns crossing missing RELIANCE-reference sessions. It assumes
same-session factors are available by next midnight IST; actual publication
latency is unverified. It applies the absolute 0.7 threshold. Missing stock files and stocks without
enough paired observations are reported separately from stocks with no qualifying
factor. This is exploratory end-of-day research, not an intraday dataset,
price-model training, or proof of trading performance.
The corrected audit uses manifest format 3 and cannot resume older format 2
results; start it in a new directory.

To also screen all local daily stock and fund files, run the same audit with
`--include-peers` and a new output directory. This uses format 4, excludes the
target stock's local file, and cannot be consumed by the existing format-3
selection command. Resume requires the same flag and `--resume`.

These are peer-price candidates, not independent economic variables. The local
catalog is retrospective, can contain renamed or delisted securities, and can
introduce survivorship bias. Checking thousands of correlations increases the
chance of finding apparently strong relationships by chance. Source rights,
corporate actions and publication timing still require verification.
In one pre-2025 audit of 2,536 local candidates, 119,113 stock/peer pairs
were scorable; none reached absolute correlation 0.49. The largest was
approximately 0.468, based on 90 paired sessions. Twelve of 67 historical
members had no scorable peer. This is a next-daily-return screen, not a
next-candle or out-of-sample performance test.

### Global factor research

The 100 additional US-listed ETF and fund proxies are drawn from
[syncster-data/historical-market-data](https://huggingface.co/datasets/syncster-data/historical-market-data),
pinned to revision `eff0dac55d843b3b83caec819efa213021a3e3e5`. The dataset
card says research and education only; corporate-action adjustments, vendor
publication latency, and commercial or redistribution rights remain unverified.
The funds overlap and are not 100 independent economic variables. Each source
file is checked against its pinned LFS SHA-256. The collection covers all nine
timeframes, with actual observations only; rows outside regular US weekday
hours are excluded and missing bars are not invented. Daily and weekly outputs
flag incomplete bars. Use separate output directories and the same controls as
above; download checkpoints occur at blocks and preparation checkpoints between
timeframes. Resume using `--resume`:

```powershell
python -m data_pipeline.global_factor_data --output "data\external\new_global_factors"
python -m modeling.global_factor_research `
  --factors-root "data\external\new_global_factors" `
  --output "data\external\new_global_audit" --train-end 2025-01-01
```

The default audit compares the latest causally available return from a complete
US session with the next NSE-session stock return. It uses SPY and RELIANCE as
session-calendar proxies, excludes returns across missing proxy sessions,
restricts both stock dates to membership and the training cutoff, and uses a
maximum factor age of five days. US daily bars are conservatively available at
the next New York midnight, so a US session that has not finished by the
decision time cannot be used. The optional `--max-close-age-minutes 5` permits
partial sessions only if their actual last observed minute ends within five
minutes of 16:00 New York and matches the daily aggregate close. This is a
near-close proxy, not a verified official close; early-close sessions are
conservatively excluded. Use a new audit output directory for this alternative.
The actual source timestamp convention and publication delay still need
verification. No correlation result is evidence of predictive performance,
particularly with overlapping funds and multiple testing.

### Per-stock factor selection and extra sector candidates

Additional sector and industry funds can be collected from the same pinned
research-only dataset. This catalog includes SPY as the US session-calendar
proxy plus 36 new tickers; their portfolios may overlap the first 100 funds:

```powershell
python -m data_pipeline.global_factor_data --sector-candidates `
  --output "data\external\new_sector_factors"
python -m modeling.global_factor_research `
  --factors-root "data\external\new_sector_factors" `
  --output "data\external\new_sector_audit" --train-end 2025-01-01 `
  --max-close-age-minutes 5
```

The additional funds are US-listed proxies, not necessarily Indian sector
indices or independent variables. Apply the same source-rights and near-close
limitations described above. No missing observations are invented.

Combine a completed format-3 local audit with one or more completed global
audits using the same membership, cutoff and minimum observation count. Global
audits must also use the same factor-age and near-close settings; in this example
both global audits use `--max-close-age-minutes 5`.

```powershell
python -m modeling.nifty_factor_selection `
  --local-audit "data\external\new_factor_audit" `
  --global-audit "data\external\new_global_audit" `
  --global-audit "data\external\new_sector_audit" `
  --output "data\external\new_factor_selection" --train-end 2025-01-01
```

`selection.json` lists each stock's qualifying factors and their full-precision
correlations. `stocks.csv` distinguishes missing stock data, insufficient
observations and no qualifying factors. The exact absolute threshold is 0.7;
it is never reduced or replaced with a fallback. Matching duplicates from the
same pinned source are counted once. Input identity and audit checkpoints are
checked; pause or stop via `training-control` and rerun with `--resume`.

These are exploratory selections based on data before the cutoff. Using them
for earlier training examples would leak future information; each chronological
fold needs a selection made before that fold's earliest example.
Unverified data rights, adjusted prices, source timestamps, publication times,
and exchange calendars also block claims of training readiness.

### Daily price-model components

`modeling.nifty_candle_examples.build_candle_examples` is a separate
next-candle example builder for all nine prepared timeframes. It pairs complete
consecutive candles within historical membership and returns the actual next
close and its availability time. Intraday examples do not cross sessions or
gaps; daily and multisession examples use the supplied reference calendar.
This builder is **not yet connected** to the price trainer, factor selection,
evaluation or report commands. It is not a trained next-candle model.

An additional reproducible exploratory screen collects four INR exchange-rate
series, the India 10-year yield, ten India indices, and gold and silver from the pinned
[Indian market dataset](https://github.com/aswinv90/indian-market-historical-data/tree/eb3d7aa4639eb9434409b3950d9730d4ae9a6624).
The source's license restricts use to non-commercial research, personal use
and education. Its attribution, publication timing, revision history and
adjustment methods require review. The screen assumes daily values are available
by the following midnight IST; this assumption is unverified. Sparse commodity
histories may not yield enough consecutive observations. Monthly CPI and policy
series are excluded because their release dates are unverified.

```powershell
python -m modeling.nifty_public_factor_research `
  --membership "data\external\nse_index_membership_0786cd915974.csv" `
  --stocks-root "data\stocks" --output "data\external\new_public_factor_audit" `
  --train-end 2025-01-01
```

The output directory must be new. A stopped run can be continued with the same
command plus `--resume`; the usual training controls apply at file and stock
boundaries. Resume checks pinned Git blobs, input fingerprints and source code
identity. Stock fingerprints use size and modification time, not content hashes.
One audit produced 817 scorable stock/series pairs across 55 stocks before
2025-01-01; 12 stocks had insufficient observations and none of these pairs
reached an absolute correlation of 0.49 with the next daily stock return. This
is neither a next-candle screen nor an exhaustive factor search. These results
are not integrated into factor selection or training.

Additional U.S. Treasury nominal, real and bill-rate levels and changes can be
screened directly from the [Treasury daily rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView)
using a new directory:

```powershell
python -m modeling.nifty_treasury_factor_research `
  --membership "data\external\nse_index_membership_0786cd915974.csv" `
  --stocks-root "data\stocks" --output "data\external\new_treasury_audit" `
  --train-end 2025-01-01
```

Add `--resume` after a stop; the usual controls work between files and stocks.
Downloaded snapshots are retained and checked against their recorded SHA-256
hashes. One audit screened 60 overlapping rate features and 3,294 scorable
stock/feature pairs before 2025-01-01; none reached absolute 0.49. Twelve
stocks had insufficient observations. This audit assumes rates are available
at the next New York midnight, but neither publication latency nor historical
revisions have been verified. Snapshot hashes do not establish point-in-time
vintages. This is exploratory daily research and is not integrated into
selection or training.

An additional exploratory screen compared RBI/FBIL reference exchange rates
from the pinned [reference-rates archive](https://github.com/the-solipsist/rbi-forex-reference-rates/tree/7fa5b91b17b9f79d74805c9db1b26ff5727c49e2)
and EIA Brent/WTI prices via [DataHub](https://datahub.io/datapressr/oil-prices).
None of 660 scorable stock/feature pairs reached absolute 0.49. The INR series
were assumed available the following midnight IST; crude observations were
delayed seven days. Publication timing, historical revisions and underlying
data rights remain unverified. AED and IDR reference histories begin in 2026
and cannot be used in the pre-2025 screen.

Do not use FRED or ALFRED data to develop or train the model: their
[terms](https://fred.stlouisfed.org/legal/) prohibit use of their services or
content in machine-learning development or training. Direct Treasury data above
is obtained from Treasury, not FRED.

### Historical same-candle correlations

The same-candle screen is separate from the next-candle screen below. For each
historical Nifty stock and factor, it matches complete historical candles at the
same timeframe and calculates overall Pearson and Spearman coefficients for
closing prices and for consecutive close-to-close returns. Treasury and bond
yield changes are absolute rate changes. A return pair is used only when both
series have the same previous candle interval. Stock candles must fall within
verified Nifty membership periods. The first and last matched candle and the
number of pairs are reported for each measure; no dates are filled in.

Run all available local peers, global factors, and the saved daily economic
factor snapshots:

```powershell
$corrArgs = @(
  '--membership', 'data\external\nse_index_membership_0786cd915974.csv',
  '--stocks', 'data\stocks',
  '--output', 'data\external\nifty_same_candle_correlations',
  '--global-factors', 'data\external\global_factor_research',
  '--global-factors', 'data\external\nifty_extra_sector_factors',
  '--india-audit', 'data\external\nifty_all_factors_india_2025',
  '--treasury-audit', 'data\external\nifty_all_factors_treasury_2025',
  '--prices-audit', 'data\external\nifty_all_factors_prices_2025'
)
python -m modeling.nifty_same_candle_correlations @corrArgs
```

Each stock has a separate CSV under the output's `stocks` directory. Fewer
than three pairs, constant values, missing candles, and factors without that
timeframe are reported, not substituted with another timeframe. Daily economic
snapshots only supply daily observations. Two-day and weekly comparisons
require the same underlying sessions. US and Indian daily closes on the same
date occur at different times; this is a retrospective date match, not a
point-in-time feature.

This full screen repeatedly reads a large amount of compressed history and
computes ranks for each pair. It may take a long time; no 24-hour completion
is guaranteed. The existing `training-control` pause, resume, and stop commands
work with this output directory. Restart a stopped run with the same arguments
and `--resume`; an interrupted stock is recomputed. Source rights, adjustment
methods and publication times remain unverified. These contemporaneous
correlations, especially price levels, do not establish predictive power or
profitability and must not be used as if they were next-candle predictions.

### Separate stock/timeframe price and trend models

The completed same-candle scan can supply the historical stock universe to
`modeling.nifty_timeframe_workflow`. This is a separate workflow from the older
pooled daily trainer below. Nothing trains on import or during `prepare`.
Run these commands from the repository root in **Command Prompt**, one at a
time, only when you want each stage to start. `train` automatically prepares
the data first, then starts the parallel training workers:

```bat
nifty.cmd train --research-only
nifty.cmd evaluate
nifty.cmd predict --symbol INFY --timeframe 1d
```

In PowerShell use `.\nifty.cmd` instead. `prepare` remains available for an
explicit preparation-only check, but is not needed before `train`.
The wrapper changes to its own directory,
forwards arguments, and preserves Python's exit code. There is no `$corrArgs`
variable to lose when reopening a terminal. The equivalent entry point is
`python -m modeling.nifty_timeframe_workflow`. Defaults use the completed
`data\external\nifty_same_candle_correlations` scan and write a **new** run under
`data\models\nifty-timeframe`. They do not reuse older trained models.

To make a rolling one-day-ahead daily prediction series with an already trained
run, use:

```bat
nifty.cmd predict-series --symbol BSE
```

The command keeps saved daily candles and downloads only later daily candles
from the existing Yahoo Finance provider. It does not check the two feeds for
consistent price adjustments: the join can create a false jump and distort
features or predictions. It requires the stock and Nifty reference series to
reach the same latest session. Before 15:45 IST it excludes the current date; afterward it can use
that session if Yahoo has published its daily candle. Targets start at the
saved model's holdout boundary, and each prediction uses only candles through
the preceding target session. The CSV and summary are written to
`data\models\nifty-timeframe\jobs\BSE\1d\prediction-series.csv` and
`prediction-series-summary.json`. This does not train a model or change saved
stock data. Actual closes are included only for historical comparison. Yahoo
Finance is not an exchange-verified source, so the reported latest session is
the latest complete session returned by that source, not a guarantee of
exchange publication or accuracy.

For a combined stock forecast report covering the latest **20 completed
sessions** of every prepared daily model:

```bat
nifty.cmd forecast-report
```

In PowerShell use `.\nifty.cmd forecast-report`. To report only selected stocks:
`nifty.cmd forecast-report --symbols BSE INFY`. A new timestamped folder under
`stock_forecasts` contains
`stock_forecast_report.pdf`, individual PNGs, the underlying per-stock CSVs,
and `report.json`. The PDF includes an executive summary, methodology,
stock-by-stock metrics and graphs, daily results, and source notes. To build a
detailed PDF from an existing report folder without fetching or predicting
again, use `nifty.cmd forecast-report --from-report <report-folder>`; it checks
the saved CSV hashes and does not overwrite an existing PDF. Report CSVs are
ignored by Git and must be retained locally to rebuild a saved report. The command
prints its output location. It may take considerable time and requires Yahoo
access and matplotlib from `requirements-forecast.txt`. Each graph shows actual
and predicted closes plus separate dashed linear trend fits over the displayed
20 sessions. Those fitted lines are retrospective summaries, not additional
predictions. Selected external and local factor names and their source
identifiers appear below each graph; missing counts describe decision-row factor
values. Selected stock factors and supported Indian index factors are extended
with later Yahoo data where available. Yahoo ticker mappings and price scales
are not verified against the saved source. Factor update status and errors are
shown in the report, and values beyond the workflow's existing availability
tolerance remain missing. BSE's daily model has no
selected factors. If a stock lacks data or a completed model, it is listed as a
failure in the output and the command exits nonzero; the PDF contains the
successful stocks only. Generating this historical comparison does not retrain
models or predict a future session.

The training universe is the union of **61 verified Nifty 50 members** whose
membership overlaps January 2022 through the available 2026 membership records
(verified through October 13, 2026). A stock need not remain in the index
throughout that period. For those stocks, all available candles from January
2022 onward are eligible, including dates before joining or after leaving.
Local stock-factor candidates use the same universe; saved global/economic
factors remain available. Stocks that left before 2022, such as AUROPHARMA, are
not training targets or local stock factors. No source files are deleted by
this filtering. Local price history currently ends in January 2026; membership
records do not create missing later prices.

This is an explicitly retrospective research universe, **not** a point-in-time
Nifty membership strategy. It can introduce universe-selection/survivorship
bias. The original completed correlation scan retains its original membership
policy; new training-period correlations are recalculated under this full-history
policy. Other legacy workflows retain their original membership rules.
The saved same-candle scan was subsequently filtered to the 61-stock universe:
excluded stock targets and local-stock factors were removed, while retained
correlations were not recomputed. Its checksums describe the filtered outputs;
the original scan cannot be resumed. The older 2025 scan still contains
unclassified local factors that may include nonmember stocks, indexes or ETFs.

#### Auditing and reconstructing candle gaps

Stop preparation/training before changing source candles. Audit all 61 stocks
and nine training timeframes without modifying them:

```bat
python -m data_pipeline.nifty_candle_repair --report candle-audit.csv
```

Add `--apply` to write verified repairs. Only complete finer candles may form
a coarser OHLCV bar, with exact, nonoverlapping interval coverage. Daily bars
can supply 2d/weekly bars, but cannot supply real intraday prices. Missing
one-minute candles remain missing. Existing complete bars are not overwritten;
intentional session-end partial bars are not invented into complete bars.
The session calendar is a RELIANCE-history proxy, not an official exchange
calendar, and coverage is checked only within each stock's available history.
Repaired rows leave stored indicator columns blank; training recalculates them.

The original correlation scan remains a historical snapshot with verified
output checksums. New preparation fingerprints the **current** source data
and recomputes training-period correlations; it does not pretend the old scan
used repaired inputs. An existing preparation/training checkpoint must not be
resumed after changing its inputs; start a new run instead.

Selection is independent for each stock/timeframe and uses **return**, not
price-level, correlations. Both absolute Pearson and absolute Spearman must
pass. Threshold pairs are `(0.70,0.70)`, `(0.69,0.65)`, `(0.68,0.60)`,
`(0.67,0.55)`, `(0.66,0.50)`, `(0.65,0.45)`, then `(0.64,0.40)` through
`(0.40,0.40)`.
Relaxation stops at the first level admitting any factor; all qualifying
factors at that level are retained, with their signed coefficients. No factor
is invented below the floor. The default minimum is **60 paired returns**.
Candidates include other surviving local Nifty stock histories plus the saved
global and daily economic collections. Self-stock factors are excluded.
Missing timeframes are not replaced with daily values.

The full-history coefficients are **not** used to select factors for earlier
training rows. Each job splits its eligible sessions chronologically at
70%/85%, recalculates factor correlations strictly before its validation
boundary (including availability), then freezes selection for validation and
holdout. Labels spanning split boundaries are purged. Different historical
members can therefore have different split dates. Missing histories, fewer
than ten sessions, and jobs without complete sequences in every split are
explicitly reported in `preparation-summary.json`; selecting 61 stocks does not
mean all 549 stock/timeframe combinations can be trained.

Every job uses the existing causal price/return, candle-shape, EMA, RSI, MACD,
ATR, Bollinger, stochastic, volume and box indicators, plus the four executable
strategy-family conditions: **box breakout, momentum, trend following, and
mean reversion**. They are present both with and without factors. These are
reusable feature/rule families, not proof that every catalog strategy is
equivalent or profitable. Factor returns are joined backward by availability
at the same timeframe, with missing-value flags and maximum ages (one candle
intraday; 7/10/14 calendar days for daily/2d/weekly). Economic snapshots lack
verified publication times; this research workflow assumes end of the following
New York calendar day, which is an assumption, not a verified release schedule.

Each usable stock/timeframe gets its own **XGBoost + LSTM** next-candle log-return
model using the existing resumable trainer. Training defaults to **CUDA and two
parallel worker processes**, each processing one independent stock/timeframe at
a time. Windows uses the multiprocessing `spawn` start method so CUDA contexts,
random seeds and checkpoints are isolated; full datasets are loaded inside
workers, not copied through the task queue. Each worker limits PyTorch CPU
threads to two. Preparation remains sequential.
Validation chooses checkpoints and
the blend, including pure-tree/pure-LSTM endpoints; no architecture is
guaranteed superior before evaluation. Price is anchor close times the
exponential of predicted log return; trend is **up/down/flat** from that same
prediction, not a calibrated classification probability.

Sequences do not cross missing/partial candles or intraday session boundaries.
Their maximum length is 20, capped to 300 minutes' worth of bars intraday, ten
bars for 2d and three for weekly so short sessions/holiday weeks remain usable.
`evaluate` writes timestamped holdout price/trend predictions and per-job
MAE/RMSE/MAPE/directional accuracy against a last-close persistence baseline.
Never tune repeatedly against this reserved holdout. The baseline is not an
execution backtest or a claim that the strategy features add predictive value.

```bat
nifty.cmd pause
nifty.cmd resume
nifty.cmd stop
```

Progress prints automatically in the running terminal: preparation stock,
timeframe and factor counts; training active jobs, tree rounds, epochs and
batch offsets about every 15 seconds, plus completed-job counts and times.
There is no separate `status` command. The run's JSON checkpoints remain on disk
for integrity and restart support.
If Windows temporarily denies access to a worker's progress file, the terminal
reports that the snapshot is unavailable and checks again on the next update.

Pause/stop work during factor loading boundaries, between preparation jobs,
at training batches/tree rounds, and between evaluation batches. Pause retains
the Python process and its memory; stop exits after saving. `resume` releases a
paused process; after exit/interruption rerun the original stage with
`--resume`, for example `nifty.cmd train --research-only --resume`.
Preparation restarts an incomplete stock/timeframe, reusing verified complete
jobs. Training resumes its saved round/batch. Evaluation restarts its current
stage without retraining. The combined command uses the same controls for
both preparation and training, including the transition between them. A stop
at the end of preparation prevents training from starting. Restart either
interrupted stage with `nifty.cmd train --research-only --resume`.
Preserve identical preparation scope and model/device options when
resuming. All training workers observe the same run-level pause/stop request;
on failure, the coordinator stops dispatching work and waits for other workers
to checkpoint before returning the error. Training summaries include per-job
elapsed times.

Both CUDA-enabled PyTorch and CUDA-enabled XGBoost are checked before dispatch;
there is no silent CPU fallback. On the 4 GiB RTX 3050 A, start with at most two
GPU workers. They share the same GPU, so two workers are **not guaranteed to be
twice as fast** and can still exhaust VRAM, especially while other applications
use the GPU. Reduce concurrency, without changing model settings, if needed:

```bat
nifty.cmd train --research-only --device cuda --workers 2
nifty.cmd train --research-only --device cuda --workers 1 --resume
```

Changing `--workers` is permitted on resume. An explicit `--device cpu` is
available for a new CPU training run; changing the device of an existing
training run is rejected to preserve checkpoint identity.
Evaluation/prediction still default to CPU; pass `--device cuda` explicitly to
accelerate those stages. Training dependencies are in `requirements-training.txt`.
A smaller explicit preparation scope is supported,
for example `nifty.cmd train --research-only --symbols INFY --timeframes 1d --run-dir data\models\infy-test`;
pass that same run directory to subsequent stages.

`predict` uses the latest eligible **prepared historical snapshot**, not live
quotes or newly downloaded factors. Its output includes the actual as-of time.
For intraday jobs the snapshot stops at the last anchor with a known eligible
next complete candle; it never extrapolates an overnight move from an
intraday-only model.
Do not treat that forecast as current market data. Inputs/code/checkpoints are
validated; changed source snapshots require a fresh preparation/run directory.
Source rights, corporate-action adjustments, exchange calendars and economic
publication times still require independent verification. Contemporaneous
sector indexes may contain the target stock and do not establish forecasting
skill or causality.

CMD is used for the lightweight launcher, not to accelerate Python training.
Microsoft documents [CMD](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/cmd)
as a command interpreter and [PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/overview)
as a broader automation environment; neither document establishes a universal
idle-resource benchmark. A local no-profile, stdin-waiting check measured about
6.8 MiB for CMD versus 63.5 MiB for Windows PowerShell, with zero measured CPU
time over a two-second idle interval for both (terminal host excluded).
This is machine-specific, not a universal performance guarantee.

### Per-stock, next-candle correlations

The following research command processes historical Nifty 50 members one stock
at a time and writes `stocks\<symbol>.csv` under the output directory. Each
file has a row for every factor and each of the nine stock timeframes. The
correlation is Pearson's correlation between the latest available **daily**
factor feature and the stock's next-candle log return; it does not calculate
intraday factor features. The unfiltered correlation, paired-example count,
status and absolute-0.49 indicator are recorded. The target stock's own local
price is marked `target_stock_excluded`; missing or insufficient data and
constant values are reported with a blank correlation rather than invented.

The command uses all local daily candidates, both prepared global fund
collections, and the saved India, Treasury, RBI-reference and oil snapshots.
These snapshots must already exist at the paths below. The end date is exclusive
and cannot exceed the 2025-01-01 snapshot cutoff. A first run requires a new
output directory:

```powershell
Set-Location 'D:\stonks\Stockmog'
$corrArgs = @(
  '--membership', 'data\external\nse_index_membership_0786cd915974.csv',
  '--stocks', 'data\stocks',
  '--intraday', 'data\external\nifty_prepared_intraday',
  '--daily', 'data\external\nifty_prepared_daily_source_v2',
  '--multisession', 'data\external\nifty_prepared_daily_multisession',
  '--global-factors', 'data\external\global_factor_research',
  '--global-factors', 'data\external\nifty_extra_sector_factors',
  '--india-audit', 'data\external\nifty_all_factors_india_2025',
  '--treasury-audit', 'data\external\nifty_all_factors_treasury_2025',
  '--prices-audit', 'data\external\nifty_all_factors_prices_2025',
  '--output', 'data\external\nifty_candle_correlations_2025',
  '--start', '2022-01-03', '--end', '2025-01-01'
)
python -m modeling.nifty_candle_correlations @corrArgs
```

In another PowerShell window, pause or stop using the standard control command:

```powershell
python main.py training-control pause --run-dir 'data\external\nifty_candle_correlations_2025'
python main.py training-control resume --run-dir 'data\external\nifty_candle_correlations_2025'
python main.py training-control stop --run-dir 'data\external\nifty_candle_correlations_2025'
```

Pause and stop are checked after factor loading and at timeframe and
256-factor boundaries. An incomplete stock is restarted when resumed; completed
stock files are checksum-verified and reused. After a stop or interruption,
rerun with `python -m modeling.nifty_candle_correlations @corrArgs --resume`
in the window where `$corrArgs` is defined. The default minimum is 60 paired
candle examples; `--symbols RELIANCE` limits a new run to one historical member.
Input fingerprints use file size and modification time, not full content hashes.

Daily factor values can repeat across many intraday candles, so those paired
examples are not independent. Source publication times, revisions, adjustments
and data rights remain unverified; the India and global datasets have
research-use restrictions. Multiple testing, overlapping proxies and the
retrospective local catalog can distort correlations. The reported values are
exploratory, not proof of prediction quality or a trained model.

`modeling.nifty_price_model` provides research-only building blocks, not a
fitted model. `load_prepared_candles` verifies each
preparation checkpoint and enforces a caller-supplied row limit; process one
stock at a time. `build_examples` makes next-session close-to-close log-return
labels using consecutive RELIANCE-reference sessions, with both dates in the
same stock membership interval. `build_forecast_features` constructs the same
features without the future close; its caller must supply the actual next
exchange session. Both use a decision at midnight after the source daily
session and the latest available candle from each of the nine timeframes.
Each candle contributes close relative to the daily anchor, body, range and
partial status. These are latest-candle snapshots, not every intraday bar.
Callers must specify maximum acceptable candle and factor ages; missing and
stale observations remain NaN. Missing timeframes are returned explicitly.

Factor return inputs must correspond exactly to a selection made before the
earliest decision session. They must be constructed with verified,
point-in-time availability using the audit's documented method; simply passing
arbitrary returns is not verified by this API. Features named `factor_0_return`,
etc. correspond to selected `(source, factor)` pairs sorted alphabetically.
Local factor returns are used only for the matching decision session; an older
local return is not carried forward. Global returns use the audit's backward
availability and freshness rules.
The existing selection has no qualifying factors; stock-only research requires
an explicit `allow_no_factors=True`. An insufficient-observations status is not
silently treated as no qualifying factor. Training must perform separate
past-only selection for each chronological fold, fit all imputation and scaling
on the relevant training partition only, and purge labels that cross boundaries.

`build_price_tree` creates an untrained XGBoost log-return regressor;
`build_price_lstm` creates an untrained LSTM log-return regressor using the
existing hybrid model's two-layer architecture. `predict_hybrid_prices` combines
outputs from fitted models with an explicitly chosen tree weight and converts
the result to a positive price relative to the decision-session close. Its LSTM
input must already be normalized using training-only statistics. No calibration,
trained weights, forecast accuracy, live-data readiness or trading suitability
is implied. Market-data rights, adjustments and timing assumptions remain
unresolved.

### Pooled price training and controls

`modeling.nifty_price_training.prepare_price_dataset` pools explicitly supplied
per-stock `build_examples` frames. It aligns qualified factors by source and
name rather than treating different stocks' factor slots as equivalent. It
requires a selection fixed before each stock's first example, builds LSTM
sequences only across consecutive reference sessions within a membership
interval, and purges targets whose dates or publication times cross the
validation boundary. Holdout rows are reserved and never used for fitting or
calibration. The caller must supply the prepared frames, their verified
point-in-time factor returns, reference calendar, membership, and chronological
split dates. The workflow below prepares and checkpoints those frames.

`train_pooled_price_model` fits the tree and LSTM only when explicitly called,
requires `research_only=True`, and learns a hybrid weight from validation
returns. A no-factor stock must also be explicitly allowed during data assembly.
It writes a run identity, progress state, one atomic checkpoint, trained
artifacts and a final summary. Normalization uses training examples only; model
artifacts carry feature order, selected-factor provenance and split boundaries.
`load_price_predictor` verifies artifact hashes and feature order;
`align_price_features` maps a stock's factor columns to the pooled schema.
The workflow records candle-age policy and input provenance and reconstructs
selected factor returns using their audit methods.

Use the existing control commands with the training output directory:

```powershell
python main.py training-control pause --run-dir "data\models\price-research"
python main.py training-control status --run-dir "data\models\price-research"
python main.py training-control resume --run-dir "data\models\price-research"
python main.py training-control stop --run-dir "data\models\price-research"
```

Pause and stop take effect after a tree round or LSTM batch; the checkpoint is
saved first. After a stop or interruption, call `train_pooled_price_model` again
with the identical reconstructed dataset, settings and dependency versions and
`resume=True`. Changed data or settings are rejected. The prepared arrays are
currently held in memory; rebuilding the in-memory pooled dataset after a stop
restarts that assembly, although its already prepared candle files are preserved.
Passing a `TrainingControl` to `prepare_price_dataset` enables pause or stop
between stocks. `training-state.json` is progress information, not the recovery
checkpoint. Do not run multiple trainers in the same directory.

There is no performance estimate. A holdout report requires a trained model.
None of the research-only data limitations is removed by training.

### Price holdout evaluation

After a research training run completes, call
`modeling.nifty_price_evaluation.evaluate_holdout(dataset, model_dir, output)`,
using the exact reconstructed `PriceDataset` used for training and a new output
directory. It refuses changed data, model artifacts, feature order, or a
nonmatching training summary. No fitting or calibration is performed. The
dataset now retains actual target closes and stock/session identities; older
programmatic datasets must be rebuilt.

`holdout-predictions.csv` contains the stock, decision and target sessions,
actual close, predicted close, last-close baseline and signed errors.
`holdout-summary.json` reports sample counts, MAE, RMSE, MAPE and directional
accuracy overall and per stock for both the model and baseline. The baseline
predicts the next close will equal the last close. Stocks included in training
but without holdout examples are listed separately. Aggregate rupee errors are
sample-weighted across stocks with different price levels; directional accuracy
counts an unchanged prediction as correct only when the actual close is also
unchanged. These are historical source prices, not necessarily independently
verified official or corporate-action-adjusted closes.

Evaluation checks the model and data identity again before publishing the
report. It writes batch checkpoints and supports the same pause and stop
controls; after interruption, call it with the same arguments and `resume=True`.
Repeated model decisions based on holdout results can invalidate it as an
untouched test.

### Research workflow commands and graphs

These commands do **not** run automatically. The `prepare` command does not
fit a model. Copy the full input data and this code before using another
machine; existing preparation and audit manifests contain absolute paths, so
recreate those preparations, audits and the selection on the new machine
instead of assuming Windows checkpoints can resume unchanged in Colab.
Keep paths and dependency versions unchanged when resuming.

Install the base and training dependencies as described in the Colab section
above, and install `requirements-forecast.txt` for graphing. The extra `hf` and
`0` lines currently in `requirements.txt` prevent relying on that manifest
unchanged.

This is a **stock-only research pilot**, not the requested external-factor
model: the current selection has no factors meeting |correlation| >= 0.7.
The selected dates and age limits below are examples, not established optimal
settings. They place all three splits inside the observed RELIANCE source
coverage. Choose a new run directory and adapt dates, inputs and limits to
actual coverage:

```powershell
python -m modeling.nifty_price_workflow prepare `
  --run-dir "data\models\nifty-price-pilot" `
  --membership "data\external\nse_index_membership_0786cd915974.csv" `
  --selection "data\external\nifty_factor_selection_2025\selection.json" `
  --reference "data\stocks\day\RELIANCE\1d.csv.gz" `
  --intraday "data\external\nifty_prepared_intraday" `
  --daily "data\external\nifty_prepared_daily_source_v2" `
  --multisession "data\external\nifty_prepared_daily_multisession" `
  --start 2025-01-01 --validation-start 2025-08-01 `
  --holdout-start 2025-11-01 --end 2026-02-01 --symbols RELIANCE `
  --intraday-age-hours 96 --daily-age-days 7 --multiday-age-days 14 `
  --factor-age-days 5 --sequence-length 20 --max-rows 5000000 `
  --allow-no-factors
```

End dates are exclusive; selection must precede the first example, and the
three chronological splits must each contain complete sequences. This command
loads one bounded stock/timeframe at a time, saves checksum-verified per-stock
example checkpoints, and reconstructs the pooled arrays in memory. An omitted
or unavailable timeframe fails unless `--allow-missing-timeframes` is explicit;
its features are then missing, not invented. `--skip-unusable` explicitly
omits stocks with insufficient selection observations, missing daily candles
or no eligible examples. `preparation-summary.json` lists omissions, missing
timeframes, selected factor counts and split sizes. Review it before training.
For all historical selection symbols, PowerShell can construct the list:

```powershell
$symbols = (Get-Content "data\external\nifty_factor_selection_2025\selection.json" -Raw | ConvertFrom-Json).stocks.symbol
```

Replace `--symbols RELIANCE` with `--symbols $symbols --skip-unusable` in a
new run, adding `--allow-missing-timeframes` only if those gaps are acceptable.
This historical universe is not a claim that every member is in today's index.
The command regenerates and checks the factor selection against its original
audits. Qualified local and global factors use the audited return calculation;
global factor age must equal the audit setting. Prepared candle parts are
checksum-verified when first read; input file sizes and modification times,
checkpoint hashes, source hashes and package versions are checked on resume.
Source files with unchanged size and modification time are not independently
rehashed on every resume.

Pause, continue, stop and inspect **preparation** using its run directory:

```powershell
python main.py training-control pause --run-dir "data\models\nifty-price-pilot"
python main.py training-control status --run-dir "data\models\nifty-price-pilot"
python main.py training-control resume --run-dir "data\models\nifty-price-pilot"
python main.py training-control stop --run-dir "data\models\nifty-price-pilot"
```

The request is observed between factors, timeframes or stocks, and during
pooled assembly between stocks; a single large file operation is not
interruptible. After stop or interruption rerun the identical prepare command
with `--resume`. Completed stock checkpoints are reused; a partially processed
stock and pooled in-memory assembly restart. A stopped workflow or report
command prints a message to stderr and exits with status 2, not success.
If dataset reconstruction stops before a training or evaluation directory has
been created, continue the parent run control and rerun that command without
`--resume`; use `--resume` only when that stage's directory already exists.
Workflow source files are included in run identities, so changes to them
require preparation in a new run directory rather than resuming an old one.

**Only when you explicitly decide to train**, an example command is:

```powershell
python -m modeling.nifty_price_workflow train `
  --run-dir "data\models\nifty-price-pilot" --research-only `
  --tree-rounds 100 --epochs 20 --batch-size 32 --learning-rate 0.001 --device cpu
```

These parameters are examples, not a guarantee of accuracy, cost or runtime.
Use `--device cuda` only with a working CUDA installation; TPU is not supported
by this trainer. Training controls use the `model` subdirectory:
`data\models\nifty-price-pilot\model`. After interruption repeat the identical
training command with `--resume`; tree rounds and LSTM batches are checkpointed.
Dataset reconstruction before training uses the parent run-directory controls.

After completed training, evaluate and generate the graphs:

```powershell
python -m modeling.nifty_price_workflow evaluate `
  --run-dir "data\models\nifty-price-pilot" --batch-size 256 --device cpu
python -m modeling.nifty_price_report `
  --evaluation "data\models\nifty-price-pilot\evaluation" `
  --output "data\models\nifty-price-pilot\report"
```

Evaluation controls use the `evaluation` subdirectory and report controls use
the `report` subdirectory; restart either identical command with `--resume`.
The report contains `stock-graphs.pdf` with one graph per stock, separate PNGs
under `graphs`, `stock-metrics.csv` and `report.json`. Each graph shows actual
and AI forecast closes together for the holdout dates, not a future forecast.
Baseline metrics are in the CSV and JSON. The report verifies the evaluation
CSV hash and per-stock metrics. No report can be produced before evaluation.

## Closing-price trend baseline

The separate `forecast_stock_prices.py` script plots the same five selected
stocks using a simple 20-session geometric trend. By default the cutoff is
January 29, 2026.
It is **not** a prediction from the saved hybrid trading models and does not
train them. Extending a fixed trend over months is highly unreliable.

Install its dependencies independently of `requirements.txt`:

```powershell
python -m pip install -r requirements-forecast.txt
python forecast_stock_prices.py
```

For a new forecast using the latest completed market session as its cutoff:

```powershell
python forecast_stock_prices.py --latest
```

Latest-cutoff runs are saved separately with the prefix `forecast_latest_`.
The 21 historical session closes include verified special sessions and must all
be available. A new run recalculates its cutoff; it does not update the error
in a previous saved forecast, so keep that forecast for a later comparison.

The script queries TimeAPI.io for the current date in India and forecasts
through the next Friday (strictly after today). It uses the NSE holiday list
from marketholidays.in and includes the verified February 1 and November 8,
2026 special sessions. Other years are rejected until their special sessions
are verified and added; API failures stop the run. Confirm calendars with the
exchange, especially around special sessions.

Each run saves a new timestamped folder under `stock_forecasts` with
`stock_graphs.png`, `daily_prices.csv`, and `run.json`. The CSV and charts show
only trading sessions, including special sessions; future days have blank actuals.
Charts use equal spacing between sessions. The script compares
historical actuals from Yahoo Finance and NSE special-session archives where
available, reporting missing historical closes. The console and `run.json`
report the overall mean absolute percentage error: the average of
`100 * abs(forecast - actual) / actual` across all stock-days with actual
closes, using unrounded prices. The comparison count is included; future and
missing closes are excluded, and the error is unavailable if none can be
compared. Closing prices come from Yahoo Finance and NSE special-session
archives. The `--end-date
YYYY-MM-DD` and `--output PATH` options allow a historical end date or a
different output location. Results before the current date are retrospective,
not independent prospective forecasts.

For programmatic use, `predict_hybrid_signals(..., include_all_candidates=True)`
also returns rejected candidates and candidates that cannot be scored; the
latter have null probabilities and a null `accepted` value. The default still
returns only accepted candidates. These scores are not automatically
out-of-sample: historical model predictions must not be used for RL training
unless the model and calibration were fitted strictly before each prediction.
`walk_forward_splits` prepares expanding training/calibration windows and a
separate final policy holdout, purging labels that cross either fitting
boundary. It does not itself fit models, run a simulator, or prevent RL
episodes from crossing the holdout boundary; callers must enforce those
requirements separately.

## Walk-forward outcome scores for RL

The `walk-forward-scores` command fits new outcome models for each
timeframe/strategy job in a batch plan, then scores every available symbol in
that job. It uses expanding chronological folds; each fold fits and calibrates
on earlier events whose forward labels finish before the relevant boundary.
The first 50% of distinct labeled event timestamps is warmup, the next 35%
is divided into three prediction folds, and the final 15% is a separate policy
holdout. These are timestamp-count fractions, not calendar-time fractions.
No existing model artifacts are overwritten.

Run this **only when ready to start model training**:

```powershell
python main.py walk-forward-scores `
  --plan "data\models\batch-plans\all-timeframes-pooled-canonical.json" `
  --output "data\models\rl-walk-forward-scores"
```

By default all labeled events are used for fitting, which can require hundreds
of GiB of temporary disk space for 1m data. `--event-stride N` instead selects
every Nth candidate per symbol in chronological order for fitting. Unlike the
normal batch's evenly spaced cap, this selection does not depend on the final
number of future events. It changes the training sample; `1` keeps all events.
Regardless of stride, all candidates after warmup are scored. A missing
candidate is not a zero score, and an unscorable candidate has null scores.

Each job's `folds.json` records its boundaries, and `scores.json` maps symbols
and original input fingerprints to per-symbol Parquet files. Score timestamps
are explicitly UTC. Each row includes its fold and `policy_holdout` flag.
The score-run manifest records the acceptance rule used to produce its rows;
previously generated scores retain their original acceptance flags.
The input fingerprint is file size and modification time, not a content hash.
Consumers must check that their input data still matches it. Only a run whose
top-level `score-run.json` says `complete` is usable; an error stops the run,
rather than silently skipping a job.

Fold boundaries are determined separately for each timeframe and strategy.
Do not pool jobs into one RL policy or evaluation without imposing a common
time-based holdout that excludes every job's holdout from policy training.
Do not use holdout metrics to tune the policy. Policy episodes must also be
kept from crossing into holdout periods.

This scoring command currently cannot resume an interrupted process: it
requires a new output directory and leaves an incomplete run for inspection.
The full-dataset fitting and scoring path has not yet been run.

## Backtrader Gymnasium environment

After installing `requirements-rl.txt`, import
`BacktraderTradingEnv` from `backtesting.rl_env`. Prepare one symbol and one
fold with `prepare_fold_episode(bars, scores, fold=..., fold_index=...,
policy_holdout=...)`; read `fold` from that job's `folds.json` and the matching
per-symbol Parquet file from `scores.json`. The helper converts the score
timestamps (UTC) to Indian market time and restricts bars and scores to the
selected fold. It rejects a mismatch between the requested and recorded
holdout status.

```python
from backtesting.rl_env import BacktraderTradingEnv, prepare_fold_episode
from backtesting.rl_execution import ExecutionConfig

episode_bars, episode_scores = prepare_fold_episode(
    bars, score_rows, fold=fold_metadata, fold_index=fold_id,
    policy_holdout=False,
)
env = BacktraderTradingEnv(
    episode_bars, episode_scores,
    config=ExecutionConfig(style="intraday", interval="15m"),
)
```

Actions are `0=wait`, `1=hold`, `2=buy/open long`, `3=sell/close long`,
`4=open short`, and `5=cover short`. `env.action_masks()` reports valid actions
for agents that support action masks. Invalid actions otherwise become the
position-appropriate neutral action and incur the configured penalty; they
never reverse or pyramid a position. Shorts are unavailable on swing data and
near the end of an episode/session so there is time to cover at a later bar
open. Every order is still executed by Backtrader.

The observation is a fixed `float32` vector: the most recent `lookback` bars
of close-normalized OHLC, close return and log relative volume, both
directions' candidate/prediction masks and loss/neutral/win/expected-return/
uncertainty scores, then position exposure, equity change, peak drawdown and
time-of-session encoding. Missing scores remain zero with a separate false
prediction mask; a candidate with no score remains distinguishable from no
candidate. At every decision the current bar is complete; the selected action
can only fill at the next bar's open. Reward is the change in marked equity,
normalized by starting capital, with a small penalty for invalid actions.
Episode completion returns final signed P/L and percentage in two-decimal
strings. Reset can start another episode over the same fixed data; callers
must choose new symbol/fold data to vary training episodes.

Use only non-holdout folds to train an agent. Reserve the holdout folds solely
for evaluation, and never let episodes cross fold boundaries. The environment
is a single-symbol, single-fold Gymnasium adapter.

## PPO agent training

The optional RL requirements install `sb3-contrib`'s `MaskablePPO`, which uses
the environment's `action_masks()` so invalid position transitions are not
sampled. `train-rl-agent` creates one policy for each available timeframe and
strategy, sequentially training it on each symbol's **non-holdout** fold
episodes. It never uses the reserved policy holdout. PPO checkpoints and
`progress.json` are written after each symbol/fold; `--resume` validates the
settings and score inputs before continuing. Only one symbol's bars and scores
are loaded at a time. A run lock prevents two trainers from modifying the same
output directory. Pause/stop requests use the `training-control` command and
take effect at a completed symbol/fold checkpoint boundary.

After generating complete walk-forward score files, the user may start the
agent training explicitly:

```powershell
python main.py train-rl-agent `
  --scores "data\models\rl-walk-forward-scores" `
  --output "data\models\rl-policies"
```

Resume an interrupted run with the same options plus `--resume`. The defaults
are 2,048 PPO steps per symbol/fold, 256-step rollouts, 64-sample minibatches,
₹50,000 starting cash, 10% exposure, a 50% peak drawdown trigger, 10 bps fees
per side, and 5 bps slippage per side. These cost and exposure settings are
simulation assumptions. The drawdown threshold triggers next-open liquidation
and cannot guarantee a maximum realized loss.

The default all-symbol plan contains 36 timeframe/strategy policies and about
2,535 symbols per job. At these defaults this can involve roughly 560 million
PPO timesteps, in addition to the much more expensive walk-forward model fitting
and scoring. Start with a copied plan restricted to a small symbol/timeframe
subset and lower timesteps to assess runtime and disk needs. The trainer has
not been run on the full data, and no training command is started automatically.

## Holdout evaluation and P/L reports

After policy training completes, evaluate each trained timeframe/strategy policy
on each symbol's reserved final holdout fold:

```powershell
python main.py evaluate-rl-agent `
  --scores "data\models\rl-walk-forward-scores" `
  --policies "data\models\rl-policies" `
  --output "data\models\rl-holdout-evaluation"
```

The evaluator verifies that score inputs, fold metadata, training progress, and
policy checkpoints match the completed training run. It uses deterministic
masked actions and the same saved execution assumptions as training. Results
are written to `holdout-results.csv` (one row per symbol) and
`holdout-summary.json` (per timeframe/strategy and aggregate totals), with
negative P/L for losses and two decimal places for monetary amounts and
percentages. Completed symbol results are checkpointed; use the same command
with `--resume` after an interrupted evaluation.

Aggregate totals add independent symbol simulations, each using the configured
starting capital. They are not a capital-constrained portfolio backtest and do
not include live brokerage execution.

## Backtrader execution component

The optional Backtrader execution component can be installed with
`python -m pip install -r requirements-rl.txt`. Backtrader is GPL-3.0 licensed.
`backtesting.rl_execution.simulate_episode` accepts one symbol's OHLCV bars
in a pandas DataFrame with a DatetimeIndex, an `ExecutionConfig`, and a policy
callback. It runs only a simulation; it does not place live orders. The callback
receives a completed bar and its timestamp is the **start** of that bar.
Timezone-aware timestamps are converted to Asia/Kolkata; naive timestamps are
assumed to be local exchange time.

The callback can wait or open a long/short when flat, hold or close a long,
and hold or cover a short. Market orders are filled by Backtrader at the next
available bar's open. Invalid actions, rejected orders, and orders left
unfilled at the end raise errors rather than silently succeeding. Shorts are
permitted only intraday and are forced to cover at the final scheduled bar's
open. If missing bars would carry a short order into another session, the
episode fails instead of claiming a valid intraday result. Intraday timestamps
must align with the regular 09:15–15:30 Indian market session; special sessions
are not supported.

Defaults are ₹50,000 starting cash, a 10% target exposure calculated at the
decision close, 10 basis points per side for fees, and 5 basis points per side
for slippage. These are simulation assumptions, not verified broker charges.
An opening gap can change actual exposure or cause an order rejection. The
50% peak-equity drawdown trigger is checked at completed bar closes; liquidation
is submitted for the next open, so a gap can cause a larger loss. It is not a
guaranteed maximum loss.

`EpisodeResult.formatted_summary()` reports equity and signed profit/loss and
percentage with two decimals; internal calculations retain precision. An open
long at the end is marked to market, not assumed liquidated. The component is
not yet a Gym/RL environment and does not generate walk-forward model scores or
train an agent. Its policy callback currently receives price, volume, position
and equity observations, not the hybrid outcome scores.

## Start training

This command is provided for the user to run; it is not run as part of setup:

```powershell
python main.py train-models `
  --input "data\stocks\minute\RELIANCE\15m.csv.gz" `
  --style intraday `
  --strategy box_breakout
```

For swing training, use an appropriate daily input and `--style swing`. Add more
symbols with additional `--input` options. Choose `--models xgboost` or
`--models lstm` for a single stage; `--models all` trains both models and then
calibrates their blend. Use `--run-dir` to choose another output location.
Start with a small symbol subset to stay within the machine's 16GB RAM limit;
the pipeline materializes strategy-event sequences in memory rather than
streaming a full-market training set.

Intraday intervals are trained as separate runs so that a bar horizon and
sequence length have a consistent time meaning. After the minute split has
generated the new files, examples include:

```powershell
python main.py train-models `
  --input "data\stocks\minute\RELIANCE\1m.csv.gz" `
  --style intraday `
  --strategy box_breakout

python main.py train-models `
  --input "data\stocks\minute\RELIANCE\20m.csv.gz" `
  --style intraday `
  --strategy box_breakout
```

These are commands for later manual use; preparing 1m/20m data does not start
training.

Each run stores its model artifacts, data/parameter manifest, per-model metrics,
validation-based hybrid calibration, and test metrics. Existing runs cannot be
overwritten accidentally. `--resume` requires the same inputs and training
parameters as the saved manifest.

## Plan pooled training across timeframes

The planner includes every supported interval with data: `1m`, `5m`, `15m`,
`20m`, `30m`, `1h`, `1d`, `2d`, and `trading_week`. Symbols are pooled together
as input observations; there is no per-symbol model/job dimension. The four
implemented strategy families (`box_breakout`, `momentum`, `trend_following`,
and `mean_reversion`) remain separate candidates. Catalog IDs are excluded
because they reuse approximate templates rather than independently validated
rules.

This is not a claim that the strategies are profitable. Compare cost-aware
chronological holdouts and portfolio backtests before selecting any candidate.

Generate the all-symbol/all-timeframe plan:

```powershell
python main.py plan-training-batch
```

By default this writes
`data\models\batch-plans\all-timeframes-pooled.json` and reserves output under
`data\models\batch-runs\all-timeframes-pooled`. For a fully populated dataset,
the plan contains 36 pooled jobs (4 strategies × 9 intervals), 72 model fits,
and 36 hybrid calibrations. The plan records each input CSV path but does not
make symbol-specific jobs. An optional `--symbols` filter is available for
small test runs:

```powershell
python main.py plan-training-batch --symbols RELIANCE HDFCBANK INFY TCS
```

Use `--intervals 20m 30m 1h trading_week` to plan only those intervals,
with a new `--output` and `--run-root` to preserve existing models.

Planning does not start training.

Pooled feature/sequence arrays are written to disk in bounded chunks and reused
for all model stages for that job. Sequence values use float32 to preserve large
indicator magnitudes. The feature builder selects only timestamp, symbol, and
OHLCV from inputs, then recomputes indicators; precomputed indicator columns in
stock CSVs are ignored so every symbol gets the same feature schema. LSTM
training reads sequence batches from the
disk-backed cache instead of normalizing the entire pooled tensor in RAM. By
default `train-batch` evenly samples at most 400 strategy events per input file
(`--max-events-per-file`, 0 = keep all) because uncapped 1m caches would need
hundreds of GiB. Cache finalization streams raw arrays into `.npy` files in
8 MiB chunks and prints copy progress and an ETA, avoiding a large memory-mapped
copy that can cause Windows to page heavily. The executor processes one pooled
job at a time and removes its dataset cache after the hybrid stage, limiting
peak cache storage.

For canonical `20m`, `30m`, and `1h` input filenames, session-local box and
relative-volume lookbacks are respectively 8/10, 5/6, and 3/3 prior bars.
These shorter windows fit within a session; 1m, 5m, and 15m retain their
existing defaults. Training and `hybrid-signals` must use the same canonical
timeframe filenames so they calculate the same features. Explicit feature
window arguments override the defaults.

Weekly inputs may contain valid signals before the 200-week EMA is available.
Only that feature may be missing on a weekly event. XGBoost handles its missing
values; the LSTM imputes missing values with the training-only mean. A feature
entirely absent from the training split is disabled for LSTM evaluation and
inference, including when it becomes available later. Other missing current
features still exclude the event. Chronological label purging is unchanged.
Models using these rules require fresh training and out-of-sample evaluation.

The complete dataset is still large (about 24,000 input files and 244 GiB
compressed; 1-minute inputs alone are about 166 GiB). Each individual CSV is
feature-engineered in memory, so a single unusual file may still exceed RAM.
The cache needs free disk space roughly equal to its stored arrays plus a
temporary build copy; the builder stops with an explicit error if it approaches
within 2 GiB of filling the drive. Check disk space and run a representative
pilot before the full batch. A larger GPU does not remove CPU decompression,
feature engineering, disk throughput, or host-memory limits.

Run the all-timeframe plan. Before each LSTM stage the trainer prints the
training-set size, your CPU/RAM/GPU, and a time estimate measured by timing real
batches on this machine. It then reports percent complete, time left in the
current epoch, and maximum time remaining (early stopping may finish sooner)
about once a minute; each epoch line shows actual epoch time.

```powershell
python main.py train-batch `
  --plan "data\models\batch-plans\all-timeframes-pooled.json"
```

To run a pilot first, create a filtered plan with the same timeframe coverage:

```powershell
python main.py plan-training-batch `
  --symbols RELIANCE HDFCBANK INFY TCS `
  --output "data\models\batch-plans\pilot-pooled.json" `
  --run-root "data\models\batch-runs\pilot-pooled"

python main.py train-batch `
  --plan "data\models\batch-plans\pilot-pooled.json"
```

Each pooled model has one `pooled` artifact directory per interval and
strategy. Checkpoints, pause/resume state, dataset caches, and input-file
fingerprints are preserved. Pause, resume, stop, and inspect a run using the
run root recorded in the plan:

```powershell
python main.py training-control pause --run-dir "data\models\batch-runs\all-timeframes-pooled"
python main.py training-control status --run-dir "data\models\batch-runs\all-timeframes-pooled"
python main.py training-control resume --run-dir "data\models\batch-runs\all-timeframes-pooled"
python main.py training-control stop --run-dir "data\models\batch-runs\all-timeframes-pooled"
```

The cursor and run manifest are stored under the run root. The executor skips
pooled jobs with no valid labeled events and records the reason; unexpected
training errors stop the batch for investigation. Jobs run one at a time in
job-major order; each job performs XGBoost, LSTM, and hybrid calibration before
moving to the next interval/strategy. `--max-jobs-per-stage 1` can be used to
checkpoint after one complete pooled job and resume later.
If any jobs were skipped, the final status is `completed_with_failures`, not
`completed`. Resuming a finished run does not retry failed jobs; create a new
plan and run root after addressing their recorded errors.

## Pause, resume, and stop

While training is running, open another terminal and use the exact run directory
printed at startup (the default below is for RELIANCE intraday box breakout):

```powershell
python main.py training-control pause `
  --run-dir "data\models\intraday\box_breakout"
```

The trainer saves a checkpoint at the next safe round/batch boundary, then waits.
Continue the same process with:

```powershell
python main.py training-control resume `
  --run-dir "data\models\intraday\box_breakout"
```

Check whether it is running, waiting, or stopped with:

```powershell
python main.py training-control status `
  --run-dir "data\models\intraday\box_breakout"
```

To checkpoint and exit instead, request a stop, then rerun the original
`train-models` command with `--resume`:

```powershell
python main.py training-control stop `
  --run-dir "data\models\intraday\box_breakout"
```

XGBoost checkpoints are written each boosting round; LSTM checkpoints are
written at least every ten batches and immediately when paused or stopped.
Keyboard interruption may lose work since the most recent checkpoint; use the
control command for a graceful pause/stop. A pause request waits for the current
safe boundary, so it is not an instantaneous interruption.

## Hybrid predictions and backtests

When both models and the hybrid calibration file exist, inspect filtered signals:

```powershell
python main.py hybrid-signals `
  --input "data\stocks\minute\RELIANCE\15m.csv.gz" `
  --style intraday `
  --strategy box_breakout `
  --model-dir "data\models\intraday\box_breakout"
```

The command reports only accepted candidates, with class probabilities,
expected return and uncertainty in R, combined uncertainty, and a simple
trend/range/volatility regime tag. The default acceptance threshold is a
calibrated profitable-outcome probability of 0.22, without expected-return or
uncertainty gating; change the threshold with `--minimum-win-probability`.
This is a research filter, not an order instruction. In one small 20MICRONS
15-minute box-breakout test the experimental rule found more profitable
outcomes but reduced accuracy and lost money in the simulation. It has not
been validated across other symbols or strategies. To restore the earlier
filter, pass `--legacy-hybrid-filter --minimum-win-probability 0.45` to
`hybrid-signals` or `backtest`.

To backtest filtered signals, pass the model directory. For one strategy, pass
its run directory; for `--strategy all`, pass the parent directory containing
one run directory per strategy:

```powershell
python main.py backtest `
  --input "data\stocks\minute\RELIANCE\15m.csv.gz" `
  --style intraday `
  --strategy box_breakout `
  --hybrid-model-dir "data\models\intraday\box_breakout"
```

Validation data selects the hybrid's XGBoost/LSTM blend and temperature. Test
data remains held out for reported metrics. Metrics assess event classification;
they do not establish profitability. Use the existing costed backtester and
walk-forward/paper evaluation before drawing performance conclusions.
