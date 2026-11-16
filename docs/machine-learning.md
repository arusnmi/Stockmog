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
