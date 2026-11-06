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
stability. The hybrid blend weights are selected by validation log loss; its
filter also requires a positive LSTM expected return and limits combined
entropy/return uncertainty.

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

Planning does not start training.

**Do not run the full pooled plan yet.** The current dataset builder retains
feature-event sequences in memory rather than streaming. The local dataset has
about 24,000 input files and 244 GiB of compressed CSV data; the 1-minute files
alone are about 166 GiB compressed. Pooling one interval across every symbol can
exceed the laptop's 16 GiB RAM. A larger-GPU machine does not automatically solve
this host-memory/data-preparation limit. The planner records this warning in its
JSON so it is not mistaken for a safe-to-run queue. First implement bounded,
on-disk dataset preparation/training, or use `--symbols` to make a smaller
pilot.

For a small filtered pilot only, after creating the plan:

```powershell
python main.py train-batch `
  --plan "data\models\batch-plans\all-timeframes-pooled.json" `
  --max-jobs-per-stage 1
```

Each pooled model has one `pooled` artifact directory per interval and
strategy. Checkpoints, pause/resume state, and input-file fingerprints are
preserved. Pause, resume, stop, and inspect a run using the run root recorded in
the plan:

```powershell
python main.py training-control pause --run-dir "data\models\batch-runs\all-timeframes-pooled"
python main.py training-control status --run-dir "data\models\batch-runs\all-timeframes-pooled"
python main.py training-control resume --run-dir "data\models\batch-runs\all-timeframes-pooled"
python main.py training-control stop --run-dir "data\models\batch-runs\all-timeframes-pooled"
```

The cursor and run manifest are stored under the run root. The executor skips
pooled jobs with no valid labeled events and records the reason; unexpected
training errors stop the batch for investigation. XGBoost jobs are stage-major,
followed by LSTM and then hybrid calibration. CUDA XGBoost jobs run one at a
time; LSTM fitting is serialized.

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
calibrated profitable-outcome probability of 0.45, positive expected return,
and combined uncertainty at most 0.95; tune the probability threshold with
`--minimum-win-probability`. This is a research filter, not an order instruction.

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
