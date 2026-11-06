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

## Plan an all-symbol batch

The full matrix across every available interval and all core/catalog strategies
is extremely large. Create a compact inventory and sequential-stage plan with:

```powershell
python main.py plan-training-batch
```

By default this writes
`data\models\batch-plans\all-symbols-all-intervals.json`, covering symbols and
CSV intervals present under `data\stocks\minute` and `data\stocks\day`.
The planned order is stage-major: XGBoost for every job, then LSTM for every
job, then XGBoost/LSTM hybrid calibration for every job. The plan reports exact
job and model-fit counts based on files found.

This command is a **planner only**. It starts no training and does not create
millions of run directories or checkpoints. The full matrix can require
millions of individual model fits and exceed the available 16 GB of RAM and
practical runtime by a large margin. A training-batch executor with bounded
memory, durable per-job checkpoints, and pause/resume is available as
`train-batch`:

```powershell
python main.py train-batch `
  --plan "data\models\batch-plans\all-symbols-all-intervals.json" `
  --max-jobs-per-stage 1
```

The bounded example handles at most one symbol/interval/strategy job in each
stage for that invocation. Rerun the same command with `--resume` to continue
from the saved stage/job and model checkpoint. For continuous full-matrix
execution, omit `--max-jobs-per-stage` and explicitly add
`--allow-large-batch`:

```powershell
python main.py train-batch `
  --plan "data\models\batch-plans\all-symbols-all-intervals.json" `
  --allow-large-batch
```

The executor completes the XGBoost stage across the matrix, then the LSTM stage,
then hybrid calibration. Hybrid blend/temperature search also uses CUDA when
available, and hybrid signal scoring uses GPU-backed XGBoost and LSTM inference.
It automatically chooses a worker count from available
CPU and memory, reserving 2 GiB for the system and budgeting 4 GiB per concurrent
job. CUDA-backed XGBoost runs one job at a time to limit GPU memory contention;
LSTM fitting is serialized because its checkpointed random-number state is
process-global. Pause, resume, stop, and inspect it in a second terminal using
the batch run root from the plan:

```powershell
python main.py training-control pause --run-dir "data\models\batch-runs"
python main.py training-control status --run-dir "data\models\batch-runs"
python main.py training-control resume --run-dir "data\models\batch-runs"
python main.py training-control stop --run-dir "data\models\batch-runs"
```

The cursor and run manifest are stored under the run root; model checkpoints
and hybrid outputs are created per symbol/interval/strategy only as they are
processed. A stopped batch is continued with the original `train-batch`
arguments plus `--resume`. `--max-jobs-per-stage` limits work during one
invocation; remove it on a later `--resume` invocation to continue the stage
without that limit. The executor releases each job's dataset after its stage.
It skips jobs with no valid labeled events and
records their reason; unexpected training errors stop the batch so they can be
investigated. Do not start the full 3.9-million-job matrix on this laptop
without first estimating runtime, output storage, and available RAM.

Each catalog ID also shares one of a small number of documented templates, so
many catalog jobs may repeat equivalent signals.

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
