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

The project roadmap targets Python 3.11. This workspace currently uses Python
3.14.7, and XGBoost/PyTorch are not installed; verify PyTorch wheel support for
the interpreter you select (or use the planned Python 3.11 environment) before
installing and training.

The model suite trains XGBoost, Random Forest, Extra Trees, and regularized
Logistic Regression on CPU, followed by a small LSTM. XGBoost and the tree
ensembles suit structured indicator/candlestick features; logistic regression
is a simpler baseline; the LSTM evaluates ordered sequences. The LSTM uses CUDA
if PyTorch detects an available GPU, otherwise it runs on CPU. Training is
sequential; `--models all` trains each estimator in turn and calibrates a
validation-weighted ensemble. LSTM progress includes process/system RAM and
CUDA memory usage when a GPU is active.

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
