# Phase 2: PPO training

## Shared Rs 50,000 portfolio

For one portfolio across all four stocks, use
[the shared-portfolio notebook](phase2_ppo_portfolio_colab.ipynb) and upload
`ppo-portfolio-top4.tar` and `ppo-portfolio-top4.sha256.json` to
`MyDrive/Stockmog`. The independent-account trainer remains available in the
repository, but its checkpoints are not interchangeable with this workflow.

The shared workflow trains one PPO policy on common 1-minute decisions, using
all nine timeframes' features and frozen price predictions. It has one Rs
50,000 cash balance and can hold multiple stocks. For each decision, orders
execute at the first later common candle open allowed by the decision time.
Sales execute first; available cash is divided equally between the stocks
selected for purchase. The whole-share constraint can leave unused cash.
There is no leverage or pyramiding, and the original 10% per-position cap does
not apply. Missing common minutes are skipped; this is a simplified simulator,
not exchange-grade execution.

The action mask uses the latest observed closes, estimated fees and slippage,
and estimated proceeds from simultaneous sales to rule out purchases whose
equal cash budgets cannot afford one share. Actual orders still fill at a
later open; a price gap can therefore cause a rejection, which incurs a
one-step reward penalty. This affordability rule does not look at future opens.
Changing it requires a fresh portfolio training output, not old checkpoints.
The current upload archive predates this fix. The notebook's affordability-fix
cell patches that exact archive version in memory after source verification;
run it on every new runtime before benchmarking, training or testing. It does
not modify the archive. Use the new affordability-specific output directory.

News and research stay in their separate bounded policy branches. The policy
is trained, validated and tested on chronological shared-portfolio sessions,
with train-only normalization. The final report compares the policy with cash
and an equal-budget buy-and-hold portfolio. Four simulated environments used
for PPO training are independent *copies* of this one-portfolio simulation;
they are not combined into a larger account.

The shared-portfolio benchmark and full run use **one** policy, with 262,144
steps by default. Its steps are more computationally expensive than the old
single-stock steps and the old benchmark is not a valid speed estimate. The
benchmark is a rough projection and includes predictor preparation and
validation overhead. Existing stock-selection bias, partial data and limited
post-freeze history still apply.

To refresh the portfolio archive after adding research, from the repository
root run:

```powershell
python -m data_pipeline.rl_colab_data --output data\colab\ppo-dataset-top4 --research data\colab\research --attach-research --archive google_colab\ppo-portfolio-top4.tar
```

Repackaging includes the updated source, so remove the legacy affordability-fix
cell after verifying the new source is present in the archive. Upload both
regenerated files and use a fresh Colab `WORK` and output directory for a new
training run.

## Run in Google Colab

This top-level folder contains everything you need to upload:

- `phase2_ppo_portfolio_colab.ipynb`: upload this notebook to Colab.
- `ppo-portfolio-top4.tar` and `ppo-portfolio-top4.sha256.json`: upload both
  to Google Drive under `MyDrive/Stockmog`.

The archives contain the Python source, dependency requirements, frozen
trend-analyser models, history, news and research. You do not need to upload
individual Python modules or install the repository separately.
Archives and their checksums are local, git-ignored files, not GitHub downloads.
The reusable source modules, tests and unpacked export data remain in their
original repository locations.

1. Upload the matching archive/checksum pair to Drive and the notebook to Colab.
2. Select a GPU under **Runtime > Change runtime type** (A100 if available).
3. Run the setup cells and review the printed news/research coverage. Then set
   `ALLOW_PARTIAL_NEWS = True` and `ALLOW_PARTIAL_RESEARCH = True` in the
   settings cell and rerun it.
4. Run the benchmark. When satisfied, set `RUN_FULL_TRAINING = True` and run
   the training cell.
5. Enable `RUN_FINAL_TEST` only after fixing the training design. Outputs and
   portfolio checkpoints are saved to Drive.

## Independent-account workflow

The legacy independent stock/timeframe trainer remains in the source tree.
The following background and local export instructions describe that trainer;
its standalone Colab notebook and upload archives have been removed.

## Local preparation

From the repository root, using the existing completed Nifty timeframe run:

```powershell
python -m data_pipeline.rl_news_stream --workers 4
python -m data_pipeline.rl_colab_data
```

The news command resumes checksum-verified successful snapshots and retries
failed ones. It collects **one 15-minute GKG release per available trading day
at 09:00 UTC**, plus a capped recent DOC API search. These are two free
endpoints of GDELT, not independent historical news providers. It does not
download article bodies. It is not a complete intraday news archive:
95 other daily releases, weekends, ambiguous aliases and unavailable files
create coverage gaps. Read `data/colab/news/coverage.json`; failures are
printed and recorded. A DOC failure does not invalidate available GKG records,
but the report remains explicitly partial.

The export produces `data/colab/ppo-dataset.tar` and
`data/colab/ppo-dataset.sha256.json`. Data, source, frozen predictor weights, selected
factor histories, outcomes and coverage metadata are included. No credentials,
virtual environments, training caches or unrelated model runs are included.
The archive is private research data, not a redistribution license.
The unpacked directory is retained locally. A new export requires a new output
path; incomplete exports are not silently accepted.
The exporter uses the unpacked directories under `data/colab` to refresh
research without rebuilding stock histories.

### Top-profit archive (trend-analyser picks only)

The trend analyser (each job's frozen XGBoost + LSTM predictor) is already an
input stream: PPO receives its predicted next-candle log return as
`model_log_return`. To train only on the stocks whose trend-analyser picks
were profitable in the shared Rs 50,000 test, export just those symbols:

```powershell
python -m data_pipeline.rl_colab_data --output data/colab/ppo-dataset-top4 --research data/colab/research --symbols "M&M" MAXHEALTH ONGC BSE
```

This produces `ppo-dataset-top4.tar` (36 jobs, about 0.31 GiB) and its
`.sha256.json` sidecar locally. Use the shared-portfolio refresh command above
to package the data for the portfolio notebook. Choosing stocks from previously
examined profit results is selection bias; treat test results as optimistic.

## Inputs and causal boundaries

Every available prepared stock/timeframe job is exported, including its full
original history. Missing source jobs are recorded, not synthesized.
PPO inputs comprise:

- That timeframe's original technical and selected external-factor features.
- As-of features from the same stock's other timeframes: 5/20-bar returns,
  RSI and their selected factors, with a 14-day maximum age and missing masks.
- The frozen XGBoost/LSTM model's predicted log return; unavailable sequences
  have a missing mask rather than a fabricated forecast.
- Rolling 24-hour and seven-day news counts, mean GKG tone and missing-tone
  masks, plus an explicit incomplete-news-history flag.
- Position exposure, cash fraction, equity, drawdown and episode time.

News is released no earlier than its GDELT timestamp plus a conservative
15-minute lag. Later duplicate metadata must not enrich earlier observations.
Price reactions are real historical outcomes; news never modifies stock OHLC.
Forward returns and their availability timestamps are included as **labels
only**, never observation columns. Missing features are train-mean imputed
with explicit observed masks; scaling is fitted on RL training rows only.

The saved price model used supervised training and validation. Therefore PPO
only uses decisions on/after its original `holdout_start`, when it was frozen.
Earlier rows provide history, not RL episodes. This reassigns the old
supervised holdout into chronological RL train/validation/test sets
(60/20/20 by decision session). Episodes cannot fill outside their split.
No frozen predictor is back-applied to its own training dates as an
out-of-sample input. Cross-timeframe selected factors are withheld until their
selection cutoff. The existing economic release-lag assumptions are retained;
this is not independently verified point-in-time macro vintage data.

Consequently, daily/weekly PPO training histories can be very short despite
including years of source history. Jobs with fewer than 15 post-freeze decision sessions are explicitly
marked ineligible in the archive and reported as unavailable by the trainer.
The full available history is retained; unavailable jobs are not fabricated
or silently skipped. More PPO steps do not create more independent historical
experience. Extending this safely requires refitting the supervised models in
walk-forward folds; that expensive operation is not disguised as a data export.
The research membership union and already-inspected report periods also limit
claims of unbiased evaluation.

## Research and report stream (M&M, MAXHEALTH, ONGC, BSE)

The four stocks that were profitable in the shared Rs 50,000 forecast analysis
receive an additional research stream. Collect it, then attach it to the
existing export (all job histories are checksum-verified before repackaging):

```powershell
python -m data_pipeline.rl_research_stream --reports data/research/reports.csv
python -m data_pipeline.rl_colab_data --output data/colab/ppo-dataset --research data/colab/research --attach-research
```

`--reports` is optional. Sources, each with its own `source` flag:

- **NSE corporate filings** (2022 onward): results, analyst/investor meets,
  presentations and press releases, credit ratings (upgrade +1, reaffirm 0,
  downgrade -1 from the filing text), corporate and legal actions.
- **NSE quarterly results XBRL** (classic and integrated filings, consolidated
  preferred): profit and revenue YoY growth versus the same quarter a year
  earlier, squashed with `tanh`. This is reported growth, not a consensus surprise.
- **GDELT research-style media**: sampled GKG items whose URL suggests ratings,
  targets or brokerage calls; currently very sparse.
- **Your broker reports**: a CSV with `symbol,published_at,rating,target_price`
  (optional `source,title`). Ratings: buy/strong buy, outperform/overweight,
  add/accumulate, hold/neutral, reduce, underperform/underweight, sell.
  `published_at` must be timezone-aware (`2025-03-11T08:00:00+05:30`) or a
  date (`2025-03-10`, then used from 09:00 IST the next day). Only report
  metadata you are licensed to use is read; report text is not stored.

Every item is available from its exchange dissemination/publication time plus
15 minutes. Twelve features are built as of each decision: filing counts
(7/30 days), analyst-meet count, latest credit action (365 days), latest
profit/revenue YoY (200 days), days since results, research-media count/score
(30 days), mean analyst rating (90 days), latest target-price upside versus the
current close (180 days) and a coverage flag. Other stocks receive missing
values with zero observed masks.

Research priority is 5/10 (news 1-2/10). A separate zero-initialized research
branch adds at most `0.15 * tanh(...)` to each action logit (`research_logit_limit`,
0 disables, max 0.3). Like news it is excluded from the primary actor and the
critic. Combined, news and research can change a logit gap by at most 0.40.
Coverage is partial: paywalled broker research is not collected, filing
subjects are keyword-classified and NSE pages can be blocked. Training requires
`--allow-partial-research` (`ALLOW_PARTIAL_RESEARCH = True` in Colab) after
reviewing `research-coverage.json`.

**Streaming:** `python -m data_pipeline.rl_research_stream --watch --interval 900`
polls NSE filings and results, recent GDELT DOC research headlines (ratings and
target prices parsed from titles) and the reports CSV, then appends deduplicated
events to `data/colab/research/events.parquet`. `ResearchStream.latest(symbol, close)`
returns the current feature vector. The stream does not retrain policies or place trades.

## Actions, reward and evaluation

MaskablePPO uses `0=stay`, `1=buy`, `2=sell`. Stay holds a long or waits in cash.
Buy opens a whole-share long; sell closes it. No shorts, pyramiding or leverage.
Defaults are Rs 50,000 per independent policy account, 10% target exposure,
0.10% fees and 0.05% slippage on each side. Insufficient capital at the fill
price is recorded as a rejection. Overnight longs are allowed.

Actions fill at the first **later** candle open whose start is not before the
decision's recorded availability, never at the observed close. In particular,
daily/2d/weekly labels must not be interpreted as close availability. The
fast NumPy environment deliberately replaces the threaded Backtrader adapter
for PPO throughput; its next-open timing, whole-share accounting, costs and
position transitions have focused tests. It does not claim exchange-grade
execution, special-session validation or brokerage parity.

The per-step training reward is:

```text
log(equity_next / equity_now)
- 0.10 * max(0, drawdown_next - drawdown_now)
- 0.01 * min(0, log(equity_next / equity_now)) ** 2
- 0.001 if an invalid action is requested
```

Fees and slippage are already charged to equity. The risk penalties discourage
new drawdowns and downside volatility without paying merely for trading.
Coefficients are explicit research assumptions, not empirically optimal
fitness parameters. The 50% drawdown threshold stops an episode; it is not
a guaranteed stop-loss or an assumed executable liquidation.

Validation and final-test reports compare deterministic masked PPO actions to
cash and buy-and-hold **at the same configured exposure and execution costs**.
They include net P/L, percentage return, maximum drawdown, downside RMS,
transactions, rejections and remaining shares. Final holdings are marked to
market, not falsely reported sold. Do not select policies using the final test.
Independent stock/timeframe account totals are not a shared-capital portfolio.

## Network, efficiency and resume

News is deliberately **low priority**. A separate, initially zero-weight news
branch adds `0.05 * tanh(news_score)` to each buy/sell/stay logit. All news
values and their missing/observed masks are excluded from the primary actor
and critic inputs. The critic does not use news. This bound is applied after
train-only feature normalization, so normalization or larger learned news
weights cannot undo it.

News can influence near-tied actions, but cannot reverse a primary preference
with a logit gap greater than 0.10 among the valid actions. This is a bound on
action scores, **not a claim that news explains exactly 5% of decisions or
profits**. Set `NEWS_LOGIT_LIMIT = 0` in the notebook to disable the branch's
direct effect. Rewards, fills and costs are unchanged and never depend on news
sentiment. Changed actions can still affect later positions and outcomes.

Old unconstrained-policy checkpoints must not be resumed as this architecture;
the changed archive/config identity requires fresh training.

Actor and critic each have hidden layers `[512, 512, 256]`. Default training is
262,144 steps per job, four in-process vector environments, 1,024-step
rollouts, 1,024-sample minibatches, five epochs, two PyTorch CPU threads.
These are starting settings, not convergence or profitability guarantees.
The default 543-job run requests 142,344,192 transitions.

The current export retains all 543 histories, but HDFC/trading_week has
insufficient post-freeze sessions, leaving **542 RL-eligible policies**.
M&M has no matched news in this sampled collection; its news inputs are
explicitly missing, not a claim that no relevant news occurred.

The environment uses precomputed NumPy inputs, avoiding article NLP,
per-tick API calls, subprocess copies and Backtrader thread synchronization
inside rollouts. Only one stock/timeframe policy trains at a time.
Predictor inference is batched and CPU-based, and only post-freeze forecasts
are generated. Parquet is already compressed; the outer TAR is intentionally
not recompressed. Extract to Colab local disk, train there, and save
checkpoints/results to Drive instead of streaming market rows from Drive.

Policies are saved every approximately 65,536 transitions and at job completion.
Unchanged dataset/config identity is required to resume. Save/checkpoint pairs
are checksum checked; an interrupted inconsistent write raises instead of
silently using an older state. Rollout rounding may exceed the requested
step count. Resume is checkpoint-based, not bit-for-bit RNG restoration.
Do not run two notebooks against the same output directory concurrently.

## Colab plan, runtime and time estimate

Start with **Pay As You Go or Pro**, not Pro+ solely for a premium GPU.
Benchmark CPU against a **T4 or L4** if available. Large MLP PPO can benefit
from a GPU, but lightweight sequential rollouts can still be CPU-bound.
An A100 is not automatically the best CU-per-trained-policy choice.
Keep CPU preprocessing separate from paid GPU experiments where practical.

The shared-portfolio notebook includes a separate benchmark; its results do
not directly predict the speed of independent-account jobs. As a historical
*planning scenario only*, 142 million
transitions at 250–1,500 end-to-end transitions/second is about **26–158 hours**,
plus archive transfer, preprocessing, predictor inference and interruptions.
This range is not a measured Colab estimate or a promise of convergence.
One million steps per policy is about 543 million transitions: approximately
101–603 hours at those hypothetical rates. Daily/weekly data scarcity may
matter much more than additional compute.

A local smoke test of the full-size network completed 4,096 transitions on
ADANIENT daily data, including predictor preparation and validation, in about
14 seconds on this computer. Naively scaling that end-to-end pilot to the full
default workload is approximately 135 hours; this also scales fixed preparation
costs and therefore is not a reliable training-only projection.
This is not a Colab hardware benchmark, a
convergence test or evidence of profit.

Colab hardware, availability, usage limits and CU burn rates vary.
The public pricing page may require sign-in; confirm current local prices
in your account rather than treating an old dollar or rupee figure as a quote.
Use `estimated_hours * displayed_CU_per_hour`, then divide by the CU quantity
in your chosen package to estimate purchases. Neither a subscription nor
these settings guarantees one uninterrupted full run.

Official resources:
[Colab plans](https://colab.research.google.com/signup),
[Colab resource limits](https://research.google.com/colaboratory/faq.html),
[GDELT DOC API scope](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/),
[GKG archive index](https://data.gdeltproject.org/gdeltv2/masterfilelist.txt).
