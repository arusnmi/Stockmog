"""Screen daily factor observations against individual next-candle stock returns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import INTRADAY_TIMEFRAMES, _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import PreparationStopped, _fingerprint, _write_json, consecutive_returns, load_closes
from modeling.global_factor_research import global_returns
from modeling.nifty_candle_examples import NoCandleExamples, build_candle_examples
from modeling.nifty_preparation import TIMEFRAMES
from modeling.nifty_price_model import load_prepared_candles
from modeling.nifty_price_training import _price_lock
from modeling.nifty_public_factor_research import FILES, REPOSITORY, REVISION, _blob_hash, _read_returns
from modeling.nifty_treasury_factor_research import TABLES, rate_factors
from modeling.nifty_universe import COVERAGE_END, COVERAGE_START, local_symbol, parse_nifty_membership


@dataclass(frozen=True)
class Factor:
    source: str
    name: str
    times: np.ndarray
    values: np.ndarray
    max_age_ns: int
    self_symbol: str | None = None


def _ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns").asi8


def _factor(source: str, name: str, timestamps: object, values: object, days: int,
            self_symbol: str | None = None) -> Factor:
    times = _ns(timestamps)
    numbers = np.asarray(values, dtype=np.float64)
    if (len(times) != len(numbers) or not np.isfinite(numbers).all()
            or (times == np.iinfo(np.int64).min).any() or (np.diff(times) <= 0).any()):
        raise ValueError(f"Invalid timestamps or values for {source}:{name}")
    return Factor(source, name, times, numbers, days * 86_400_000_000_000, self_symbol)


def _local_factor(name: str, returns: pd.Series, source: str, *, days: int = 5) -> Factor:
    values = returns.dropna()
    dates = pd.DatetimeIndex(pd.to_datetime(values.index)) + timedelta(days=1)
    available = dates.tz_localize("Asia/Kolkata").tz_convert("UTC")
    return _factor(source, name, available, values.to_numpy(), days)


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _daily_paths(daily: Path) -> list[Path]:
    return sorted(path / "1d.csv.gz" for path in daily.iterdir()
                  if path.is_dir() and (path / "1d.csv.gz").is_file())


def _read_sources(local_paths: list[Path], global_roots: list[Path], india: Path,
                  treasury: Path, prices: Path, sessions: list[date]) -> list[Factor]:
    factors = []
    for path in local_paths:
        name = path.parent.name
        series = consecutive_returns(load_closes(path), sessions)
        factor = _local_factor(name, series, "local_daily")
        factors.append(Factor(factor.source, factor.name, factor.times, factor.values,
                              factor.max_age_ns, name))
    if not factors:
        raise ValueError("No local daily factor files found")

    global_seen = {}
    for root in global_roots:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "complete":
            raise ValueError(f"Global factor collection is incomplete: {root}")
        items = summary["factors"]
        spy = pd.read_parquet(root / "prepared" / "SPY" / "1d.parquet")
        us_sessions = sorted(spy["timestamp"].dt.tz_convert("America/New_York").dt.date.tolist())
        if len(us_sessions) != len(set(us_sessions)):
            raise ValueError(f"Duplicate US reference sessions: {root}")
        for item in items:
            if item.get("status") == "invalid_source":
                raise ValueError(f"Unavailable global factor {item['symbol']} in {root}")
            name = item["symbol"]
            path = root / "prepared" / name / "1d.parquet"
            identity = (summary["source"], summary["revision"], item["source_sha256"], _source_hash(path))
            if name in global_seen:
                if global_seen[name] != identity:
                    raise ValueError(f"Conflicting global factor {name}")
                continue
            global_seen[name] = identity
            data = global_returns(pd.read_parquet(path), us_sessions)
            factors.append(_factor("global:" + summary["source"], name,
                                   data["factor_available_at"], data["factor_return"], 5))

    india_identity = json.loads((india / "identity.json").read_text(encoding="utf-8"))
    india_summary = json.loads((india / "summary.json").read_text(encoding="utf-8"))
    if (india_identity["repository"] != REPOSITORY or india_identity["revision"] != REVISION
            or india_summary["status"] != "complete"):
        raise ValueError("Unsupported or incomplete India factor snapshot")
    for name, columns in FILES.items():
        path = india / "source" / Path(name).name
        content = path.read_bytes()
        info = india_identity["catalog"][name]
        if len(content) != info["bytes"] or _blob_hash(content) != info["git_blob_sha1"]:
            raise ValueError(f"India factor snapshot changed: {path}")
        for factor_name, values in _read_returns(path, columns, sessions).items():
            factors.append(_local_factor(factor_name, values, "india:" + REPOSITORY))

    treasury_summary = json.loads((treasury / "summary.json").read_text(encoding="utf-8"))
    if treasury_summary["status"] != "complete":
        raise ValueError("Treasury factor snapshot is incomplete")
    sources = json.loads((treasury / "sources.json").read_text(encoding="utf-8"))
    for name in TABLES:
        frames = []
        for key in sorted(sources):
            if not key.startswith(name + "_"):
                continue
            path = treasury / "source" / f"{key}.csv"
            if _source_hash(path) != sources[key]["sha256"]:
                raise ValueError(f"Treasury factor snapshot changed: {path}")
            frames.append(pd.read_csv(path, na_values=["N/A", "."]))
        if not frames:
            raise ValueError(f"Missing Treasury factor table: {name}")
        for factor_name, data in rate_factors(frames, name).items():
            factors.append(_factor("us_treasury", factor_name,
                                   data["factor_available_at"], data["factor_return"], 5))

    price_summary = json.loads((prices / "summary.json").read_text(encoding="utf-8"))
    if price_summary["status"] != "complete":
        raise ValueError("Independent price snapshot is incomplete")
    for source in ("RBI_FX", "BRENT", "WTI"):
        path = prices / f"{source}.csv"
        if _source_hash(path) != price_summary["sources"][source]["sha256"]:
            raise ValueError(f"Independent price snapshot changed: {path}")
        frame = pd.read_csv(path)
        date_column = "date" if source == "RBI_FX" else "Date"
        frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
        if frame[date_column].isna().any() or frame[date_column].duplicated().any():
            raise ValueError(f"Invalid dates in {path}")
        frame = frame.sort_values(date_column).reset_index(drop=True)
        columns = ("USD", "EUR", "GBP", "JPY", "AED", "IDR") if source == "RBI_FX" else ("Price",)
        timezone = "Asia/Kolkata" if source == "RBI_FX" else "America/New_York"
        lag = 1 if source == "RBI_FX" else 7
        available = (frame[date_column] + timedelta(days=lag)).dt.tz_localize(timezone).dt.tz_convert("UTC")
        for column in columns:
            values = pd.to_numeric(frame[column], errors="raise")
            if (not np.isfinite(values.dropna()).all()
                    or (source == "RBI_FX" and (values.dropna() <= 0).any())):
                raise ValueError(f"Invalid {source} {column}")
            change = values.where(values > 0).pct_change(fill_method=None).where(
                (frame[date_column] - frame[date_column].shift()).dt.days <= 4)
            for kind, data in (("level", values), ("change", change)):
                valid = data.notna()
                factors.append(_factor(source, f"{column}:{kind}", available[valid], data[valid], 14))
    keys = [(factor.source, factor.name) for factor in factors]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate factor identities")
    return factors


def _groups(examples: pd.DataFrame, events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = _ns(examples["available_at"])
    targets = examples["target_log_return"].to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all() or len(times) != len(targets):
        raise ValueError("Invalid next-candle targets")
    if len(targets):
        targets = targets - targets.mean()
    ids = np.searchsorted(events, times, side="right")
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=targets)
    squares = np.bincount(inverse, weights=targets * targets)
    return times[first], count, total, squares


def _correlation(factor: Factor, groups: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 minimum: int) -> tuple[int, float | None, str]:
    times, counts, totals, squares = groups
    if not len(factor.times) or not len(times):
        return 0, None, "insufficient_pairs"
    indices = np.searchsorted(factor.times, times, side="right") - 1
    valid = indices >= 0
    safe = np.maximum(indices, 0)
    valid &= times - factor.times[safe] <= factor.max_age_ns
    n = int(counts[valid].sum())
    if n < minimum:
        return n, None, "insufficient_pairs"
    x = factor.values[safe[valid]]
    weights, y, yy = counts[valid], totals[valid], squares[valid]
    sx, sy = float(np.dot(weights, x)), float(y.sum())
    centered_x = x - sx / n
    if np.all(x == x[0]):
        return n, None, "constant_values"
    xx = float(np.dot(weights, centered_x * centered_x))
    yy_sum = float(yy.sum()) - sy * sy / n
    if xx <= 0 or yy_sum <= 0:
        return n, None, "constant_values"
    correlation = float(np.dot(centered_x, y)) / math.sqrt(xx * yy_sum)
    if not math.isfinite(correlation) or abs(correlation) > 1.00000001:
        raise ValueError(f"Invalid correlation for {factor.source}:{factor.name}")
    return n, max(-1.0, min(1.0, correlation)), "scored"


def _events(factors: list[Factor]) -> np.ndarray:
    events: set[int] = set()
    for factor in factors:
        events.update(factor.times.tolist())
        events.update((factor.times + factor.max_age_ns + 1).tolist())
    return np.asarray(sorted(events), dtype=np.int64)


def run(*, membership: Path, stocks: Path, intraday: Path, daily: Path, multisession: Path,
        global_roots: list[Path], india: Path, treasury: Path, prices: Path, output: Path,
        start: date, end: date, minimum: int = 60, max_rows: int = 500_000,
        symbols: list[str] | None = None, resume: bool = False) -> dict:
    if not COVERAGE_START <= start < end <= COVERAGE_END or minimum < 3 or max_rows < 1:
        raise ValueError("Invalid correlation dates, minimum observations or row limit")
    members = parse_nifty_membership(membership.read_text(encoding="utf-8-sig"))
    eligible = {entry.symbol for entry in members if entry.valid_from < end
                and (entry.valid_to is None or entry.valid_to > start)}
    if symbols is not None:
        if not symbols or len(symbols) != len(set(symbols)) or not set(symbols) <= eligible:
            raise ValueError("Requested symbols must be unique historical Nifty members in the window")
        eligible = set(symbols)
    ordered = sorted(eligible)
    if not ordered:
        raise ValueError("No historical Nifty members in the requested window")
    roots = {tf: intraday if tf in INTRADAY_TIMEFRAMES else daily if tf == "1d" else multisession
             for tf in TIMEFRAMES}
    inventories = {root: json.loads((root / "coverage.json").read_text(encoding="utf-8"))
                   for root in sorted(set(roots.values()))}
    for path in (india / "summary.json", treasury / "summary.json", prices / "summary.json"):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        cutoff = snapshot.get("train_end_exclusive", snapshot.get("cutoff_exclusive"))
        if not cutoff or end > date.fromisoformat(cutoff):
            raise ValueError(f"Requested window extends beyond the factor snapshot cutoff: {path}")
    inputs = [membership, stocks / "day" / "RELIANCE" / "1d.csv.gz",
              *(root / "coverage.json" for root in inventories)]
    local_paths = _daily_paths(stocks / "day")
    inputs.extend(local_paths)
    for root, inventory in inventories.items():
        for job in inventory["jobs"]:
            if job["kind"] != "stock" or job["symbol"] not in eligible or job["status"] != "prepared":
                continue
            folder = root / "stock" / _safe_symbol(job["symbol"]) / job["timeframe"]
            for index in range(job["chunks"]):
                part = folder / f"part-{index:06d}.parquet"
                inputs.extend([part, part.with_suffix(".json")])
    for root in global_roots:
        inputs.append(root / "summary.json")
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        inputs.extend(root / "prepared" / item["symbol"] / "1d.parquet" for item in summary["factors"])
    inputs.extend([india / "identity.json", india / "summary.json",
                   *(india / "source" / Path(name).name for name in FILES),
                   treasury / "summary.json", treasury / "sources.json",
                   *sorted((treasury / "source").glob("*.csv")), prices / "summary.json",
                   *(prices / f"{name}.csv" for name in ("RBI_FX", "BRENT", "WTI"))])
    code = [Path(__file__), Path(__file__).with_name("nifty_candle_examples.py"),
            Path(__file__).with_name("nifty_preparation.py"), Path(__file__).with_name("nifty_price_model.py"),
            Path(__file__).with_name("factor_research.py"), Path(__file__).with_name("global_factor_research.py"),
            Path(__file__).with_name("nifty_public_factor_research.py"),
            Path(__file__).with_name("nifty_treasury_factor_research.py"),
            Path(__file__).with_name("nifty_universe.py"), Path(__file__).with_name("control.py"),
            Path(__file__).with_name("nifty_price_training.py"),
            Path(__file__).parent.parent / "data_pipeline" / "huggingface_stocks.py"]
    identity = {"version": 1, "start": start.isoformat(), "end_exclusive": end.isoformat(),
                "minimum": minimum, "max_rows": max_rows, "symbols": ordered,
                "inputs": [_fingerprint(path) for path in inputs],
                "code": {str(path.resolve()): _source_hash(path) for path in code}}
    if output.exists():
        if not resume:
            raise FileExistsError(f"Output directory already exists: {output}")
    else:
        if resume:
            raise FileNotFoundError(f"Output directory does not exist: {output}")
        output.mkdir(parents=True)
    with _price_lock(output):
        manifest = output / "identity.json"
        if resume:
            if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("Inputs or settings changed; use a new output directory")
        else:
            _write_json(manifest, identity)
        control = TrainingControl(output)
        if resume:
            control.request("continue")

        def boundary(stock: str, timeframe: str, completed: int) -> None:
            state = {"stock": stock, "timeframe": timeframe, "completed_stocks": completed,
                     "total_stocks": len(ordered)}
            _write_json(output / "state.json", {**state, "status": "running"})
            if not control.checkpoint_boundary(lambda: _write_json(output / "state.json",
                                                                   {**state, "status": "checkpointed"})):
                _write_json(output / "state.json", {**state, "status": "stopped"})
                raise PreparationStopped("Stopped between factor batches; rerun with --resume")

        sessions = list(load_closes(stocks / "day" / "RELIANCE" / "1d.csv.gz").index)
        boundary("", "loading_factors", 0)
        factors = _read_sources(local_paths, global_roots, india, treasury, prices, sessions)
        events = _events(factors)
        folder = output / "stocks"
        folder.mkdir(exist_ok=True)
        counts = {}
        for stock_index, symbol in enumerate(ordered):
            boundary(symbol, "starting", stock_index)
            csv_path = folder / f"{_safe_symbol(symbol)}.csv"
            checkpoint = csv_path.with_suffix(".json")
            if checkpoint.exists():
                meta = json.loads(checkpoint.read_text(encoding="utf-8"))
                if not csv_path.is_file() or _source_hash(csv_path) != meta["sha256"]:
                    raise ValueError(f"Completed stock output changed: {csv_path}")
                counts[symbol] = meta["timeframes"]
                continue
            temp = csv_path.with_suffix(".csv.tmp")
            timeframe_counts = {}
            with temp.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["stock", "timeframe", "source", "factor", "paired_examples",
                                 "correlation", "status", "qualifies_at_0_49"])
                for timeframe in TIMEFRAMES:
                    boundary(symbol, timeframe, stock_index)
                    jobs = [job for job in inventories[roots[timeframe]]["jobs"]
                            if job["kind"] == "stock" and job["symbol"] == symbol
                            and job["timeframe"] == timeframe]
                    if len(jobs) > 1:
                        raise ValueError(f"Duplicate preparation jobs for {symbol} {timeframe}")
                    data = None if not jobs else load_prepared_candles(roots[timeframe], symbol, timeframe,
                                                                       max_rows=max_rows)
                    status = "missing_stock_candles" if data is None else None
                    if data is not None:
                        try:
                            examples = build_candle_examples(
                                symbol, timeframe, data, sessions,
                                tuple(entry for entry in members if entry.symbol == symbol), start=start, end=end)
                        except NoCandleExamples:
                            status = "no_eligible_examples"
                        else:
                            groups = _groups(examples, events)
                            timeframe_counts[timeframe] = {"examples": len(examples)}
                            del examples, data
                    if status:
                        timeframe_counts[timeframe] = {"examples": 0, "status": status}
                    local_name = _safe_symbol(local_symbol(symbol, members))
                    for index, factor in enumerate(factors):
                        if index and index % 256 == 0:
                            boundary(symbol, timeframe, stock_index)
                        if factor.self_symbol == local_name:
                            n, value, row_status = 0, None, "target_stock_excluded"
                        elif status:
                            n, value, row_status = 0, None, status
                        else:
                            n, value, row_status = _correlation(factor, groups, minimum)
                        writer.writerow([symbol, timeframe, factor.source, factor.name, n,
                                         "" if value is None else repr(value), row_status,
                                         "" if value is None else str(abs(value) >= 0.49).lower()])
            _replace_with_retry(temp, csv_path)
            meta = {"sha256": _source_hash(csv_path), "timeframes": timeframe_counts}
            _write_json(checkpoint, meta)
            counts[symbol] = timeframe_counts
            print(f"Completed {stock_index + 1}/{len(ordered)}: {symbol}", flush=True)
        if (_daily_paths(stocks / "day") != local_paths
                or [_fingerprint(path) for path in inputs] != identity["inputs"]):
            raise ValueError("Correlation inputs changed during screening")
        summary = {"status": "complete", "research_only": True, "stocks": counts,
                   "factor_count": len(factors), "timeframes": list(TIMEFRAMES),
                   "minimum_paired_examples": minimum, "start": start.isoformat(),
                   "end_exclusive": end.isoformat(), "threshold_absolute": 0.49,
                   "limitations": [
                       "Daily factor features against each next-candle log return; factors are not intraday bars.",
                       "Publication times, point-in-time revisions, source rights and adjustment methods are unverified.",
                       "Daily factors repeat within sessions; paired examples are not independent observations.",
                       "Multiple testing, overlapping proxies and survivorship can produce spurious correlations.",
                       "RELIANCE and SPY are calendar proxies, not authoritative exchange calendars.",
                       "Exploratory results must not be used for earlier training examples.",
                   ]}
        _write_json(output / "summary.json", summary)
        _write_json(output / "state.json", {"status": "complete", "completed_stocks": len(ordered),
                                            "total_stocks": len(ordered)})
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--stocks", type=Path, required=True)
    parser.add_argument("--intraday", type=Path, required=True)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--multisession", type=Path, required=True)
    parser.add_argument("--global-factors", type=Path, action="append", required=True)
    parser.add_argument("--india-audit", type=Path, required=True)
    parser.add_argument("--treasury-audit", type=Path, required=True)
    parser.add_argument("--prices-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--minimum", type=int, default=60)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(membership=args.membership, stocks=args.stocks, intraday=args.intraday, daily=args.daily,
            multisession=args.multisession, global_roots=args.global_factors, india=args.india_audit,
            treasury=args.treasury_audit, prices=args.prices_audit, output=args.output, start=args.start,
            end=args.end, minimum=args.minimum, max_rows=args.max_rows, symbols=args.symbols, resume=args.resume)
    except PreparationStopped as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
