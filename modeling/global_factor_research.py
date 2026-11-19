"""Screen global factors against next-session Nifty returns without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import PreparationStopped, _fingerprint, _write_json, load_closes
from modeling.nifty_preparation import _daily_dates
from modeling.nifty_universe import COVERAGE_END, COVERAGE_START, Membership, local_symbol, parse_nifty_membership


def near_close_flags(frame: pd.DataFrame, minutes: pd.DataFrame, max_age_minutes: int) -> pd.Series:
    if max_age_minutes < 0:
        raise ValueError("Maximum close age must not be negative")
    required = {"timestamp", "available_at", "close"}
    if not required.issubset(minutes.columns):
        raise ValueError("Global minute factor is missing required columns")
    minutes = minutes.copy().sort_values("timestamp")
    if (not isinstance(minutes["timestamp"].dtype, pd.DatetimeTZDtype)
            or not isinstance(minutes["available_at"].dtype, pd.DatetimeTZDtype)
            or minutes["timestamp"].isna().any() or minutes["available_at"].isna().any()
            or minutes["timestamp"].duplicated().any()
            or not (minutes["available_at"] == minutes["timestamp"] + timedelta(minutes=1)).all()):
        raise ValueError("Invalid global minute factor timestamps")
    local = minutes["timestamp"].dt.tz_convert("America/New_York")
    minutes["session"] = local.dt.date
    latest = minutes.groupby("session", sort=False).tail(1).set_index("session")
    day = frame["timestamp"].dt.tz_convert("America/New_York").dt.date
    matched = latest.reindex(day)
    if matched["available_at"].isna().any():
        raise ValueError("Daily factor session has no minute observation")
    minute_close = pd.to_numeric(matched["close"], errors="raise").to_numpy()
    daily_close = pd.to_numeric(frame["close"], errors="raise").to_numpy()
    if not np.isfinite(minute_close).all() or not np.isclose(
        minute_close, daily_close, rtol=1e-10, atol=1e-10
    ).all():
        raise ValueError("Daily factor close does not match last minute observation")
    scheduled_close = pd.to_datetime(day).dt.tz_localize("America/New_York") + timedelta(hours=16)
    observed_at = pd.Series(pd.DatetimeIndex(matched["available_at"]), index=frame.index)
    if (observed_at > frame["available_at"]).any():
        raise ValueError("Daily factor is available before its last minute observation")
    age = scheduled_close - observed_at
    return (age >= timedelta(0)) & (age <= timedelta(minutes=max_age_minutes))


def global_returns(
    frame: pd.DataFrame, reference_sessions: list[date], *,
    minutes: pd.DataFrame | None = None, max_close_age_minutes: int | None = None,
) -> pd.DataFrame:
    columns = {"timestamp", "available_at", "close", "is_partial"}
    if not columns.issubset(frame.columns):
        raise ValueError("Global daily factor is missing required columns")
    frame = frame.copy().sort_values("available_at").reset_index(drop=True)
    if (not isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype)
            or not isinstance(frame["available_at"].dtype, pd.DatetimeTZDtype)
            or frame["timestamp"].isna().any() or frame["available_at"].isna().any()
            or (frame["available_at"] <= frame["timestamp"]).any()):
        raise ValueError("Invalid global factor timestamps")
    day = frame["timestamp"].dt.tz_convert("America/New_York").dt.date
    if day.duplicated().any() or not day.is_monotonic_increasing:
        raise ValueError("Global factor sessions are not sorted and unique")
    close = pd.to_numeric(frame["close"], errors="raise")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Invalid global factor prices")
    if frame["is_partial"].isna().any() or not frame["is_partial"].isin([True, False]).all():
        raise ValueError("Invalid global factor partial-candle flags")
    previous_session = {current: previous for previous, current in zip(reference_sessions, reference_sessions[1:])}
    previous_day = day.shift(1)
    consecutive = pd.Series([previous_session.get(current) == prior for current, prior in zip(day, previous_day)])
    if minutes is None and max_close_age_minutes is None:
        complete = ~frame["is_partial"].astype(bool)
    elif minutes is not None and max_close_age_minutes is not None:
        complete = near_close_flags(frame, minutes, max_close_age_minutes)
    else:
        raise ValueError("Minute observations and maximum close age must be provided together")
    valid = consecutive & complete & complete.shift(1, fill_value=False)
    factor_return = close.div(close.shift(1)).sub(1)
    return pd.DataFrame({
        "factor_available_at": frame.loc[valid, "available_at"].to_numpy(),
        "factor_return": factor_return.loc[valid].to_numpy(),
    }).sort_values("factor_available_at").reset_index(drop=True)


def next_session_targets(
    closes: pd.Series, reference_sessions: list[date], membership: tuple[Membership, ...], cutoff: date,
) -> pd.DataFrame:
    if not COVERAGE_START < cutoff <= COVERAGE_END:
        raise ValueError("Training cutoff is outside supported membership coverage")
    rows = []
    for day, target_day in zip(reference_sessions, reference_sessions[1:]):
        if day >= cutoff or target_day >= cutoff or day not in closes.index or target_day not in closes.index:
            continue
        if not any(interval.contains(day) and interval.contains(target_day) for interval in membership):
            continue
        rows.append((day, target_day, float(closes.loc[target_day] / closes.loc[day] - 1)))
    frame = pd.DataFrame(rows, columns=["session", "target_session", "target_return"])
    frame["available_at"] = pd.to_datetime(frame["session"].astype(str)).dt.tz_localize(
        "Asia/Kolkata").add(timedelta(days=1)).dt.tz_convert("UTC")
    return frame


def correlate(targets: pd.DataFrame, factor: pd.DataFrame, *, max_age: timedelta, minimum: int) -> tuple[int, float] | None:
    if max_age <= timedelta(0) or minimum < 3:
        raise ValueError("Factor age must be positive and minimum observations at least three")
    if targets.empty or factor.empty:
        return None
    joined = pd.merge_asof(
        targets.sort_values("available_at"), factor.sort_values("factor_available_at"),
        left_on="available_at", right_on="factor_available_at",
        direction="backward", tolerance=max_age,
    )
    pairs = joined[["target_return", "factor_return"]].dropna()
    if len(pairs) < minimum:
        return None
    value = pairs["target_return"].corr(pairs["factor_return"])
    return (len(pairs), float(value)) if math.isfinite(value) else None


def run(
    membership_path: Path, stocks_root: Path, factors_root: Path, output: Path, *,
    train_end: date, minimum_observations: int = 60, max_age_days: int = 5,
    resume: bool = False, max_close_age_minutes: int | None = None,
) -> dict:
    if (minimum_observations < 3 or max_age_days < 1
            or (max_close_age_minutes is not None and max_close_age_minutes < 0)):
        raise ValueError("Invalid minimum observations or factor age")
    intervals = parse_nifty_membership(membership_path.read_text(encoding="utf-8-sig"))
    if not COVERAGE_START < train_end <= COVERAGE_END:
        raise ValueError("Training cutoff is outside supported membership coverage")
    symbols = sorted({entry.symbol for entry in intervals
                      if entry.valid_from < train_end
                      and (entry.valid_to is None or entry.valid_to > COVERAGE_START)})
    stock_paths = {symbol: stocks_root / "day" / _safe_symbol(local_symbol(symbol, intervals)) / "1d.csv.gz"
                   for symbol in symbols}
    reference = stocks_root / "day" / "RELIANCE" / "1d.csv.gz"
    factor_summary = factors_root / "summary.json"
    source_info = json.loads(factor_summary.read_text(encoding="utf-8"))
    usable = [item["symbol"] for item in source_info["factors"] if item.get("status") != "invalid_source"]
    unavailable = [item["symbol"] for item in source_info["factors"] if item.get("status") == "invalid_source"]
    if "SPY" not in usable:
        raise ValueError("SPY is required as a US reference session calendar")
    factor_paths = {symbol: factors_root / "prepared" / symbol / "1d.parquet" for symbol in usable}
    minute_paths = ({symbol: factors_root / "prepared" / symbol / "1m.parquet" for symbol in usable}
                    if max_close_age_minutes is not None else {})
    inputs = [membership_path, reference, factor_summary, *factor_paths.values(), *stock_paths.values()]
    inputs.extend(minute_paths.values())
    identity = {
        "version": 1, "membership_sha256": hashlib.sha256(membership_path.read_bytes()).hexdigest(),
        "inputs": [_fingerprint(path) for path in inputs], "train_end": train_end.isoformat(),
        "minimum_observations": minimum_observations, "max_age_days": max_age_days,
    }
    if max_close_age_minutes is not None:
        identity["max_close_age_minutes"] = max_close_age_minutes
    manifest = output / "identity.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Audit output exists: {output}")
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Audit inputs or settings changed; use a new directory")
    else:
        if resume:
            raise FileNotFoundError(f"Audit output does not exist: {output}")
        output.mkdir(parents=True)
        _write_json(manifest, identity)
    control = TrainingControl(output)
    if resume:
        control.request("continue")
    cache = output / "stock_checkpoints"
    cache.mkdir(exist_ok=True)

    def boundary() -> None:
        if not control.checkpoint_boundary(lambda: None):
            raise PreparationStopped("Global correlation audit stopped at a checkpoint")

    nse_sessions = _daily_dates(reference)
    spy = pd.read_parquet(factor_paths["SPY"])
    us_sessions = sorted(spy["timestamp"].dt.tz_convert("America/New_York").dt.date.tolist())
    if len(set(us_sessions)) != len(us_sessions):
        raise ValueError("SPY reference contains duplicate sessions")
    factors = {}
    for symbol, path in factor_paths.items():
        boundary()
        minute_data = (pd.read_parquet(minute_paths[symbol], columns=["timestamp", "available_at", "close"])
                       if max_close_age_minutes is not None else None)
        factors[symbol] = global_returns(pd.read_parquet(path), us_sessions, minutes=minute_data,
                                         max_close_age_minutes=max_close_age_minutes)
    results = []
    missing, no_scorable, no_qualifying = [], [], []
    for symbol in symbols:
        boundary()
        path = stock_paths[symbol]
        if not path.is_file():
            missing.append(symbol)
            continue
        checkpoint = cache / f"{_safe_symbol(symbol)}.json"
        if checkpoint.is_file():
            scored = json.loads(checkpoint.read_text(encoding="utf-8"))
            if any(name not in factors or count < minimum_observations or not math.isfinite(value)
                   or abs(value) > 1.000000001 for name, count, value in scored):
                raise ValueError(f"Invalid global audit checkpoint: {checkpoint}")
        else:
            stock_membership = tuple(entry for entry in intervals if entry.symbol == symbol)
            targets = next_session_targets(load_closes(path), nse_sessions, stock_membership, train_end)
            scored = []
            for name, factor in factors.items():
                match = correlate(targets, factor, max_age=timedelta(days=max_age_days),
                                  minimum=minimum_observations)
                if match is not None:
                    scored.append((name, match[0], match[1]))
            _write_json(checkpoint, scored)
        results.extend((symbol, name, count, value) for name, count, value in scored)
        if not scored:
            no_scorable.append(symbol)
        elif not any(abs(value) >= 0.7 for _, _, value in scored):
            no_qualifying.append(symbol)
    csv_path = output / "correlations.csv"
    temporary = output / "correlations.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["symbol", "factor", "paired_sessions", "correlation", "qualifies"])
        writer.writerows((symbol, factor, count, f"{value:.8f}", abs(value) >= 0.7)
                         for symbol, factor, count, value in results)
    _replace_with_retry(temporary, csv_path)
    summary = {
        "status": "research_only", "factor_candidates": len(usable), "unavailable_factors": unavailable,
        "historical_symbols": len(symbols), "train_end_exclusive": train_end.isoformat(),
        "threshold_absolute": 0.7, "minimum_observations": minimum_observations,
        "maximum_factor_age_days": max_age_days, "correlation_count": len(results),
        "missing_daily_files": missing, "no_scorable_factor": no_scorable,
        "no_qualifying_factor": no_qualifying,
        "method": (
            "Last available US-session return using actual last-minute closes within "
            f"{max_close_age_minutes} minutes of 16:00 New York, including partially observed sessions; "
            "vs next NSE-session stock return; decision at next midnight IST."
            if max_close_age_minutes is not None else
            "Last available complete US-session return vs next NSE-session stock return; decision at next midnight IST."
        ),
        "limitations": [
            "Exploratory daily screening only; not training or a production feature-selection procedure.",
            "SPY and RELIANCE sessions are reference proxies, not authoritative exchange calendars.",
            "Source adjustment method and publication latency are not verified; dataset is for research and education only.",
            "Multiple testing and overlapping funds mean correlations are not evidence of predictive performance.",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, default=Path("data") / "external" / "nse_index_membership_0786cd915974.csv")
    parser.add_argument("--stocks-root", type=Path, default=Path("data") / "stocks")
    parser.add_argument("--factors-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--minimum-observations", type=int, default=60)
    parser.add_argument("--max-age-days", type=int, default=5)
    parser.add_argument("--max-close-age-minutes", type=int, default=None,
                        help="Research-only: permit partial sessions when their observed last bar is this close to 16:00 New York")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.membership, args.stocks_root, args.factors_root, args.output, train_end=args.train_end,
                     minimum_observations=args.minimum_observations, max_age_days=args.max_age_days, resume=args.resume,
                     max_close_age_minutes=args.max_close_age_minutes)
        print(f"Global audit: {result['correlation_count']} correlations; {len(result['no_qualifying_factor'])} stocks without qualifying factors")
    except PreparationStopped as exc:
        print(str(exc))


if __name__ == "__main__":
    main()
