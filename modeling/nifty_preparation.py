"""Inventory and causally prepare local Nifty stock and market-factor candles."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data_pipeline.huggingface_stocks import INTRADAY_TIMEFRAMES, _safe_symbol
from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped, _fingerprint, _write_json
from modeling.nifty_universe import (
    COVERAGE_END,
    COVERAGE_START,
    Membership,
    local_symbol,
    parse_nifty_membership,
)


TIMEFRAMES = tuple(INTRADAY_TIMEFRAMES) + ("1d", "2d", "trading_week")
SOURCE = "https://huggingface.co/datasets/Saintforest/indian-stock-market-minute-data"


def _daily_dates(path: Path) -> list[date]:
    frame = pd.read_csv(path, usecols=["timestamp"])
    timestamps = _timestamps(frame["timestamp"])
    days = timestamps.dt.tz_convert("Asia/Kolkata").dt.date
    if days.duplicated().any() or not days.is_monotonic_increasing:
        raise ValueError(f"Daily sessions are not sorted and unique: {path}")
    return days.tolist()


def _timestamps(values: pd.Series) -> pd.Series:
    if values.isna().any() or not values.astype(str).str.contains(r"(?:Z|[+-]\d\d:?\d\d)$").all():
        raise ValueError("Candle timestamps must include an explicit timezone")
    return pd.to_datetime(values, utc=True, errors="raise")


def _paths(root: Path, name: str, source: str = "minute") -> dict[str, Path]:
    return {tf: root / source / _safe_symbol(name) / f"{tf}.csv.gz" for tf in TIMEFRAMES}


def _factor_names(root: Path) -> list[str]:
    daily = root / "day"
    return sorted(
        p.name for p in daily.iterdir()
        if p.is_dir()
        and (p.name.startswith("NIFTY") or p.name in {"BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"})
        and (p / "1d.csv.gz").is_file()
    )


def _stock_names(intervals: tuple[Membership, ...]) -> dict[str, str]:
    symbols = sorted({entry.symbol for entry in intervals
                      if entry.valid_from <= COVERAGE_END
                      and (entry.valid_to is None or entry.valid_to > COVERAGE_START)})
    names = {symbol: local_symbol(symbol, intervals) for symbol in symbols}
    if len({_safe_symbol(name) for name in names.values()}) != len(names):
        raise ValueError("Multiple membership symbols resolve to the same local filename")
    return names


def _boundary(control: TrainingControl) -> None:
    if not control.checkpoint_boundary(lambda: None):
        raise PreparationStopped("Stopped at a completed preparation checkpoint")


def build_catalog(membership_path: Path, root: Path, output: Path, *, resume: bool = False) -> dict:
    intervals = parse_nifty_membership(membership_path.read_text(encoding="utf-8-sig"))
    stocks = _stock_names(intervals)
    factors = _factor_names(root)
    entries = [("stock", name, local) for name, local in stocks.items()]
    entries += [("factor", name, name) for name in factors]
    reference = root / "minute" / "RELIANCE" / "1d.csv.gz"
    daily_reference = root / "day" / "RELIANCE" / "1d.csv.gz"
    if not reference.is_file():
        raise FileNotFoundError(f"Missing reference session calendar: {reference}")
    if not daily_reference.is_file():
        raise FileNotFoundError(f"Missing daily-source reference session calendar: {daily_reference}")
    inputs = [membership_path, reference, daily_reference]
    for _, _, local in entries:
        inputs.extend(_paths(root, local).values())
        inputs.append(root / "day" / _safe_symbol(local) / "1d.csv.gz")
    identity = {
        "version": 2,
        "membership_sha256": hashlib.sha256(membership_path.read_bytes()).hexdigest(),
        "inputs": [_fingerprint(path) for path in inputs],
    }
    manifest = output / "identity.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Catalog already exists: {output}")
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Catalog inputs changed or identity is missing; use a new directory")
    else:
        if resume:
            raise FileNotFoundError(f"Catalog does not exist: {output}")
        output.mkdir(parents=True)
        _write_json(manifest, identity)
    control = TrainingControl(output)
    if resume:
        control.request("continue")
    cache = output / "entries"
    cache.mkdir(exist_ok=True)
    reference_days = set(_daily_dates(reference))
    daily_reference_days = set(_daily_dates(daily_reference))
    result = []
    for kind, name, local in entries:
        _boundary(control)
        checkpoint = cache / f"{kind}_{_safe_symbol(name)}.json"
        if checkpoint.is_file():
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            paths = _paths(root, local)
            daily_path = paths["1d"]
            days = _daily_dates(daily_path) if daily_path.is_file() else []
            if kind == "stock":
                relevant = [entry for entry in intervals if entry.symbol == name]
                expected = {day for day in reference_days
                            if COVERAGE_START <= day <= COVERAGE_END
                            and any(entry.contains(day) for entry in relevant)}
                daily_expected = {day for day in daily_reference_days
                                  if COVERAGE_START <= day <= COVERAGE_END
                                  and any(entry.contains(day) for entry in relevant)}
            else:
                expected = reference_days
                daily_expected = daily_reference_days
            missing = sorted(expected - set(days))
            daily_source = root / "day" / _safe_symbol(local) / "1d.csv.gz"
            daily_days = _daily_dates(daily_source) if daily_source.is_file() else []
            record = {
                "kind": kind, "symbol": name, "local_symbol": local,
                "daily_first": days[0].isoformat() if days else None,
                "daily_last": days[-1].isoformat() if days else None,
                "daily_sessions": len(days),
                "missing_reference_sessions": [day.isoformat() for day in missing],
                "daily_source": {
                    "path": str(daily_source.resolve()), "exists": daily_source.is_file(),
                    "first": daily_days[0].isoformat() if daily_days else None,
                    "last": daily_days[-1].isoformat() if daily_days else None,
                    "sessions": len(daily_days),
                    "missing_reference_sessions": [day.isoformat() for day in sorted(daily_expected - set(daily_days))],
                },
                "timeframes": {tf: {"path": str(path.resolve()), "exists": path.is_file()}
                               for tf, path in paths.items()},
            }
            _write_json(checkpoint, record)
        result.append(record)
    catalog = {
        "source": SOURCE,
        "source_license": "Dataset card declares MIT; confirm data rights and terms before redistribution.",
        "provenance_note": "Local files follow the repository's configured Hugging Face source; upstream revision is not recorded.",
        "factor_note": "Name-based local index/fund candidates, not independent global or economic factors; overlap and constituent inclusion are possible.",
        "coverage_note": "First/last and missing sessions refer to minute-derived daily candles only; existence does not establish intraday coverage.",
        "reference_calendar": str(reference.resolve()),
        "daily_source_reference_calendar": str(daily_reference.resolve()),
        "records": result,
    }
    _write_json(output / "catalog.json", catalog)
    return catalog


def _aggregate_sessions(daily_path: Path, timeframe: str) -> dict[date, tuple[date, int]]:
    days = _daily_dates(daily_path)
    groups: dict[date, list[date]] = {}
    if timeframe == "2d":
        for index in range(0, len(days), 2):
            group = days[index:index + 2]
            groups[group[0]] = group
    elif timeframe == "trading_week":
        for day in days:
            monday = day - timedelta(days=day.weekday())
            groups.setdefault(monday, []).append(day)
    return {first: (group[-1], len(group)) for first, group in groups.items()}


def available_candles(
    frame: pd.DataFrame,
    timeframe: str,
    aggregate_sessions: dict[date, tuple[date, int]] | None = None,
    membership: tuple[Membership, ...] | None = None,
    *,
    daily_source: bool = False,
) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume",
                "source_bars", "trading_days", "is_partial"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing candle columns: {sorted(required - set(frame.columns))}")
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if daily_source and timeframe != "1d":
        raise ValueError("The daily source is supported only for individual daily candles")
    result = frame.loc[:, sorted(required)].copy()
    result["timestamp"] = _timestamps(result["timestamp"])
    if result["timestamp"].duplicated().any() or not result["timestamp"].is_monotonic_increasing:
        raise ValueError("Candle timestamps are not sorted and unique")
    local = result["timestamp"].dt.tz_convert("Asia/Kolkata")
    starts = local.dt.date
    if timeframe in INTRADAY_TIMEFRAMES:
        minute = local.dt.hour * 60 + local.dt.minute
        width = INTRADAY_TIMEFRAMES[timeframe]
        if ((minute < 555) | (minute >= 930) | ((minute - 555) % width != 0)).any():
            raise ValueError("Intraday bar is outside the expected session or not session-aligned")
        end = local + timedelta(minutes=width)
        close = local.dt.normalize() + timedelta(hours=15, minutes=30)
        result["available_at"] = end.where(end <= close, close).dt.tz_convert("UTC")
        final_days = starts
    elif timeframe == "1d":
        minute = local.dt.hour * 60 + local.dt.minute
        if ((minute != 0) & ((minute < 555) | (minute >= 930))).any():
            raise ValueError("Daily candle labels must be midnight or within the market session")
        final_days = starts
        delay = timedelta(days=1) if daily_source else timedelta(hours=15, minutes=30)
        result["available_at"] = (local.dt.normalize() + delay).dt.tz_convert("UTC")
    else:
        if aggregate_sessions is None:
            raise ValueError("Aggregated candles require their underlying daily session dates")
        groups = [aggregate_sessions.get(day) for day in starts]
        if any(group is None for group in groups):
            raise ValueError("Aggregated candle is missing its underlying daily sessions")
        final_days = pd.Series([group[0] for group in groups if group is not None], index=result.index)
        counts = [group[1] for group in groups if group is not None]
        if list(pd.to_numeric(result["trading_days"], errors="raise")) != counts:
            raise ValueError("Aggregated trading-day counts do not match daily source")
        if timeframe == "2d":
            incomplete = pd.Series(counts, index=result.index) != 2
            partial = result["is_partial"].astype(str).str.lower().eq("true")
            if (incomplete & ~partial).any():
                raise ValueError("Incomplete two-session candle is not marked partial")
            result["available_at"] = pd.to_datetime(final_days.astype(str)).dt.tz_localize(
                "Asia/Kolkata").add(timedelta(hours=15, minutes=30)).dt.tz_convert("UTC")
        else:
            if (local.dt.weekday != 0).any() or (local != local.dt.normalize()).any():
                raise ValueError("Weekly candle labels must be Monday local midnight")
            result["available_at"] = (local + timedelta(days=7)).dt.tz_convert("UTC")
    if timeframe == "2d":
        result = result.loc[~incomplete].copy()
        starts = starts.loc[result.index]
        final_days = final_days.loc[result.index]
    if membership is not None:
        eligible = pd.Series(False, index=result.index)
        for entry in membership:
            eligible |= (starts >= entry.valid_from) & (
                True if entry.valid_to is None else final_days < entry.valid_to
            )
        eligible &= (starts >= COVERAGE_START) & (final_days <= COVERAGE_END)
        result = result.loc[eligible].copy()
    return result.reset_index(drop=True)


def join_available(stock: pd.DataFrame, factor: pd.DataFrame, *, tolerance: timedelta) -> pd.DataFrame:
    if tolerance <= timedelta(0):
        raise ValueError("A positive maximum factor age is required")
    if not isinstance(stock["available_at"].dtype, pd.DatetimeTZDtype) or not isinstance(
        factor["available_at"].dtype, pd.DatetimeTZDtype
    ):
        raise ValueError("Availability timestamps must have explicit timezones")
    if stock["available_at"].isna().any() or factor["available_at"].isna().any():
        raise ValueError("Availability timestamps cannot be missing")
    if factor["available_at"].duplicated().any():
        raise ValueError("Factor availability timestamps must be unique")
    left = stock.copy()
    right = factor.rename(columns={"available_at": "factor_available_at"}).copy()
    left["available_at"] = left["available_at"].dt.tz_convert("UTC")
    right["factor_available_at"] = right["factor_available_at"].dt.tz_convert("UTC")
    return pd.merge_asof(
        left.sort_values("available_at"), right.sort_values("factor_available_at"),
        left_on="available_at", right_on="factor_available_at",
        direction="backward", tolerance=tolerance, suffixes=("", "_factor"),
    )


def daily_source_aggregates(
    frame: pd.DataFrame, timeframe: str, membership: tuple[Membership, ...] | None = None,
) -> pd.DataFrame:
    if timeframe not in {"2d", "trading_week"}:
        raise ValueError("Daily-source aggregation supports only 2d and trading_week")
    daily = available_candles(frame, "1d", daily_source=True)
    if daily.empty:
        return daily
    days = daily["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
    if days.duplicated().any() or (pd.to_numeric(daily["trading_days"], errors="raise") != 1).any():
        raise ValueError("Daily-source aggregation requires one candle per session")
    daily["_session"] = days
    if timeframe == "2d":
        daily["_group"] = [index // 2 for index in range(len(daily))]
    else:
        daily["_group"] = days.map(lambda day: day - timedelta(days=day.weekday()))
    groups = daily.groupby("_group", sort=True, as_index=False).agg(
        timestamp=("timestamp", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        source_bars=("source_bars", "sum"), trading_days=("_session", "size"),
        is_partial=("is_partial", "any"), last_session=("_session", "last"),
        available_at=("available_at", "last"),
    )
    if timeframe == "2d":
        groups = groups.loc[groups["trading_days"] == 2].copy()
    else:
        groups["is_partial"] = groups["is_partial"] | (groups["trading_days"] < 5)
        next_monday = pd.to_datetime(groups["_group"].astype(str)) + timedelta(days=7)
        groups["available_at"] = next_monday.dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
    starts = groups["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
    last = groups["last_session"]
    if membership is not None:
        eligible = pd.Series(False, index=groups.index)
        for entry in membership:
            eligible |= (starts >= entry.valid_from) & (True if entry.valid_to is None else last < entry.valid_to)
        eligible &= (starts >= COVERAGE_START) & (last <= COVERAGE_END)
        groups = groups.loc[eligible]
    return groups.drop(columns=["_group", "last_session"]).reset_index(drop=True)


def prepare(
    membership_path: Path, root: Path, output: Path, *, stock_symbols: list[str] | None,
    factor_symbols: list[str] | None, timeframes: list[str], chunk_rows: int = 50000,
    resume: bool = False, candle_source: str = "minute",
) -> dict:
    if chunk_rows < 1 or not timeframes or any(tf not in TIMEFRAMES for tf in timeframes):
        raise ValueError("Invalid chunk size or timeframe selection")
    if candle_source not in {"minute", "day"} or (
        candle_source == "day" and any(tf not in {"1d", "2d", "trading_week"} for tf in timeframes)
    ):
        raise ValueError("The day candle source supports only 1d, 2d and trading_week")
    intervals = parse_nifty_membership(membership_path.read_text(encoding="utf-8-sig"))
    stocks, factors = _stock_names(intervals), _factor_names(root)
    selected_stocks = sorted(stocks if stock_symbols is None else stock_symbols)
    selected_factors = sorted(factors if factor_symbols is None else factor_symbols)
    if set(selected_stocks) - set(stocks) or set(selected_factors) - set(factors):
        raise ValueError("Unknown stock or factor selection")
    jobs = [("stock", symbol, stocks[symbol], tf) for symbol in selected_stocks for tf in timeframes]
    jobs += [("factor", symbol, symbol, tf) for symbol in selected_factors for tf in timeframes]
    if len(set(jobs)) != len(jobs):
        raise ValueError("Duplicate preparation job")
    input_paths = [membership_path]
    for _, _, local, tf in jobs:
        paths = _paths(root, local, candle_source)
        input_paths.append(paths["1d"] if candle_source == "day" else paths[tf])
        if candle_source == "minute" and tf in {"2d", "trading_week"}:
            input_paths.append(paths["1d"])
    identity = {
        "version": 1, "jobs": jobs, "chunk_rows": chunk_rows,
        "membership_sha256": hashlib.sha256(membership_path.read_bytes()).hexdigest(),
        "inputs": [_fingerprint(path) for path in input_paths],
    }
    if candle_source == "day":
        identity["candle_source"] = candle_source
    manifest = output / "identity.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Preparation already exists: {output}")
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(identity)):
            raise ValueError("Preparation inputs or settings changed; use a new directory")
    else:
        if resume:
            raise FileNotFoundError(f"Preparation does not exist: {output}")
        output.mkdir(parents=True)
        _write_json(manifest, identity)
    control = TrainingControl(output)
    if resume:
        control.request("continue")
    results = []
    for kind, symbol, local, timeframe in jobs:
        _boundary(control)
        paths = _paths(root, local, candle_source)
        source = paths["1d"] if candle_source == "day" else paths[timeframe]
        daily = paths["1d"]
        if not source.is_file() or (candle_source == "minute" and timeframe in {"2d", "trading_week"} and not daily.is_file()):
            results.append({"kind": kind, "symbol": symbol, "timeframe": timeframe, "status": "missing_input"})
            continue
        folder = output / kind / _safe_symbol(symbol) / timeframe
        folder.mkdir(parents=True, exist_ok=True)
        sessions = _aggregate_sessions(daily, timeframe) if candle_source == "minute" and timeframe in {"2d", "trading_week"} else None
        member = tuple(item for item in intervals if item.symbol == symbol) if kind == "stock" else None
        previous = None
        chunks = 0
        rows = 0
        columns = ["timestamp", "open", "high", "low", "close", "volume",
                   "source_bars", "trading_days", "is_partial"]
        reader = pd.read_csv(source, chunksize=chunk_rows, usecols=columns)
        if candle_source == "day" and timeframe in {"2d", "trading_week"}:
            first = next(reader, None)
            if first is not None and next(reader, None) is not None:
                raise ValueError(f"{source}: daily-source aggregates exceed --chunk-rows; increase the limit")
            frames = [] if first is None else [first]
        else:
            frames = reader
        for index, frame in enumerate(frames):
            _boundary(control)
            timestamps = _timestamps(frame["timestamp"])
            if previous is not None and timestamps.iloc[0] <= previous:
                raise ValueError(f"Unsorted or duplicate candle across chunks: {source}")
            previous = timestamps.iloc[-1]
            target = folder / f"part-{index:06d}.parquet"
            checkpoint = folder / f"part-{index:06d}.json"
            if checkpoint.is_file():
                data = json.loads(checkpoint.read_text(encoding="utf-8"))
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != data["sha256"]:
                    raise ValueError(f"Prepared checkpoint is missing or corrupt: {target}")
            else:
                try:
                    if candle_source == "day" and timeframe in {"2d", "trading_week"}:
                        prepared = daily_source_aggregates(frame, timeframe, member)
                    elif candle_source == "day":
                        prepared = available_candles(frame, timeframe, sessions, member, daily_source=True)
                    else:
                        prepared = available_candles(frame, timeframe, sessions, member)
                except ValueError as exc:
                    raise ValueError(f"{source}: {exc}") from exc
                temporary = target.with_suffix(".tmp")
                prepared.to_parquet(temporary, index=False)
                temporary.replace(target)
                data = {"rows": len(prepared), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
                _write_json(checkpoint, data)
            if not isinstance(data.get("rows"), int) or data["rows"] < 0:
                raise ValueError(f"Invalid prepared row count: {checkpoint}")
            rows += data["rows"]
            chunks += 1
        results.append({"kind": kind, "symbol": symbol, "timeframe": timeframe,
                        "status": "prepared" if rows else "no_eligible_rows", "chunks": chunks, "rows": rows})
    summary = {"status": "complete_with_gaps" if any(r["status"] != "prepared" for r in results) else "complete",
               "jobs": results}
    _write_json(output / "summary.json", summary)
    return summary


def report_coverage(output: Path) -> dict:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    jobs = []
    for job in summary["jobs"]:
        record = dict(job)
        if record["status"] != "missing_input":
            folder = output / record["kind"] / _safe_symbol(record["symbol"]) / record["timeframe"]
            checkpoints = sorted(folder.glob("part-*.json"))
            if len(checkpoints) != record["chunks"]:
                raise ValueError(f"Missing preparation checkpoints: {folder}")
            rows = 0
            for checkpoint in checkpoints:
                if not checkpoint.with_suffix(".parquet").is_file():
                    raise ValueError(f"Missing prepared chunk for checkpoint: {checkpoint}")
                data = json.loads(checkpoint.read_text(encoding="utf-8"))
                if not isinstance(data.get("rows"), int) or data["rows"] < 0:
                    raise ValueError(f"Invalid prepared row count: {checkpoint}")
                rows += data["rows"]
            record.update(rows=rows, status="prepared" if rows else "no_eligible_rows")
        jobs.append(record)
    report = {"status": "complete_with_gaps" if any(job["status"] != "prepared" for job in jobs) else "complete",
              "basis": "Row counts from preparation checkpoints; missing source files are listed separately.",
              "jobs": jobs}
    _write_json(output / "coverage.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["catalog", "prepare", "coverage"])
    parser.add_argument("--membership", type=Path, default=Path("data") / "external" / "nse_index_membership_0786cd915974.csv")
    parser.add_argument("--stocks-root", type=Path, default=Path("data") / "stocks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stocks", nargs="*", default=None)
    parser.add_argument("--factors", nargs="*", default=None)
    parser.add_argument("--timeframes", nargs="+", choices=TIMEFRAMES, default=list(TIMEFRAMES))
    parser.add_argument("--chunk-rows", type=int, default=50000)
    parser.add_argument("--candle-source", choices=["minute", "day"], default="minute")
    args = parser.parse_args()
    try:
        if args.command == "catalog":
            result = build_catalog(args.membership, args.stocks_root, args.output, resume=args.resume)
            print(f"Cataloged {len(result['records'])} stock/factor series in {args.output}")
        elif args.command == "prepare":
            result = prepare(args.membership, args.stocks_root, args.output, stock_symbols=args.stocks,
                             factor_symbols=args.factors, timeframes=args.timeframes,
                             chunk_rows=args.chunk_rows, resume=args.resume, candle_source=args.candle_source)
            print(f"Preparation {result['status']}: {args.output}")
        else:
            result = report_coverage(args.output)
            print(f"Coverage {result['status']}: {args.output}")
    except PreparationStopped as exc:
        print(str(exc))


if __name__ == "__main__":
    main()
