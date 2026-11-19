"""Compare historical stock and peer candles at matching timeframes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import INTRADAY_TIMEFRAMES, _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import PreparationStopped, _fingerprint, _write_json, load_closes
from modeling.nifty_candle_correlations import _source_hash
from modeling.nifty_preparation import TIMEFRAMES, _aggregate_sessions, available_candles
from modeling.nifty_price_training import _price_lock
from modeling.nifty_public_factor_research import FILES, REPOSITORY, REVISION, _blob_hash
from modeling.nifty_treasury_factor_research import TABLES
from modeling.nifty_universe import COVERAGE_END, COVERAGE_START, Membership, local_symbol, parse_nifty_membership


def _read_candles(
    root: Path,
    symbol: str,
    timeframe: str,
    reference: list[date],
    membership: tuple[Membership, ...] | None = None,
) -> pd.DataFrame | None:
    source = "minute" if timeframe in INTRADAY_TIMEFRAMES else "day"
    folder = root / source / _safe_symbol(symbol)
    path = folder / f"{timeframe}.csv.gz"
    daily = folder / "1d.csv.gz"
    if not path.is_file() or (timeframe in {"2d", "trading_week"} and not daily.is_file()):
        return None
    columns = ["timestamp", "open", "high", "low", "close", "volume",
               "source_bars", "trading_days", "is_partial"]
    raw = pd.read_csv(path, usecols=columns)
    groups = _aggregate_sessions(daily, timeframe) if timeframe in {"2d", "trading_week"} else None
    bars = available_candles(raw, timeframe, groups, membership, daily_source=timeframe == "1d")
    close = pd.to_numeric(bars["close"], errors="raise")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError(f"Invalid candle close in {path}")
    if bars["is_partial"].isna().any() or not bars["is_partial"].isin([True, False]).all():
        raise ValueError(f"Invalid candle completeness flag in {path}")
    local = bars["timestamp"].dt.tz_convert("Asia/Kolkata")
    days = local.dt.date
    previous_reference = dict(zip(reference[1:], reference))
    if timeframe in INTRADAY_TIMEFRAMES:
        start = bars["timestamp"].dt.as_unit("ns").astype("int64")
        end = bars["available_at"].dt.as_unit("ns").astype("int64")
        sessions = pd.Series("", index=bars.index)
        consecutive = ((start - start.shift()) == INTRADAY_TIMEFRAMES[timeframe] * 60 * 1_000_000_000)
        consecutive &= days == days.shift()
    else:
        start = days
        if groups is None:
            end = days
            sessions = days.astype(str)
            consecutive = days.map(previous_reference) == days.shift()
        else:
            end = days.map(lambda day: groups[day][0])
            consecutive = days.map(previous_reference) == end.shift()
            sessions = days.astype(str) + "," + end.astype(str)
            if timeframe == "trading_week":
                expected: dict[date, list[date]] = {}
                for day in reference:
                    expected.setdefault(day - timedelta(days=day.weekday()), []).append(day)
                daily_days = list(load_closes(daily).index)
                actual: dict[date, list[date]] = {}
                for day in daily_days:
                    actual.setdefault(day - timedelta(days=day.weekday()), []).append(day)
                complete = days.map(lambda day: actual.get(day) == expected.get(day))
                sessions = days.map(lambda day: ",".join(map(str, actual[day])))
            else:
                complete = end.map(previous_reference) == days
            bars["is_partial"] = bars["is_partial"] | ~complete
    valid = ~bars["is_partial"].astype(bool)
    valid_previous = valid & valid.shift(fill_value=False) & consecutive.fillna(False)
    values = pd.DataFrame({
        "start": start, "end": end, "sessions": sessions, "close": close,
        "previous_start": start.shift(), "previous_end": end.shift(), "previous_sessions": sessions.shift(),
        "return": close.div(close.shift()).sub(1).where(valid_previous),
    })
    values = values.loc[valid].reset_index(drop=True)
    if values.duplicated(["start", "end", "sessions"]).any():
        raise ValueError(f"Duplicate candle intervals in {path}")
    return values


def _coefficients(stock: pd.DataFrame, factor: pd.DataFrame, minimum: int) -> dict[str, object]:
    pairs = stock.merge(factor, on=["start", "end", "sessions"], how="inner", suffixes=("_stock", "_factor"),
                        validate="one_to_one")

    def boundary(frame: pd.DataFrame, first: bool) -> str | None:
        if frame.empty:
            return None
        value = frame["start"].min() if first else frame["start"].max()
        if isinstance(value, (int, np.integer)):
            return pd.Timestamp(int(value), unit="ns", tz="UTC").isoformat()
        return str(value)

    def calculate(left: pd.Series, right: pd.Series) -> tuple[str, float | None, float | None]:
        if len(left) < minimum:
            return "insufficient_pairs", None, None
        if left.nunique() < 2 or right.nunique() < 2:
            return "constant_values", None, None
        return "scored", float(left.corr(right)), float(left.rank().corr(right.rank()))

    price_status, price_pearson, price_spearman = calculate(pairs["close_stock"], pairs["close_factor"])
    valid_returns = (
        pairs["return_stock"].notna() & pairs["return_factor"].notna()
        & (pairs["previous_start_stock"] == pairs["previous_start_factor"])
        & (pairs["previous_end_stock"] == pairs["previous_end_factor"])
        & (pairs["previous_sessions_stock"] == pairs["previous_sessions_factor"])
    )
    returns = pairs.loc[valid_returns]
    return_status, return_pearson, return_spearman = calculate(returns["return_stock"], returns["return_factor"])
    return {
        "price_pairs": len(pairs), "price_first": boundary(pairs, True), "price_last": boundary(pairs, False),
        "price_status": price_status,
        "price_pearson": price_pearson, "price_spearman": price_spearman,
        "return_pairs": len(returns), "return_first": boundary(returns, True),
        "return_last": boundary(returns, False), "return_status": return_status,
        "return_pearson": return_pearson, "return_spearman": return_spearman,
    }


def _observations(dates: pd.Series, values: pd.Series, *, rate: bool = False) -> pd.DataFrame:
    dates = pd.to_datetime(dates, errors="raise")
    if dates.isna().any():
        raise ValueError("Missing daily factor dates")
    numbers = pd.to_numeric(values, errors="raise")
    valid = numbers.notna()
    data = pd.DataFrame({"start": dates.dt.date, "close": numbers}).loc[valid].sort_values("start")
    if data["start"].duplicated().any() or not np.isfinite(data["close"]).all():
        raise ValueError("Duplicate dates or non-finite daily factor values")
    data["end"] = data["start"]
    data["sessions"] = data["start"].astype(str)
    data["previous_start"] = data["start"].shift()
    data["previous_end"] = data["end"].shift()
    data["previous_sessions"] = data["sessions"].shift()
    if rate:
        data["return"] = data["close"].diff()
    else:
        data["return"] = data["close"].div(data["close"].shift()).sub(1).where(
            (data["close"] > 0) & (data["close"].shift() > 0))
    return data.reset_index(drop=True)


def _daily_external(india: Path, treasury: Path, prices: Path) -> tuple[dict[tuple[str, str], pd.DataFrame], list[Path]]:
    result: dict[tuple[str, str], pd.DataFrame] = {}
    paths: list[Path] = []
    identity_path = india / "identity.json"
    summary_paths = [india / "summary.json", treasury / "summary.json", prices / "summary.json"]
    paths.extend([identity_path, *summary_paths])
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity["repository"] != REPOSITORY or identity["revision"] != REVISION:
        raise ValueError("Unsupported India factor snapshot")
    for path in summary_paths:
        if json.loads(path.read_text(encoding="utf-8"))["status"] != "complete":
            raise ValueError(f"Incomplete factor snapshot: {path}")
    for name, columns in FILES.items():
        path = india / "source" / Path(name).name
        paths.append(path)
        content = path.read_bytes()
        info = identity["catalog"][name]
        if len(content) != info["bytes"] or _blob_hash(content) != info["git_blob_sha1"]:
            raise ValueError(f"India factor snapshot changed: {path}")
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        dates = frame["Date"] if path.suffix == ".csv" else pd.Series(frame.index, index=frame.index)
        for column in columns:
            result[("india:" + REPOSITORY, f"{path.stem}:{column}")] = _observations(
                dates.reset_index(drop=True), frame[column].reset_index(drop=True),
                rate=path.stem == "in10y_bond_yield")

    sources_path = treasury / "sources.json"
    paths.append(sources_path)
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    for name in TABLES:
        files = []
        for key in sorted(sources):
            if key.startswith(name + "_"):
                path = treasury / "source" / f"{key}.csv"
                paths.append(path)
                if _source_hash(path) != sources[key]["sha256"]:
                    raise ValueError(f"Treasury factor snapshot changed: {path}")
                files.append(pd.read_csv(path, na_values=["N/A", "."]))
        if not files:
            raise ValueError(f"No Treasury sources for {name}")
        frame = pd.concat(files, ignore_index=True)
        dates = pd.to_datetime(frame["Date"], format="%m/%d/%Y", errors="raise")
        for column in frame.columns.drop("Date"):
            result[("us_treasury", f"{name}:{column}")] = _observations(dates, frame[column], rate=True)

    summary = json.loads((prices / "summary.json").read_text(encoding="utf-8"))
    for source in ("RBI_FX", "BRENT", "WTI"):
        path = prices / f"{source}.csv"
        paths.append(path)
        if _source_hash(path) != summary["sources"][source]["sha256"]:
            raise ValueError(f"Price factor snapshot changed: {path}")
        frame = pd.read_csv(path)
        date_column = "date" if source == "RBI_FX" else "Date"
        columns = ("USD", "EUR", "GBP", "JPY", "AED", "IDR") if source == "RBI_FX" else ("Price",)
        for column in columns:
            result[(source, column)] = _observations(frame[date_column], frame[column])
    return result, paths


def _global_candles(root: Path, symbol: str, timeframe: str) -> pd.DataFrame | None:
    path = root / "prepared" / symbol / f"{timeframe}.parquet"
    if not path.is_file():
        return None
    bars = pd.read_parquet(path)
    close = pd.to_numeric(bars["close"], errors="raise")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError(f"Invalid global candle close in {path}")
    if bars["is_partial"].isna().any() or not bars["is_partial"].isin([True, False]).all():
        raise ValueError(f"Invalid global candle completeness in {path}")
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="raise")
    days = timestamps.dt.tz_convert("America/New_York").dt.date
    if timeframe in INTRADAY_TIMEFRAMES:
        start = timestamps.dt.as_unit("ns").astype("int64")
        end = pd.to_datetime(bars["available_at"], utc=True, errors="raise").dt.as_unit("ns").astype("int64")
        sessions = pd.Series("", index=bars.index)
    elif timeframe == "1d":
        start = end = days
        sessions = days.astype(str)
    else:
        daily_path = root / "prepared" / symbol / "1d.parquet"
        daily = pd.read_parquet(daily_path, columns=["timestamp"])
        daily_days = pd.to_datetime(daily["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
        groups: dict[date, list[date]] = {}
        if timeframe == "2d":
            for index in range(0, len(daily_days), 2):
                group = daily_days.iloc[index:index + 2].tolist()
                groups[group[0]] = group
            labels = days
        else:
            for day in daily_days:
                groups.setdefault(day - timedelta(days=day.weekday()), []).append(day)
            labels = days.map(lambda day: day - timedelta(days=day.weekday()))
        if any(day not in groups for day in labels):
            raise ValueError(f"Global aggregated candle has no daily sessions: {path}")
        if any(len(groups[day]) != count for day, count in zip(labels, bars["trading_days"])):
            raise ValueError(f"Global aggregated candle session count mismatch: {path}")
        start = labels
        end = labels.map(lambda day: groups[day][-1])
        sessions = labels.map(lambda day: ",".join(map(str, groups[day])))
    valid = ~bars["is_partial"].astype(bool)
    values = pd.DataFrame({
        "start": start, "end": end, "sessions": sessions, "close": close,
        "previous_start": start.shift(), "previous_end": end.shift(), "previous_sessions": sessions.shift(),
        "return": close.div(close.shift()).sub(1).where(valid & valid.shift(fill_value=False)),
    })
    values = values.loc[valid].reset_index(drop=True)
    if values.duplicated(["start", "end", "sessions"]).any():
        raise ValueError(f"Duplicate global candle intervals: {path}")
    return values


def run(
    membership: Path, stocks: Path, output: Path, *, symbols: list[str] | None = None,
    factors: list[str] | None = None, timeframes: list[str] | None = None,
    minimum: int = 3, resume: bool = False, global_roots: list[Path] | None = None,
    india: Path | None = None, treasury: Path | None = None, prices: Path | None = None,
) -> dict:
    if minimum < 3:
        raise ValueError("At least three paired observations are required")
    selected_timeframes = list(TIMEFRAMES if timeframes is None else timeframes)
    if not selected_timeframes or len(set(selected_timeframes)) != len(selected_timeframes) or any(
        timeframe not in TIMEFRAMES for timeframe in selected_timeframes
    ):
        raise ValueError("Invalid timeframe selection")
    members = parse_nifty_membership(membership.read_text(encoding="utf-8-sig"))
    eligible = sorted({entry.symbol for entry in members if entry.valid_from <= COVERAGE_END
                       and (entry.valid_to is None or entry.valid_to > COVERAGE_START)})
    if symbols is not None:
        if not symbols or len(set(symbols)) != len(symbols) or not set(symbols) <= set(eligible):
            raise ValueError("Unknown or duplicate historical Nifty stock")
        eligible = sorted(symbols)
    local_names = sorted(path.name for path in (stocks / "day").iterdir()
                         if path.is_dir() and (path / "1d.csv.gz").is_file())
    if factors is not None:
        if not factors or len(set(factors)) != len(factors) or not set(factors) <= set(local_names):
            raise ValueError("Unknown or duplicate local factor")
        local_names = sorted(factors)
    if not local_names:
        raise ValueError("No local factor data found")
    if any(path is not None for path in (india, treasury, prices)) and not all(
        path is not None for path in (india, treasury, prices)
    ):
        raise ValueError("India, Treasury and independent price snapshots must be supplied together")
    external: dict[tuple[str, str], pd.DataFrame] = {}
    external_paths: list[Path] = []
    if india is not None and treasury is not None and prices is not None:
        external, external_paths = _daily_external(india, treasury, prices)
    globals_by_name: dict[tuple[str, str], Path] = {}
    global_paths: list[Path] = []
    for root in global_roots or []:
        summary_path = root / "summary.json"
        global_paths.append(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["status"] != "complete":
            raise ValueError(f"Incomplete global factor snapshot: {root}")
        source = "global:" + summary["source"]
        for item in summary["factors"]:
            key = (source, item["symbol"])
            if key in globals_by_name:
                previous = globals_by_name[key]
                if any(_source_hash(previous / "prepared" / item["symbol"] / f"{tf}.parquet")
                       != _source_hash(root / "prepared" / item["symbol"] / f"{tf}.parquet")
                       for tf in selected_timeframes):
                    raise ValueError(f"Conflicting global factor: {key}")
                continue
            globals_by_name[key] = root
            global_paths.extend(root / "prepared" / item["symbol"] / f"{tf}.parquet"
                                for tf in selected_timeframes)
            if any(tf in {"2d", "trading_week"} for tf in selected_timeframes):
                global_paths.append(root / "prepared" / item["symbol"] / "1d.parquet")
    reference_path = stocks / "day" / "RELIANCE" / "1d.csv.gz"
    reference = list(load_closes(reference_path).index)
    input_paths = {membership, reference_path, *external_paths, *global_paths}
    for symbol in eligible:
        local = _safe_symbol(local_symbol(symbol, members))
        for timeframe in selected_timeframes:
            folder = stocks / ("minute" if timeframe in INTRADAY_TIMEFRAMES else "day") / local
            input_paths.add(folder / f"{timeframe}.csv.gz")
            if timeframe in {"2d", "trading_week"}:
                input_paths.add(folder / "1d.csv.gz")
    for factor in local_names:
        for timeframe in selected_timeframes:
            folder = stocks / ("minute" if timeframe in INTRADAY_TIMEFRAMES else "day") / factor
            input_paths.add(folder / f"{timeframe}.csv.gz")
            if timeframe in {"2d", "trading_week"}:
                input_paths.add(folder / "1d.csv.gz")
    identity = {
        "version": 1, "symbols": eligible, "factors": local_names, "timeframes": selected_timeframes,
        "minimum": minimum, "global_factors": sorted([list(key) for key in globals_by_name]),
        "daily_factors": sorted([list(key) for key in external]),
        "inputs": [_fingerprint(path) for path in sorted(input_paths)],
        "code": [_fingerprint(path) for path in (
            Path(__file__), Path(__file__).with_name("nifty_preparation.py"),
            Path(__file__).with_name("nifty_universe.py"), Path(__file__).with_name("factor_research.py"),
            Path(__file__).with_name("nifty_candle_correlations.py"),
            Path(__file__).with_name("nifty_public_factor_research.py"),
            Path(__file__).with_name("nifty_treasury_factor_research.py"),
            Path(__file__).with_name("control.py"),
            Path(__file__).parent.parent / "data_pipeline" / "huggingface_stocks.py",
        )],
    }
    if output.exists():
        if not resume:
            raise FileExistsError(f"Output directory already exists: {output}")
        if json.loads((output / "identity.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("Inputs or settings changed; use a new output directory")
    else:
        if resume:
            raise FileNotFoundError(f"Output directory does not exist: {output}")
        output.mkdir(parents=True)
        _write_json(output / "identity.json", identity)
    with _price_lock(output):
        return _screen(output, stocks, eligible, members, local_names, selected_timeframes, reference,
                       globals_by_name, external, input_paths, identity, minimum, resume)


def _screen(
    output: Path, stocks: Path, eligible: list[str], members: tuple[Membership, ...], local_names: list[str],
    selected_timeframes: list[str], reference: list[date], globals_by_name: dict[tuple[str, str], Path],
    external: dict[tuple[str, str], pd.DataFrame], input_paths: set[Path], identity: dict,
    minimum: int, resume: bool,
) -> dict:
    control = TrainingControl(output)
    if resume:
        control.request("continue")
    directory = output / "stocks"
    directory.mkdir(exist_ok=True)
    fields = ["stock", "timeframe", "source", "factor", "error", "price_pairs", "price_first", "price_last", "price_status",
              "price_pearson", "price_spearman", "return_pairs", "return_first", "return_last", "return_status",
              "return_pearson", "return_spearman"]
    for index, symbol in enumerate(eligible):
        destination = directory / f"{_safe_symbol(symbol)}.csv"
        checkpoint = destination.with_suffix(".json")
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if not destination.is_file() or _source_hash(destination) != saved["sha256"]:
                raise ValueError(f"Completed stock output changed: {destination}")
            continue
        temporary = destination.with_suffix(".csv.tmp")
        local = local_symbol(symbol, members)
        intervals = tuple(entry for entry in members if entry.symbol == symbol)
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for timeframe in selected_timeframes:
                target_error = None
                try:
                    target = _read_candles(stocks, local, timeframe, reference, intervals)
                except ValueError as exc:
                    target = None
                    target_error = str(exc)
                    print(f"Invalid stock candles for {symbol} {timeframe}: {exc}", file=sys.stderr, flush=True)
                candidates = ([("local", factor) for factor in local_names]
                              + sorted(globals_by_name) + sorted(external))
                for factor_index, (source, factor) in enumerate(candidates):
                    state = {"status": "running", "stock": symbol, "timeframe": timeframe,
                             "factor": factor, "source": source,
                             "completed_stocks": index, "total_stocks": len(eligible),
                             "completed_factors_in_timeframe": factor_index,
                             "total_factors_in_timeframe": len(candidates)}
                    if factor_index % 64 == 0:
                        _write_json(output / "state.json", state)
                    if not control.checkpoint_boundary(lambda: _write_json(output / "state.json", state)):
                        _write_json(output / "state.json", {**state, "status": "stopped"})
                        raise PreparationStopped("Stopped; rerun with --resume")
                    status = None
                    error = None
                    if source == "local" and _safe_symbol(local) == factor:
                        status = "target_stock_excluded"
                    elif target_error is not None:
                        status, error = "invalid_stock_candles", target_error
                    elif target is None:
                        status = "missing_stock_candles"
                    elif (source, factor) in external:
                        if timeframe != "1d":
                            status = "missing_factor_timeframe"
                        else:
                            other = external[(source, factor)]
                    elif (source, factor) in globals_by_name:
                        try:
                            other = _global_candles(globals_by_name[(source, factor)], factor, timeframe)
                        except ValueError as exc:
                            status, error = "invalid_factor_candles", str(exc)
                        else:
                            if other is None:
                                status = "missing_factor_timeframe"
                    else:
                        try:
                            other = _read_candles(stocks, factor, timeframe, reference)
                        except ValueError as exc:
                            status, error = "invalid_factor_candles", str(exc)
                        else:
                            if other is None:
                                status = "missing_factor_candles"
                    if status == "invalid_factor_candles":
                        print(f"Invalid factor candles for {source}:{factor} {timeframe}: {error}",
                              file=sys.stderr, flush=True)
                    row = {"stock": symbol, "timeframe": timeframe, "source": source, "factor": factor,
                           "error": error}
                    if status:
                        row.update(price_pairs=0, price_status=status, return_pairs=0, return_status=status)
                    else:
                        row.update(_coefficients(target, other, minimum))
                    writer.writerow(row)
        _replace_with_retry(temporary, destination)
        _write_json(checkpoint, {"sha256": _source_hash(destination)})
        _write_json(output / "state.json", {"status": "running", "completed_stocks": index + 1,
                                            "total_stocks": len(eligible), "stock": symbol})
        print(f"Completed {index + 1}/{len(eligible)}: {symbol}", flush=True)
    if [_fingerprint(path) for path in sorted(input_paths)] != identity["inputs"]:
        raise ValueError("Correlation inputs changed during screening")
    summary = {"status": "complete", "stocks": len(eligible), "local_factors": len(local_names),
               "global_factors": len(globals_by_name), "daily_factors": len(external),
               "timeframes": selected_timeframes, "minimum_pairs": minimum,
               "limitations": [
                   "Same-period associations are retrospective, not next-candle predictive correlations.",
                   "Price-level correlations can reflect shared trends rather than predictive relationships.",
                   "Daily observations are not intraday candles; missing factor timeframes are reported.",
                   "Same-date US and Indian daily closes occur at different times and are not causal predictors.",
                   "Treasury and bond-yield changes are absolute rate changes, not percentage returns.",
                   "Incomplete candles and mismatched previous intervals are excluded.",
               ]}
    _write_json(output / "summary.json", summary)
    _write_json(output / "state.json", {"status": "complete", "completed_stocks": len(eligible),
                                        "total_stocks": len(eligible)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--stocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--factors", nargs="+")
    parser.add_argument("--timeframes", nargs="+", choices=TIMEFRAMES)
    parser.add_argument("--global-factors", type=Path, action="append", default=[])
    parser.add_argument("--india-audit", type=Path)
    parser.add_argument("--treasury-audit", type=Path)
    parser.add_argument("--prices-audit", type=Path)
    parser.add_argument("--minimum", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args.membership, args.stocks, args.output, symbols=args.symbols, factors=args.factors,
            timeframes=args.timeframes, minimum=args.minimum, resume=args.resume, global_roots=args.global_factors,
            india=args.india_audit, treasury=args.treasury_audit, prices=args.prices_audit)
    except PreparationStopped as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
