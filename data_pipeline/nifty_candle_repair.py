"""Audit Nifty candle coverage and recover only bars backed by complete finer data."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from data_pipeline.huggingface_stocks import INTRADAY_TIMEFRAMES, _safe_symbol, resample_intraday
from modeling.control import _replace_with_retry
from modeling.nifty_preparation import TIMEFRAMES, daily_source_aggregates
from modeling.nifty_timeframe_data import training_universe
from modeling.nifty_universe import local_symbol, parse_nifty_membership


CORE = ["timestamp", "open", "high", "low", "close", "volume",
        "source_bars", "trading_days", "is_partial"]


def read_candles(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    data = pd.read_csv(path, usecols=CORE, compression="gzip")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    if data["timestamp"].isna().any() or data["timestamp"].duplicated().any():
        raise ValueError(f"Missing or duplicate candle timestamps: {path}")
    if data["is_partial"].isna().any() or not data["is_partial"].isin([True, False]).all():
        raise ValueError(f"Invalid candle completeness flag: {path}")
    prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices <= 0).any()
            or (data["high"] < data[["open", "close", "low"]].max(axis=1)).any()
            or (data["low"] > data[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError(f"Invalid OHLC candles: {path}")
    if (not np.isfinite(data["volume"]).all() or (data["volume"] < 0).any()):
        raise ValueError(f"Invalid candle volume: {path}")
    return data.sort_values("timestamp").reset_index(drop=True)


def session_days(data: pd.DataFrame) -> list:
    return data["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date.tolist()


def expected_labels(reference: list, days: list, timeframe: str) -> pd.DatetimeIndex:
    available = [day for day in reference if days[0] <= day <= days[-1]]
    if timeframe in INTRADAY_TIMEFRAMES:
        width = INTRADAY_TIMEFRAMES[timeframe]
        midnight = pd.DatetimeIndex(available).tz_localize("Asia/Kolkata")
        labels = (midnight.to_numpy()[:, None]
                  + np.array([timedelta(minutes=555 + minute)
                              for minute in range(0, 375, width)], dtype=object)[None, :])
        return pd.DatetimeIndex(labels.ravel()).tz_convert("UTC")
    if timeframe == "2d":
        available = available[::2]
    elif timeframe == "trading_week":
        available = sorted({day - timedelta(days=day.weekday()) for day in available})
    return pd.DatetimeIndex(available).tz_localize("Asia/Kolkata").tz_convert("UTC")


def derived_candidates(minutes: pd.DataFrame | None, daily: pd.DataFrame | None,
                       reference: list, timeframe: str, symbol: str) -> pd.DataFrame:
    if timeframe == "1m":
        return pd.DataFrame(columns=CORE)
    if timeframe in INTRADAY_TIMEFRAMES or timeframe == "1d":
        if minutes is None or minutes.empty:
            return pd.DataFrame(columns=CORE)
        source = minutes.loc[~minutes["is_partial"]].copy()
        # A complete one-minute source must really represent exactly one exchange minute.
        local = source["timestamp"].dt.tz_convert("Asia/Kolkata")
        valid = ((source["source_bars"] == 1) & (local.dt.second == 0)
                 & (local.dt.microsecond == 0) & (local.dt.nanosecond == 0)
                 & (local.dt.dayofweek < 5))
        if not valid.all():
            raise ValueError(f"Invalid complete one-minute source bars: {symbol}")
        source["oi"] = 0.
        if timeframe in INTRADAY_TIMEFRAMES:
            bars = resample_intraday(pl.from_pandas(source),
                                     minutes=INTRADAY_TIMEFRAMES[timeframe], symbol=symbol).to_pandas()
        else:
            from data_pipeline.huggingface_stocks import resample_daily

            bars = resample_daily(pl.from_pandas(source), symbol=symbol).to_pandas()
            days = bars["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
            bars["timestamp"] = pd.to_datetime(days.astype(str)).dt.tz_localize(
                "Asia/Kolkata").dt.tz_convert("UTC")
        return bars.loc[~bars["is_partial"], CORE].reset_index(drop=True)
    if daily is None or daily.empty:
        return pd.DataFrame(columns=CORE)
    bars = daily_source_aggregates(daily, timeframe)
    days = pd.Series(session_days(daily), index=daily.index)
    good = ~daily["is_partial"]
    reference_index = {day: index for index, day in enumerate(reference)}
    if timeframe == "2d":
        pairs = [days.iloc[index:index + 2].tolist() for index in range(0, len(days), 2)]
        valid = {pair[0] for index, pair in enumerate(pairs)
                 if len(pair) == 2 and good.iloc[index * 2:index * 2 + 2].all()
                 and pair[0] in reference_index and pair[1] in reference_index
                 and reference_index[pair[1]] == reference_index[pair[0]] + 1}
        bars = bars.loc[bars["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date.isin(valid)]
    else:
        actual = {}
        expected = {}
        for day in reference:
            expected.setdefault(day - timedelta(days=day.weekday()), []).append(day)
        for day, complete in zip(days, good):
            monday = day - timedelta(days=day.weekday())
            actual.setdefault(monday, []).append((day, bool(complete)))
        valid = {monday for monday, group in actual.items()
                 if [day for day, _ in group] == expected.get(monday)
                 and all(complete for _, complete in group)
                 and monday >= days.iloc[0] and monday + timedelta(days=4) <= days.iloc[-1]}
        labels = bars["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date.map(
            lambda day: day - timedelta(days=day.weekday()))
        bars = bars.loc[labels.isin(valid)].copy()
        bars["timestamp"] = pd.to_datetime(labels.loc[bars.index].astype(str)).dt.tz_localize(
            "Asia/Kolkata").dt.tz_convert("UTC")
        bars["is_partial"] = False
    return bars.loc[~bars["is_partial"], CORE].reset_index(drop=True)


def recoverable_rows(existing: pd.DataFrame | None, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    if existing is None:
        return candidates
    complete_labels = existing.loc[~existing["is_partial"], "timestamp"]
    return candidates.loc[~candidates["timestamp"].isin(complete_labels)].copy()


def recover_from_intervals(sources: dict[str, pd.DataFrame | None],
                           labels: pd.DatetimeIndex, timeframe: str) -> pd.DataFrame:
    """Tile a missing bar with exact, nonoverlapping finer bars; never split OHLC."""
    width = INTRADAY_TIMEFRAMES.get(timeframe, 375)
    if timeframe == "1m" or labels.empty:
        return pd.DataFrame(columns=CORE)
    wanted_days = set(labels.tz_convert("Asia/Kolkata").date)
    edges = {}
    minute_ns = 60_000_000_000
    for source_frame, data in sources.items():
        size = INTRADAY_TIMEFRAMES[source_frame]
        if size >= width or data is None or data.empty:
            continue
        local = data["timestamp"].dt.tz_convert("Asia/Kolkata")
        offset = local.dt.hour * 60 + local.dt.minute - 555
        valid = (~data["is_partial"] & (data["source_bars"] == size)
                 & local.dt.date.isin(wanted_days) & (offset >= 0)
                 & (offset + size <= 375) & (offset % size == 0)
                 & (local.dt.second == 0) & (local.dt.microsecond == 0)
                 & (local.dt.nanosecond == 0))
        for row in data.loc[valid, CORE].to_dict("records"):
            start = row["timestamp"].value
            edges.setdefault(start, []).append((start + size * minute_ns, row))
    result = []
    for label in labels:
        start = label.value + (555 * minute_ns if timeframe == "1d" else 0)
        end = start + width * minute_ns
        # A session-end short bar is intentionally partial, not missing full data.
        local_start = pd.Timestamp(start, tz="UTC").tz_convert("Asia/Kolkata")
        if local_start.hour * 60 + local_start.minute + width > 930:
            continue
        paths = {end: None}
        for position in range(end - minute_ns, start - minute_ns, -minute_ns):
            for next_position, row in sorted(edges.get(position, []), key=lambda edge: edge[0]):
                if next_position in paths:
                    paths[position] = (next_position, row)
                    break
        if start not in paths:
            continue
        rows = []
        position = start
        while position != end:
            position, row = paths[position]
            rows.append(row)
        result.append({
            "timestamp": label, "open": rows[0]["open"],
            "high": max(row["high"] for row in rows), "low": min(row["low"] for row in rows),
            "close": rows[-1]["close"], "volume": sum(row["volume"] for row in rows),
            "source_bars": width, "trading_days": 1, "is_partial": False,
        })
    return pd.DataFrame(result, columns=CORE)


def save_repair(path: Path, recovered: pd.DataFrame, symbol: str) -> None:
    if recovered.empty:
        return
    if path.exists():
        original = pd.read_csv(path)
        original["timestamp"] = pd.to_datetime(original["timestamp"], utc=True, errors="raise")
        original = original.loc[~original["timestamp"].isin(recovered["timestamp"])]
        if "symbol" in original:
            recovered = recovered.assign(symbol=symbol)
        # Existing derived indicator columns remain blank on repaired rows: training recomputes them.
        result = pd.concat([original, recovered], ignore_index=True).sort_values("timestamp")
    else:
        result = recovered.assign(symbol=symbol).sort_values("timestamp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".repair.tmp")
    result.to_csv(temporary, index=False, compression="gzip")
    checked = read_candles(temporary)
    if checked is None or len(checked) != len(result):
        raise ValueError(f"Repair failed verification: {path}")
    _replace_with_retry(temporary, path)


def run(stocks: Path, membership: Path, report: Path, *, apply: bool = False) -> pd.DataFrame:
    members = parse_nifty_membership(membership.read_text(encoding="utf-8-sig"))
    ref = read_candles(stocks / "day" / "RELIANCE" / "1d.csv.gz")
    if ref is None:
        raise FileNotFoundError("RELIANCE daily reference calendar is missing")
    reference = session_days(ref)
    records = []
    for symbol in training_universe(members):
        local = _safe_symbol(local_symbol(symbol, members))
        daily = read_candles(stocks / "day" / local / "1d.csv.gz")
        minutes = read_candles(stocks / "minute" / local / "1m.csv.gz")
        intraday = {frame: minutes if frame == "1m" else read_candles(
            stocks / "minute" / local / f"{frame}.csv.gz") for frame in INTRADAY_TIMEFRAMES}
        bounds = daily if daily is not None and not daily.empty else minutes
        if bounds is None or bounds.empty:
            raise ValueError(f"No source history for {symbol}")
        days = session_days(bounds)
        for timeframe in TIMEFRAMES:
            group = "minute" if timeframe in INTRADAY_TIMEFRAMES else "day"
            path = stocks / group / local / f"{timeframe}.csv.gz"
            existing = (minutes if timeframe == "1m" else daily if timeframe == "1d"
                        else intraday[timeframe] if timeframe in intraday else read_candles(path))
            expected = expected_labels(reference, days, timeframe)
            existing_labels = pd.DatetimeIndex([]) if existing is None else pd.DatetimeIndex(existing["timestamp"])
            missing = len(expected.difference(existing_labels))
            partial = 0 if existing is None else int(existing["is_partial"].sum())
            candidates = derived_candidates(minutes, daily, reference, timeframe, symbol)
            recovered = recoverable_rows(existing, candidates)
            if timeframe in INTRADAY_TIMEFRAMES or timeframe == "1d":
                complete = ([] if existing is None else
                            existing.loc[~existing["is_partial"], "timestamp"].tolist())
                missing_labels = expected.difference(pd.DatetimeIndex(complete)).difference(
                    pd.DatetimeIndex(recovered["timestamp"]))
                alternate = recover_from_intervals(intraday, missing_labels, timeframe)
                if not alternate.empty:
                    recovered = (alternate if recovered.empty else
                                 pd.concat([recovered, alternate], ignore_index=True))
            replaced_partial = (0 if existing is None else int(existing.loc[
                existing["is_partial"], "timestamp"].isin(recovered["timestamp"]).sum()))
            if apply:
                save_repair(path, recovered, symbol)
            if timeframe == "1d" and not recovered.empty:
                # Let the same audit discover multi-session repairs from newly recovered days.
                daily = pd.concat([
                    daily.loc[~daily["timestamp"].isin(recovered["timestamp"])]
                    if daily is not None else pd.DataFrame(columns=CORE), recovered,
                ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            records.append({
                "stock": symbol, "timeframe": timeframe, "file_missing_before": existing is None,
                "rows_before": 0 if existing is None else len(existing),
                "missing_reference_labels_before": missing, "partial_rows_before": partial,
                "recoverable_rows": len(recovered), "replaced_partial_rows": replaced_partial,
                "recovered_rows": len(recovered) if apply else 0,
                "missing_reference_labels_after": len(expected.difference(
                    existing_labels.union(pd.DatetimeIndex(recovered["timestamp"]))))
                if apply else missing,
                "partial_rows_after": partial - replaced_partial if apply else partial,
                "source": "not_recoverable_from_coarser_candles" if timeframe == "1m"
                else "complete_finer_intraday" if timeframe in INTRADAY_TIMEFRAMES
                or timeframe == "1d" else "1d",
            })
        print(f"Checked {symbol}: {sum(r['recoverable_rows'] for r in records[-len(TIMEFRAMES):])} "
              "recoverable candles", flush=True)
    result = pd.DataFrame(records)
    report.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(report, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks", type=Path, default=Path("data") / "stocks")
    parser.add_argument("--membership", type=Path, default=Path("data") / "external" /
                        "nse_index_membership_0786cd915974.csv")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = run(args.stocks, args.membership, args.report, apply=args.apply)
    print(result.groupby("timeframe")[["missing_reference_labels_before", "recoverable_rows",
                                       "recovered_rows"]].sum().to_string())


if __name__ == "__main__":
    main()
