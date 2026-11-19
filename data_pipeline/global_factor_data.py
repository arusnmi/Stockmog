"""Download and causally prepare diversified US-listed market proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped, _write_json
from modeling.nifty_preparation import TIMEFRAMES


REVISION = "eff0dac55d843b3b83caec819efa213021a3e3e5"
REPOSITORY = "syncster-data/historical-market-data"
API = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}?blobs=true"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
START = date(2021, 12, 29)
END = date(2026, 1, 21)
MAX_SOURCE_ROWS = 5_000_000
GROUPS = {
    "us_sector": "XLE XLF XLK XLI XLU XLV XLB XLY XLP XLRE XLC",
    "broad_market": "SPY QQQ DIA IWM VTI VWO EFA EEM ACWI",
    "country": "INDA EWA EWC EWJ EWH EWZ EWG EWU EWT EWY EWS EWI EWP EWL EWN EWM EWW EZA FXI",
    "commodity_proxy": "GLD SLV USO UNG DBA DBC CPER PALL PPLT",
    "industry": "SMH SOXX KRE KBE IBB XBI IHI ITA XOP OIH XME XRT XHB IYR VNQ XTN XTL XPH XAR XSD XSW TAN ICLN JETS GDX GDXJ COPX LIT REM",
    "fixed_income": "TLT IEF SHY TIP LQD HYG EMB BND AGG",
    "currency_proxy": "UUP FXE FXY FXB FXA FXC",
    "investment_style": "USMV MTUM QUAL VTV VUG VIG VYM RSP",
}
SECTOR_GROUPS = {
    "calendar": "SPY",
    "sector": "VGT VHT VFH VDE VIS VDC VCR VAW VPU VOX",
    "industry": "KBWB KIE XHE XHS IEO IEZ IYT IYW IGV IHF ITB PEJ GUNR WOOD PICK PAVE ARKG MOO PBJ KWEB FINX FDN IYM IYK IYC IXG",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boundary(control: TrainingControl, save=lambda: None) -> None:
    if not control.checkpoint_boundary(save):
        raise PreparationStopped("Global factor preparation stopped at a checkpoint")


def source_manifest(groups: dict[str, str] = GROUPS) -> dict:
    with urlopen(API, timeout=120) as response:
        payload = json.load(response)
    if payload["sha"] != REVISION:
        raise ValueError("Global factor dataset revision does not match the pinned revision")
    files = {item["rfilename"]: item for item in payload["siblings"]}
    selected = []
    seen = set()
    for group, names in groups.items():
        for symbol in names.split():
            if symbol in seen:
                raise ValueError(f"Duplicate global factor: {symbol}")
            seen.add(symbol)
            path = f"data/stocks/1m/{symbol[0]}/{symbol}.parquet"
            item = files.get(path)
            if item is None or "lfs" not in item:
                raise ValueError(f"Missing pinned LFS data for {symbol}")
            selected.append({"symbol": symbol, "group": group, "path": path,
                             "size": item["size"], "sha256": item["lfs"]["sha256"]})
    return {"version": 1, "repository": REPOSITORY, "revision": REVISION,
            "first_us_session": START.isoformat(), "last_us_session": END.isoformat(),
            "max_source_rows": MAX_SOURCE_ROWS, "factors": selected}


def download(item: dict, directory: Path, control: TrainingControl) -> Path:
    path = directory / f"{item['symbol']}.parquet"
    if path.exists():
        if path.stat().st_size != item["size"] or _sha(path) != item["sha256"]:
            raise ValueError(f"Downloaded factor has changed: {path}")
        return path
    partial = directory / f"{item['symbol']}.partial"
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item["size"]:
        raise ValueError(f"Partial download is too large: {partial}")
    if offset < item["size"]:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with urlopen(Request(f"{BASE_URL}/{item['path']}", headers=headers), timeout=120) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError(f"Server did not honor resume range for {item['symbol']}")
            with partial.open("ab") as destination:
                while True:
                    _boundary(control, destination.flush)
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    if destination.tell() > item["size"]:
                        raise ValueError(f"Downloaded factor exceeds expected size: {partial}")
    if partial.stat().st_size != item["size"] or _sha(partial) != item["sha256"]:
        raise ValueError(f"Global factor checksum mismatch: {partial}")
    partial.replace(path)
    return path


def regular_bars(frame: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    required = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{symbol} source is missing columns")
    if not isinstance(frame["ts"].dtype, pd.DatetimeTZDtype) or frame["ts"].isna().any():
        raise ValueError(f"{symbol} source timestamps must be timezone-aware")
    frame = frame.loc[:, sorted(required)].copy()
    frame["ts"] = frame["ts"].dt.tz_convert("UTC")
    if (frame["symbol"] != symbol).any() or frame["ts"].duplicated().any():
        raise ValueError(f"{symbol} source has wrong symbols or duplicate timestamps")
    frame = frame.sort_values("ts").reset_index(drop=True)
    local = frame["ts"].dt.tz_convert("America/New_York")
    days = local.dt.date
    minute = local.dt.hour * 60 + local.dt.minute
    requested = (days >= START) & (days <= END)
    regular = (local.dt.weekday < 5) & (minute >= 570) & (minute < 960)
    outside = int((requested & ~regular).sum())
    frame = frame.loc[requested & regular].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{symbol} has no regular-session data in the requested period")
    if (frame["ts"].dt.second != 0).any() or (frame["ts"].dt.microsecond != 0).any():
        raise ValueError(f"{symbol} contains timestamps not aligned to whole minutes")
    numeric = ["open", "high", "low", "close", "volume"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError(f"{symbol} contains non-finite prices or volume")
    if ((frame[["open", "high", "low", "close"]] <= 0).any().any()
            or (frame["volume"] < 0).any()
            or (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any()
            or (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError(f"{symbol} has invalid OHLCV data")
    frame = frame.drop(columns="symbol").rename(columns={"ts": "timestamp"})
    report = {"symbol": symbol, "first": str(frame["timestamp"].iloc[0]),
              "last": str(frame["timestamp"].iloc[-1]), "regular_minute_rows": len(frame),
              "excluded_extended_hours_rows": outside}
    return frame, report


def _groups(frame: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    values = frame.assign(_group=keys.to_numpy())
    result = values.groupby("_group", sort=True, as_index=False).agg(
        timestamp=("timestamp", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        source_bars=("source_bars", "sum"), trading_days=("trading_days", "sum"),
        is_partial=("is_partial", "any"),
    )
    return result.drop(columns="_group")


def _ny_midnight(days: pd.Series) -> pd.Series:
    return pd.to_datetime(days.astype(str)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")


def make_timeframe(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    data = bars.copy()
    data["source_bars"] = 1
    data["trading_days"] = 1
    data["is_partial"] = False
    widths = {"1m": 1, "5m": 5, "15m": 15, "20m": 20, "30m": 30, "1h": 60}
    if timeframe in widths:
        width = widths[timeframe]
        if width == 1:
            data["available_at"] = data["timestamp"] + timedelta(minutes=1)
            return data
        bucket = data["timestamp"].dt.floor(f"{width}min")
        result = _groups(data, bucket)
        result["timestamp"] = bucket.drop_duplicates().reset_index(drop=True)
        result["available_at"] = result["timestamp"] + timedelta(minutes=width)
        result["trading_days"] = 1
        result["is_partial"] = result["source_bars"] < width
        return result
    trading_date = data["timestamp"].dt.tz_convert("America/New_York").dt.date
    daily = _groups(data, trading_date)
    daily["trading_days"] = 1
    daily["is_partial"] = daily["source_bars"] < 390
    daily_days = pd.Series(sorted(trading_date.unique()))
    if timeframe == "1d":
        daily["available_at"] = _ny_midnight(daily_days + timedelta(days=1))
        return daily
    if timeframe == "2d":
        groups = pd.Series(np.arange(len(daily)) // 2)
        result = _groups(daily, groups)
        result = result.loc[result["trading_days"] == 2].reset_index(drop=True)
        last_days = daily_days.iloc[1::2].reset_index(drop=True)
        result["available_at"] = _ny_midnight(last_days + timedelta(days=1))
        return result
    monday = daily_days.map(lambda day: day - timedelta(days=day.weekday()))
    result = _groups(daily, monday)
    result["is_partial"] = result["is_partial"] | (result["trading_days"] < 5)
    next_mondays = pd.Series(sorted(monday.unique())) + timedelta(days=7)
    result["available_at"] = _ny_midnight(next_mondays)
    return result


def prepare_factor(item: dict, path: Path, output: Path, control: TrainingControl) -> dict:
    folder = output / item["symbol"]
    folder.mkdir(parents=True, exist_ok=True)
    checkpoints = {tf: folder / f"{tf}.json" for tf in TIMEFRAMES}
    for tf, checkpoint in checkpoints.items():
        if checkpoint.is_file():
            meta = json.loads(checkpoint.read_text(encoding="utf-8"))
            target = folder / f"{tf}.parquet"
            if not target.is_file() or _sha(target) != meta["sha256"] or meta["source_sha256"] != item["sha256"]:
                raise ValueError(f"Global factor checkpoint is corrupt: {checkpoint}")
    report_path = folder / "coverage.json"
    if all(path.is_file() for path in checkpoints.values()) and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows > MAX_SOURCE_ROWS:
        raise ValueError(f"{item['symbol']} exceeds the bounded source row limit")
    _boundary(control)
    bars, report = regular_bars(parquet.read().to_pandas(), item["symbol"])
    report.update(group=item["group"], source_path=item["path"], source_sha256=item["sha256"])
    for timeframe, checkpoint in checkpoints.items():
        _boundary(control)
        if checkpoint.is_file():
            continue
        prepared = make_timeframe(bars, timeframe)
        target = folder / f"{timeframe}.parquet"
        temporary = folder / f"{timeframe}.tmp"
        prepared.to_parquet(temporary, index=False)
        temporary.replace(target)
        _write_json(checkpoint, {"rows": len(prepared), "sha256": _sha(target), "source_sha256": item["sha256"]})
    report["timeframes"] = {tf: json.loads(checkpoint.read_text(encoding="utf-8"))["rows"]
                            for tf, checkpoint in checkpoints.items()}
    _write_json(report_path, report)
    return report


def run(output: Path, *, resume: bool = False, sector_candidates: bool = False) -> dict:
    identity = source_manifest(SECTOR_GROUPS if sector_candidates else GROUPS)
    manifest_path = output / "identity.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Global factor output already exists: {output}")
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Global factor inputs or settings changed; use a new directory")
    else:
        if resume:
            raise FileNotFoundError(f"Global factor output does not exist: {output}")
        output.mkdir(parents=True)
        _write_json(manifest_path, identity)
    raw = output / "raw"
    prepared = output / "prepared"
    raw.mkdir(exist_ok=True)
    prepared.mkdir(exist_ok=True)
    control = TrainingControl(output)
    if resume:
        control.request("continue")
    results = []
    for item in identity["factors"]:
        _boundary(control)
        source = download(item, raw, control)
        try:
            result = prepare_factor(item, source, prepared, control)
        except ValueError as exc:
            print(f"Invalid source {item['symbol']}: {exc}", flush=True)
            result = {"symbol": item["symbol"], "group": item["group"], "status": "invalid_source", "error": str(exc)}
        results.append(result)
        print(f"Processed {len(results)}/{len(identity['factors'])}: {item['symbol']}", flush=True)
    summary = {"status": "complete_with_invalid_sources" if any(r.get("status") == "invalid_source" for r in results) else "complete",
               "source": REPOSITORY, "revision": REVISION,
               "rights": "Dataset card specifies research and education only; license is other. Verify rights before other use or redistribution.",
               "adjustment": "Corporate-action adjustment and vendor publication latency are not established; not yet approved for training.",
               "notes": "US-listed proxies can overlap, are not independent and may not trade during NSE hours. No missing candles are filled.",
               "factors": results}
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sector-candidates", action="store_true",
                        help="Collect additional sector/industry proxies and SPY calendar")
    args = parser.parse_args()
    try:
        result = run(args.output, resume=args.resume, sector_candidates=args.sector_candidates)
        print(f"Global factor data {result['status']}: {args.output}")
    except PreparationStopped as exc:
        print(str(exc))


if __name__ == "__main__":
    main()
