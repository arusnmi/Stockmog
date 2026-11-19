"""Recover HDFC daily candles from a checksum-pinned public NSE archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from modeling.control import TrainingControl
from modeling.factor_research import PreparationStopped, _write_json


URL = (
    "https://github.com/olo-labs/olo-ai-training-data-nse/releases/download/"
    "V1.0.0/olo-testdata-nse-V1.0.0-daily.zip"
)
ARCHIVE_SIZE = 146979775
ARCHIVE_SHA256 = "c5000730508d5961dcfb6d11c573e67862acc8a180b0de23c6ee917a675053fc"
MEMBER = "olo-testdata-nse-V1.0.0/database/h/HDFC/day.parquet"
FIRST = date(2022, 1, 1)
LAST = date(2023, 7, 12)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(frame: pd.DataFrame, reference_days: set[date]) -> tuple[pd.DataFrame, dict]:
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if not set(columns).issubset(frame.columns):
        raise ValueError("HDFC source is missing required candle columns")
    frame = frame.copy()
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamps.dt.tz is not None or timestamps.isna().any() or (timestamps != timestamps.dt.normalize()).any():
        raise ValueError("HDFC source daily timestamps must be naive local midnight")
    frame["timestamp"] = timestamps
    frame = frame.loc[(timestamps.dt.date >= FIRST) & (timestamps.dt.date <= LAST)]
    if frame.empty:
        raise ValueError("HDFC source contains no candles in the requested period")
    duplicates = len(frame) - len(frame.drop_duplicates())
    frame = frame.drop_duplicates()
    if frame["timestamp"].duplicated().any():
        raise ValueError("HDFC source has conflicting duplicate daily records")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    values = frame[columns[1:]]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("HDFC source has non-finite candle values")
    if (
        (frame[["open", "high", "low", "close"]] <= 0).any().any()
        or (frame["volume"] < 0).any()
        or (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any()
        or (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any()
    ):
        raise ValueError("HDFC source has invalid OHLCV values")
    days = set(frame["timestamp"].dt.date)
    missing = sorted(reference_days - days)
    extra = sorted(days - reference_days)
    frame = frame[columns].copy()
    frame["timestamp"] = frame["timestamp"].dt.tz_localize("Asia/Kolkata").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    frame["symbol"] = "HDFC"
    frame["source_bars"] = 1
    frame["trading_days"] = 1
    frame["is_partial"] = False
    report = {
        "first": min(days).isoformat(), "last": max(days).isoformat(), "sessions": len(days),
        "identical_rows_removed": duplicates,
        "missing_reference_sessions": [day.isoformat() for day in missing],
        "extra_sessions": [day.isoformat() for day in extra],
    }
    return frame, report


def download_archive(path: Path, control: TrainingControl) -> None:
    if path.is_file():
        if path.stat().st_size != ARCHIVE_SIZE or sha256_file(path) != ARCHIVE_SHA256:
            raise ValueError(f"Existing archive does not match the pinned release: {path}")
        return
    partial = path.with_suffix(".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > ARCHIVE_SIZE:
        raise ValueError(f"Partial download is larger than the pinned archive: {partial}")
    if offset < ARCHIVE_SIZE:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with urlopen(Request(URL, headers=headers), timeout=120) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("Archive server did not honor the resume byte range")
            with partial.open("ab") as destination:
                while True:
                    if not control.checkpoint_boundary(destination.flush):
                        raise PreparationStopped("HDFC download stopped at a completed block")
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    if destination.tell() > ARCHIVE_SIZE:
                        raise ValueError("Downloaded archive exceeds the pinned size")
    if partial.stat().st_size != ARCHIVE_SIZE or sha256_file(partial) != ARCHIVE_SHA256:
        raise ValueError(f"Downloaded archive does not match the pinned checksum: {partial}")
    partial.replace(path)


def collect(reference_daily: Path, output: Path, stock_daily: Path, *, resume: bool = False) -> dict:
    reference_daily = reference_daily.resolve()
    stock_daily = stock_daily.resolve()
    identity = {
        "archive_url": URL, "archive_sha256": ARCHIVE_SHA256, "archive_size": ARCHIVE_SIZE,
        "archive_member": MEMBER, "reference": str(reference_daily),
        "reference_sha256": sha256_file(reference_daily), "destination": str(stock_daily),
    }
    manifest = output / "identity.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Run directory already exists: {output}")
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("HDFC daily inputs changed or run identity is missing")
    else:
        if resume:
            raise FileNotFoundError(f"Run directory does not exist: {output}")
        output.mkdir(parents=True)
        _write_json(manifest, identity)
    control = TrainingControl(output)
    if resume:
        control.request("continue")
    archive = output / "source.zip"
    download_archive(archive, control)
    if not control.checkpoint_boundary(lambda: None):
        raise PreparationStopped("HDFC preparation stopped before extraction")
    with zipfile.ZipFile(archive) as source:
        raw = source.read(MEMBER)
    source_sha = hashlib.sha256(raw).hexdigest()
    frame = pd.read_parquet(io.BytesIO(raw))
    reference = pd.read_csv(reference_daily, usecols=["timestamp"])
    days = pd.to_datetime(reference["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    reference_days = {day for day in days if FIRST <= day <= LAST}
    if not reference_days:
        raise ValueError("Reference calendar has no sessions in the requested period")
    candles, report = normalize(frame, reference_days)
    if report["missing_reference_sessions"]:
        raise ValueError(f"HDFC source misses reference sessions: {report['missing_reference_sessions']}")
    summary_path = output / "summary.json"
    staged_path = output / "staged.json"
    if stock_daily.exists():
        provenance = summary_path if summary_path.is_file() else staged_path
        if not provenance.is_file():
            raise FileExistsError(f"Destination already exists without this run's provenance: {stock_daily}")
        existing = json.loads(provenance.read_text(encoding="utf-8"))
        if existing["destination_sha256"] != sha256_file(stock_daily):
            raise ValueError(f"Existing HDFC destination has changed: {stock_daily}")
        if not summary_path.is_file():
            _write_json(summary_path, existing)
        return existing
    stock_daily.parent.mkdir(parents=True, exist_ok=True)
    temporary = output / "1d.csv.gz.tmp"
    candles.to_csv(temporary, index=False, compression="gzip")
    output_sha = sha256_file(temporary)
    report.update({
        "source": URL, "archive_sha256": ARCHIVE_SHA256, "archive_member": MEMBER,
        "source_member_sha256": source_sha, "reference_calendar": str(reference_daily),
        "destination": str(stock_daily), "destination_sha256": output_sha,
        "availability": "Daily source only; use following midnight IST for conservative availability.",
        "rights": "Project code is Apache-2.0; upstream market-data rights require independent verification.",
    })
    _write_json(staged_path, report)
    os.link(temporary, stock_daily)
    temporary.unlink()
    _write_json(summary_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-daily", type=Path, default=Path("data") / "stocks" / "day" / "RELIANCE" / "1d.csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("data") / "stocks" / "day" / "HDFC" / "1d.csv.gz")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        report = collect(args.reference_daily, args.output, args.destination, resume=args.resume)
        print(f"Recovered {report['sessions']} daily HDFC sessions: {args.destination}")
    except PreparationStopped as exc:
        print(str(exc))


if __name__ == "__main__":
    main()
