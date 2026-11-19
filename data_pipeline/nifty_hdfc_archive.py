"""Collect the available, unmodified HDFC archive files for research."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd

from modeling.control import TrainingControl


REVISION = "c95d7f187c2a317ca6a2287148cf10c50a80e928"
BASE_URL = f"https://raw.githubusercontent.com/gsidhu/nse-intraday-data/{REVISION}"
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def archive_url(day: date) -> str:
    month = MONTHS[day.month - 1]
    directory = f"{day.day:02d}{month}"
    return f"{BASE_URL}/{day.year}/{month}/{directory}/{directory}/HDFC.txt"


def validate_archive(content: bytes, day: date) -> int:
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    if not rows:
        raise ValueError(f"Empty HDFC archive file for {day}")
    for row in rows:
        if len(row) != 9 or row[0] != "HDFC" or row[1] != day.strftime("%Y%m%d"):
            raise ValueError(f"Unexpected HDFC archive row for {day}: {row!r}")
        pd.to_datetime(row[2], format="%H:%M", errors="raise")
        for value in row[3:]:
            float(value)
    return len(rows)


def collect(reference_daily: Path, run_dir: Path, *, resume: bool = False) -> dict:
    reference_daily = reference_daily.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "revision": REVISION,
        "reference_daily": str(reference_daily),
        "reference_size": reference_daily.stat().st_size,
        "reference_mtime_ns": reference_daily.stat().st_mtime_ns,
        "first_date": "2022-01-01",
        "last_date": "2023-07-12",
    }
    identity_path = run_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("HDFC archive run inputs changed; use a new run directory")
    if not identity_path.exists():
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")

    daily = pd.read_csv(reference_daily, usecols=["timestamp"])
    days = pd.to_datetime(daily["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    days = sorted({day for day in days if date(2022, 1, 1) <= day <= date(2023, 7, 12)})
    if not days:
        raise ValueError("Reference daily file has no sessions in the requested period")
    control = TrainingControl(run_dir)
    if resume:
        control.request("continue")
    files_dir = run_dir / "raw"
    files_dir.mkdir(exist_ok=True)
    result = []
    for day in days:
        if not control.checkpoint_boundary(lambda: None):
            return {"status": "stopped", "sessions_processed": len(result)}
        if day > date(2023, 2, 28):
            result.append({"date": day.isoformat(), "status": "not_in_archive"})
            continue
        path = files_dir / f"{day.isoformat()}.txt"
        missing_path = files_dir / f"{day.isoformat()}.missing"
        invalid_path = files_dir / f"{day.isoformat()}.invalid.txt"
        if path.exists():
            content = path.read_bytes()
        elif missing_path.exists():
            result.append({"date": day.isoformat(), "status": "not_found"})
            continue
        elif invalid_path.exists():
            result.append({"date": day.isoformat(), "status": "invalid_source", "path": str(invalid_path)})
            continue
        else:
            try:
                with urlopen(archive_url(day), timeout=30) as response:
                    content = response.read()
            except HTTPError as exc:
                if exc.code != 404:
                    raise
                missing_path.write_text(archive_url(day) + "\n")
                result.append({"date": day.isoformat(), "status": "not_found"})
                continue
            try:
                validate_archive(content, day)
            except ValueError as exc:
                invalid_path.write_bytes(content)
                print(f"Rejecting invalid archive file for {day}: {exc}", flush=True)
                result.append({"date": day.isoformat(), "status": "invalid_source", "path": str(invalid_path)})
                continue
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        count = validate_archive(content, day)
        result.append({
            "date": day.isoformat(), "status": "downloaded", "rows": count,
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    summary = {
        "status": "partial",
        "source": BASE_URL,
        "usage": "README states MIT; no separate LICENSE file was found. Verify data rights before redistribution.",
        "timestamp_semantics": "Unverified; raw files are not yet safe to merge into the training dataset.",
        "reference_calendar": str(reference_daily),
        "sessions": result,
    }
    temporary = run_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(run_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-daily", type=Path, default=Path("data") / "stocks" / "day" / "RELIANCE" / "1d.csv.gz")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = collect(args.reference_daily, args.run_dir, resume=args.resume)
    print(f"Status: {summary['status']}; run directory: {args.run_dir}")


if __name__ == "__main__":
    main()
