"""Explore US Treasury rate factors against next-session Nifty stock returns."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import PreparationStopped, _fingerprint, _write_json, load_closes
from modeling.global_factor_research import correlate, next_session_targets
from modeling.nifty_price_training import _price_lock
from modeling.nifty_universe import COVERAGE_END, COVERAGE_START, local_symbol, parse_nifty_membership


BASE = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
TABLES = {
    "nominal": "daily_treasury_yield_curve",
    "real": "daily_treasury_real_yield_curve",
    "bill": "daily_treasury_bill_rates",
}


def rate_factors(frames: list[pd.DataFrame], name: str) -> dict[str, pd.DataFrame]:
    frame = pd.concat(frames, ignore_index=True)
    if "Date" not in frame or len(frame.columns) < 2:
        raise ValueError(f"Treasury {name} CSV is missing rate columns")
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y", errors="raise")
    if frame["Date"].isna().any() or frame["Date"].duplicated().any():
        raise ValueError(f"Treasury {name} has missing or duplicate dates")
    frame = frame.sort_values("Date").reset_index(drop=True)
    available = (frame["Date"] + timedelta(days=1)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    factors = {}
    for column in frame.columns.drop("Date"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values.dropna()).all():
            raise ValueError(f"Treasury {name} has non-finite {column} rates")
        for transform, data in (("level", values), ("change", values.diff())):
            mask = data.notna()
            factors[f"{name}:{column}:{transform}"] = pd.DataFrame({
                "factor_available_at": available[mask].reset_index(drop=True),
                "factor_return": data[mask].reset_index(drop=True),
            })
    return factors


def run(membership: Path, stocks_root: Path, output: Path, *, train_end: date,
        start_year: int = 2022, minimum_observations: int = 60,
        max_age_days: int = 5, resume: bool = False) -> dict:
    if (not COVERAGE_START < train_end <= COVERAGE_END or start_year < 1990
            or start_year > train_end.year or minimum_observations < 3 or max_age_days < 1):
        raise ValueError("Invalid Treasury audit dates or settings")
    membership, stocks_root, output = membership.resolve(), stocks_root.resolve(), output.resolve()
    members = parse_nifty_membership(membership.read_text(encoding="utf-8-sig"))
    symbols = sorted({item.symbol for item in members if item.valid_from < train_end
                      and (item.valid_to is None or item.valid_to > COVERAGE_START)})
    reference = stocks_root / "day" / "RELIANCE" / "1d.csv.gz"
    stock_paths = {symbol: stocks_root / "day" / _safe_symbol(local_symbol(symbol, members)) / "1d.csv.gz"
                   for symbol in symbols}
    code = [Path(__file__), Path(__file__).with_name("factor_research.py"),
            Path(__file__).with_name("global_factor_research.py"),
            Path(__file__).with_name("nifty_universe.py")]
    identity = {"version": 1, "membership_sha256": hashlib.sha256(membership.read_bytes()).hexdigest(),
                "inputs": [_fingerprint(path) for path in (membership, reference, *stock_paths.values())],
                "code_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in code},
                "train_end": train_end.isoformat(), "start_year": start_year,
                "minimum_observations": minimum_observations, "max_age_days": max_age_days,
                "tables": TABLES}
    if output.exists():
        if not resume:
            raise FileExistsError(f"Output directory already exists: {output}")
    else:
        if resume:
            raise FileNotFoundError(f"Output directory does not exist: {output}")
        output.mkdir(parents=True)
    with _price_lock(output):
        identity_path = output / "identity.json"
        if resume:
            if not identity_path.is_file() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Treasury audit inputs or settings changed; use a new directory")
        else:
            _write_json(identity_path, identity)
        control = TrainingControl(output)
        if resume:
            control.request("continue")
        state_path = output / "research-state.json"

        def boundary(stage: str, completed: int, total: int) -> None:
            state = {"stage": stage, "completed": completed, "total": total}
            _write_json(state_path, {**state, "status": "running"})
            if not control.checkpoint_boundary(lambda: _write_json(state_path, {**state, "status": "checkpointed"})):
                _write_json(state_path, {**state, "status": "stopped"})
                raise PreparationStopped("Treasury research stopped; rerun with --resume")

        source_dir = output / "source"
        source_dir.mkdir(exist_ok=True)
        sources_path = output / "sources.json"
        sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}
        factors = {}
        years = range(start_year, (train_end - timedelta(days=1)).year + 1)
        jobs = [(name, kind, year) for name, kind in TABLES.items() for year in years]
        tables = {name: [] for name in TABLES}
        for index, (name, kind, year) in enumerate(jobs):
            boundary("sources", index, len(jobs))
            key = f"{name}_{year}"
            url = f"{BASE}/{year}/all?type={kind}&field_tdr_date_value={year}&page&_format=csv"
            path = source_dir / f"{key}.csv"
            if key not in sources:
                if path.exists():
                    raise ValueError(f"Unrecorded Treasury source file: {path}")
                with urlopen(url, timeout=120) as response:
                    content = response.read(2_000_001)
                if not content or len(content) > 2_000_000:
                    raise ValueError(f"Empty or oversized Treasury response: {url}")
                frame = pd.read_csv(io.BytesIO(content), na_values=["N/A", "."])
                if "Date" not in frame.columns:
                    raise ValueError(f"Invalid Treasury CSV: {url}")
                temporary = path.with_suffix(".csv.tmp")
                temporary.write_bytes(content)
                _replace_with_retry(temporary, path)
                sources[key] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
                _write_json(sources_path, sources)
            else:
                content = path.read_bytes()
                if sources[key] != {"url": url, "sha256": hashlib.sha256(content).hexdigest()}:
                    raise ValueError(f"Treasury source snapshot changed: {path}")
                frame = pd.read_csv(io.BytesIO(content), na_values=["N/A", "."])
            tables[name].append(frame)
        for name, frames in tables.items():
            factors.update(rate_factors(frames, name))
        sessions = list(load_closes(reference).index)
        stock_dir = output / "stock_checkpoints"
        stock_dir.mkdir(exist_ok=True)
        statuses = {}
        for index, symbol in enumerate(symbols):
            boundary("stocks", index, len(symbols))
            if not stock_paths[symbol].is_file():
                statuses[symbol] = {"status": "missing_stock_data", "scored": 0}
                continue
            checkpoint = stock_dir / f"{_safe_symbol(symbol)}.json"
            if checkpoint.exists():
                rows = json.loads(checkpoint.read_text(encoding="utf-8"))
                if (not isinstance(rows, list) or any(
                    not isinstance(row, list) or len(row) != 3 or row[0] not in factors
                    or not isinstance(row[1], int) or row[1] < minimum_observations
                    or not isinstance(row[2], (int, float)) or not math.isfinite(row[2])
                    or abs(row[2]) > 1.000000001 for row in rows
                ) or len({row[0] for row in rows}) != len(rows)):
                    raise ValueError(f"Invalid Treasury stock checkpoint: {checkpoint}")
            else:
                targets = next_session_targets(load_closes(stock_paths[symbol]), sessions,
                                               tuple(item for item in members if item.symbol == symbol), train_end)
                rows = []
                for factor, values in factors.items():
                    result = correlate(targets, values, max_age=timedelta(days=max_age_days),
                                       minimum=minimum_observations)
                    if result is not None:
                        rows.append((factor, *result))
                _write_json(checkpoint, rows)
            statuses[symbol] = {"status": "scored" if rows else "insufficient_observations",
                                "scored": len(rows)}
        if [_fingerprint(path) for path in (membership, reference, *stock_paths.values())] != identity["inputs"]:
            raise ValueError("Treasury audit inputs changed during screening")
        summary = {"status": "complete", "research_only": True, "train_end_exclusive": train_end.isoformat(),
                   "minimum_observations": minimum_observations, "max_age_days": max_age_days,
                   "factor_candidates": sorted(factors), "stocks": statuses, "sources": sources,
                   "limitations": [
                       "Exploratory next-daily-return correlations; not next-candle correlations.",
                       "Treasury quotes assumed available by next midnight New York; publication timing is unverified.",
                       "Source snapshots are current and may contain historical revisions; point-in-time vintage is unverified.",
                       "Treasury rates overlap; multiple testing may yield spurious results.",
                       "Rates may not be suitable for training or commercial use without rights and timing review.",
                   ]}
        _write_json(output / "summary.json", summary)
        _write_json(state_path, {"status": "complete", "stage": "stocks",
                                "completed": len(symbols), "total": len(symbols)})
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--stocks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--minimum-observations", type=int, default=60)
    parser.add_argument("--max-age-days", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        summary = run(args.membership, args.stocks_root, args.output, train_end=args.train_end,
                      start_year=args.start_year, minimum_observations=args.minimum_observations,
                      max_age_days=args.max_age_days, resume=args.resume)
        print(f"Screened {len(summary['factor_candidates'])} Treasury rate features")
    except PreparationStopped as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
