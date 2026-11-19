"""Collect and screen additional public India-market factor candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import (
    PreparationStopped, _fingerprint, _write_json, consecutive_returns, load_closes, screen_factors,
)
from modeling.nifty_price_training import _price_lock
from modeling.nifty_universe import COVERAGE_END, COVERAGE_START, local_symbol, parse_nifty_membership


REVISION = "eb3d7aa4639eb9434409b3950d9730d4ae9a6624"
REPOSITORY = "aswinv90/indian-market-historical-data"
API = f"https://api.github.com/repos/{REPOSITORY}/contents/data"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/data"
FILES = {
    "MACRO/forex_rates.csv": ("USD_INR", "EUR_INR", "GBP_INR", "JPY_INR"),
    "MACRO/in10y_bond_yield.csv": ("Close",),
    "INDICES/INDIA_VIX.parquet": ("Close",),
    "INDICES/NIFTY_100.parquet": ("Close",),
    "INDICES/NIFTY_200.parquet": ("Close",),
    "INDICES/NIFTY_50.parquet": ("Close",),
    "INDICES/NIFTY_500.parquet": ("Close",),
    "INDICES/NIFTY_BANK.parquet": ("Close",),
    "INDICES/NIFTY_IT.parquet": ("Close",),
    "INDICES/NIFTY_MIDCAP_50.parquet": ("Close",),
    "INDICES/NIFTY_PHARMA.parquet": ("Close",),
    "INDICES/SENSEX.parquet": ("Close",),
    "COMMODITIES/GOLD_INR.parquet": ("Gold_24K_10g",),
    "COMMODITIES/SILVER_INR.parquet": ("Silver_999_1kg",),
}


def _boundary(control: TrainingControl, state_path: Path, stage: str, completed: int, total: int) -> None:
    state = {"stage": stage, "completed": completed, "total": total}
    _write_json(state_path, {**state, "status": "running"})
    if not control.checkpoint_boundary(lambda: _write_json(state_path, {**state, "status": "checkpointed"})):
        _write_json(state_path, {**state, "status": "stopped"})
        raise PreparationStopped("Public-factor research stopped; rerun with --resume")


def _blob_hash(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def _catalog() -> dict[str, dict]:
    result = {}
    for folder in ("MACRO", "INDICES", "COMMODITIES"):
        with urlopen(f"{API}/{folder}?ref={REVISION}", timeout=60) as response:
            items = json.load(response)
        for item in items:
            if item["path"].removeprefix("data/") in FILES:
                result[item["path"].removeprefix("data/")] = {
                    "git_blob_sha1": item["sha"], "bytes": item["size"]}
    if set(result) != set(FILES):
        raise ValueError(f"Pinned public factor catalog is incomplete: {sorted(set(FILES) - set(result))}")
    return result


def _read_returns(path: Path, columns: tuple[str, ...], reference: list[date]) -> dict[str, pd.Series]:
    if path.suffix == ".csv":
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    else:
        frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError(f"Factor sessions must be dated and unique: {path}")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    days = frame.index.date
    result = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        series = pd.Series(values.to_numpy(), index=days).dropna().sort_index()
        if (series.index.has_duplicates or not np.isfinite(series).all() or (series <= 0).any()):
            raise ValueError(f"Invalid public factor values in {path}, {column}")
        result[f"{path.stem}:{column}"] = consecutive_returns(series, reference)
    return result


def run(membership: Path, stocks_root: Path, output: Path, *, train_end: date,
        minimum_observations: int = 60, resume: bool = False) -> dict:
    if not COVERAGE_START < train_end <= COVERAGE_END or minimum_observations < 3:
        raise ValueError("Invalid audit cutoff or minimum observations")
    membership, stocks_root, output = membership.resolve(), stocks_root.resolve(), output.resolve()
    members = parse_nifty_membership(membership.read_text(encoding="utf-8-sig"))
    symbols = sorted({item.symbol for item in members if item.valid_from < train_end
                      and (item.valid_to is None or item.valid_to > COVERAGE_START)})
    reference = stocks_root / "day" / "RELIANCE" / "1d.csv.gz"
    stock_paths = {symbol: stocks_root / "day" / _safe_symbol(local_symbol(symbol, members)) / "1d.csv.gz"
                   for symbol in symbols}
    catalog = _catalog()
    code = [Path(__file__), Path(__file__).with_name("factor_research.py"),
            Path(__file__).with_name("nifty_universe.py"),
            Path(__file__).parent.parent / "data_pipeline" / "huggingface_stocks.py"]
    identity = {"version": 1, "repository": REPOSITORY, "revision": REVISION, "catalog": catalog,
                "membership_sha256": hashlib.sha256(membership.read_bytes()).hexdigest(),
                "code_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in code},
                "inputs": [_fingerprint(path) for path in (membership, reference, *stock_paths.values())],
                "cutoff": train_end.isoformat(), "minimum_observations": minimum_observations}
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
                raise ValueError("Public-factor inputs or settings changed; use a new directory")
        else:
            _write_json(manifest, identity)
        control = TrainingControl(output)
        if resume:
            control.request("continue")
        folder = output / "source"
        folder.mkdir(exist_ok=True)
        factors = {}
        sessions = list(load_closes(reference).index)
        state_path = output / "research-state.json"
        for index, (name, columns) in enumerate(FILES.items()):
            _boundary(control, state_path, "factors", index, len(FILES))
            path = folder / Path(name).name
            expected = catalog[name]
            if not path.is_file():
                with urlopen(f"{RAW}/{name}", timeout=120) as response:
                    content = response.read(expected["bytes"] + 1)
                if len(content) != expected["bytes"] or _blob_hash(content) != expected["git_blob_sha1"]:
                    raise ValueError(f"Downloaded factor does not match pinned Git blob: {name}")
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(content)
                _replace_with_retry(temporary, path)
            else:
                content = path.read_bytes()
                if len(content) != expected["bytes"] or _blob_hash(content) != expected["git_blob_sha1"]:
                    raise ValueError(f"Downloaded factor has changed: {path}")
            factors.update(_read_returns(path, columns, sessions))
        matrix = pd.DataFrame(factors)
        checkpoint_dir = output / "stock_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        counts = {}
        for index, symbol in enumerate(symbols):
            _boundary(control, state_path, "stocks", index, len(symbols))
            path = stock_paths[symbol]
            if not path.is_file():
                counts[symbol] = {"status": "missing_stock_data", "scored": 0}
                continue
            checkpoint = checkpoint_dir / f"{_safe_symbol(symbol)}.json"
            if checkpoint.is_file():
                rows = json.loads(checkpoint.read_text(encoding="utf-8"))
                if (not isinstance(rows, list) or any(
                    not isinstance(row, list) or len(row) != 3 or
                    not isinstance(row[0], str) or row[0] not in factors or
                    not isinstance(row[1], int) or row[1] < minimum_observations or
                    not isinstance(row[2], (int, float)) or not np.isfinite(row[2]) or
                    abs(row[2]) > 1.000000001 for row in rows
                ) or len({row[0] for row in rows}) != len(rows)):
                    raise ValueError(f"Invalid public-factor stock checkpoint: {checkpoint}")
            else:
                rows = screen_factors(load_closes(path), matrix,
                                      tuple(item for item in members if item.symbol == symbol),
                                      train_end=train_end, minimum_observations=minimum_observations,
                                      reference_sessions=sessions)
                _write_json(checkpoint, rows)
            counts[symbol] = {"status": "scored" if rows else "insufficient_observations",
                              "scored": len(rows)}
        if [_fingerprint(path) for path in (membership, reference, *stock_paths.values())] != identity["inputs"]:
            raise ValueError("Public-factor inputs changed during screening")
        result = {"status": "complete", "research_only": True, "repository": REPOSITORY,
                  "revision": REVISION, "train_end_exclusive": train_end.isoformat(),
                  "minimum_observations": minimum_observations, "factor_candidates": sorted(factors),
                  "stocks": counts, "timing_assumption": "Factor daily close assumed available at next midnight IST; source timing and revisions unverified.",
                  "license": "Non-commercial research, personal use and educational use only.",
                  "excluded": "Monthly CPI and policy series lack verified release timestamps; yearly commodity histories can be sparse."}
        _write_json(output / "summary.json", result)
        _write_json(state_path, {"status": "complete", "stage": "stocks",
                                "completed": len(symbols), "total": len(symbols)})
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--stocks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--minimum-observations", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.membership, args.stocks_root, args.output, train_end=args.train_end,
                     minimum_observations=args.minimum_observations, resume=args.resume)
        print(f"Screened {len(result['factor_candidates'])} public candidate series")
    except PreparationStopped as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
