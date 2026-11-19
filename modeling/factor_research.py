"""Exploratory, point-in-time screening of local market-factor candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.nifty_universe import (
    COVERAGE_END,
    COVERAGE_START,
    SOURCE_REVISION,
    Membership,
    local_symbol,
    parse_nifty_membership,
)


class PreparationStopped(RuntimeError):
    """Preparation stopped after a completed file checkpoint."""


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _replace_with_retry(temporary, path)


def _fingerprint(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        result.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def _factor_cache_write(path: Path, values: pd.Series) -> None:
    temporary = path.with_name(path.name + ".tmp")
    values.dropna().rename("factor_return").rename_axis("date").to_csv(temporary)
    _replace_with_retry(temporary, path)


def _factor_cache_read(path: Path) -> pd.Series:
    frame = pd.read_csv(path, dtype={"date": str})
    if list(frame.columns) != ["date", "factor_return"]:
        raise ValueError(f"Invalid factor checkpoint: {path}")
    days = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise").dt.date
    values = pd.to_numeric(frame["factor_return"], errors="raise")
    if days.isna().any() or not np.isfinite(values).all() or days.duplicated().any():
        raise ValueError(f"Invalid values in factor checkpoint: {path}")
    return pd.Series(values.to_numpy(), index=days.to_numpy(), name="factor_return")


def load_closes(path: Path) -> pd.Series:
    frame = pd.read_csv(path, usecols=["timestamp", "close"])
    days = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    prices = pd.to_numeric(frame["close"], errors="raise")
    if days.isna().any() or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"Invalid timestamps or closes in {path}")
    result = pd.Series(prices.to_numpy(), index=days.to_numpy(), name="close").sort_index()
    if result.index.has_duplicates:
        raise ValueError(f"Duplicate daily candle date in {path}")
    return result


def screen_factors(
    closes: pd.Series,
    factor_returns: pd.DataFrame,
    membership: tuple[Membership, ...],
    *,
    train_end: date,
    minimum_observations: int = 60,
    reference_sessions: list[date],
) -> list[tuple[str, int, float]]:
    if not COVERAGE_START < train_end <= COVERAGE_END:
        raise ValueError("Training cutoff is outside supported membership coverage")
    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least three")
    stock_return = closes.shift(-1).div(closes).sub(1).rename("target")
    next_day = pd.Series(closes.index, index=closes.index).shift(-1)
    next_reference = dict(zip(reference_sessions, reference_sessions[1:]))
    eligible = [
        day < train_end
        and next_day.loc[day] is not None
        and pd.notna(next_day.loc[day])
        and next_day.loc[day] < train_end
        and next_reference.get(day) == next_day.loc[day]
        and COVERAGE_START <= day <= COVERAGE_END
        and any(interval.contains(day) and interval.contains(next_day.loc[day]) for interval in membership)
        for day in closes.index
    ]
    joined = factor_returns.join(stock_return.loc[eligible], how="inner")
    results = []
    for name in factor_returns.columns:
        pairs = joined[[name, "target"]].dropna()
        if len(pairs) < minimum_observations:
            continue
        correlation = pairs[name].corr(pairs["target"])
        if math.isfinite(correlation):
            results.append((name, len(pairs), float(correlation)))
    return results


def consecutive_returns(closes: pd.Series, reference_sessions: list[date]) -> pd.Series:
    previous_reference = dict(zip(reference_sessions[1:], reference_sessions))
    days = pd.Series(closes.index, index=closes.index)
    previous = days.shift(1)
    valid = [previous_reference.get(day) == prior for day, prior in zip(days, previous)]
    return closes.pct_change(fill_method=None).where(valid)


def run_audit(
    membership_path: Path,
    stocks_root: Path,
    output: Path,
    *,
    train_end: date,
    minimum_observations: int,
    resume: bool = False,
    include_peers: bool = False,
) -> None:
    if not COVERAGE_START < train_end <= COVERAGE_END:
        raise ValueError("Training cutoff is outside supported membership coverage")
    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least three")
    intervals = parse_nifty_membership(membership_path.read_text(encoding="utf-8-sig"))
    daily = stocks_root / "day"
    factor_paths = sorted(
        p / "1d.csv.gz"
        for p in daily.iterdir()
        if p.is_dir()
        and (include_peers or p.name.startswith("NIFTY") or p.name in {"BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"})
        and (p / "1d.csv.gz").is_file()
    )
    if not factor_paths:
        raise ValueError("No market-factor candidate files found")
    symbols = sorted(
        {
            entry.symbol
            for entry in intervals
            if entry.valid_from < train_end
            and (entry.valid_to is None or entry.valid_to > COVERAGE_START)
        }
    )
    local_names = {symbol: _safe_symbol(local_symbol(symbol, intervals)) for symbol in symbols}
    stock_paths = {symbol: daily / local_names[symbol] / "1d.csv.gz" for symbol in symbols}
    reference = daily / "RELIANCE" / "1d.csv.gz"
    reference_sessions = list(load_closes(reference).index)
    if len(set(local_names.values())) != len(symbols):
        raise ValueError("Membership symbols map to duplicate local filenames")
    manifest = {
        "format_version": 4 if include_peers else 3,
        "membership": _fingerprint(membership_path),
        "membership_sha256": hashlib.sha256(membership_path.read_bytes()).hexdigest(),
        "stocks_root": str(stocks_root.resolve()),
        "train_end": train_end.isoformat(),
        "minimum_observations": minimum_observations,
        "factor_inputs": [_fingerprint(path) for path in factor_paths],
        "stock_inputs": {symbol: _fingerprint(path) for symbol, path in stock_paths.items()},
        "reference_calendar": _fingerprint(reference),
    }
    if include_peers:
        manifest["candidate_policy"] = "all local daily files, excluding the target stock itself"
    manifest_path = output / "preparation-manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Output directory already exists: {output}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No resumable preparation manifest: {manifest_path}")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Inputs or settings changed; use a new output directory")
    else:
        if resume:
            raise FileNotFoundError(f"No resumable output directory: {output}")
        output.mkdir(parents=True)
        _write_json(manifest_path, manifest)
    factor_cache = output / "factor_checkpoints"
    stock_cache = output / "stock_checkpoints"
    factor_cache.mkdir(exist_ok=True)
    stock_cache.mkdir(exist_ok=True)
    control = TrainingControl(output)
    if resume:
        control.request("continue")
    state_path = output / "preparation-state.json"

    def boundary(stage: str, completed: int, total: int) -> None:
        state = {"stage": stage, "completed": completed, "total": total}
        _write_json(state_path, {**state, "status": "running"})

        def save() -> None:
            _write_json(state_path, {**state, "status": "checkpointed"})

        if not control.checkpoint_boundary(save):
            _write_json(state_path, {**state, "status": "stopped"})
            raise PreparationStopped(
                f"Preparation stopped at {stage} {completed}/{total}; resume with --resume."
            )

    boundary("factors", 0, len(factor_paths))
    cached_factors: dict[str, pd.Series] = {}
    for index, path in enumerate(factor_paths, 1):
        checkpoint = factor_cache / f"{path.parent.name}.csv"
        if not checkpoint.is_file():
            _factor_cache_write(checkpoint, consecutive_returns(load_closes(path), reference_sessions))
        cached_factors[path.parent.name] = _factor_cache_read(checkpoint)
        boundary("factors", index, len(factor_paths))
    factors = pd.concat(cached_factors, axis=1)

    correlations: list[tuple[str, str, int, float]] = []
    missing: list[str] = []
    no_scorable: list[str] = []
    no_qualifying: list[str] = []
    boundary("stocks", 0, len(symbols))
    for index, symbol in enumerate(symbols, 1):
        stock_path = stock_paths[symbol]
        if not stock_path.is_file():
            missing.append(symbol)
            boundary("stocks", index, len(symbols))
            continue
        checkpoint = stock_cache / f"{_safe_symbol(symbol)}.json"
        if checkpoint.is_file():
            rows = json.loads(checkpoint.read_text(encoding="utf-8"))
            results = [(str(name), int(count), float(value)) for name, count, value in rows]
            if any(name not in factors.columns or count < minimum_observations
                   or not math.isfinite(value) or abs(value) > 1.000000001
                   or (include_peers and name == local_names[symbol])
                   for name, count, value in results):
                raise ValueError(f"Invalid stock checkpoint: {checkpoint}")
        else:
            stock_intervals = tuple(entry for entry in intervals if entry.symbol == symbol)
            results = screen_factors(
                load_closes(stock_path),
                factors.drop(columns=[local_names[symbol]], errors="ignore") if include_peers else factors,
                stock_intervals,
                train_end=train_end,
                minimum_observations=minimum_observations,
                reference_sessions=reference_sessions,
            )
            _write_json(checkpoint, results)
        correlations.extend((symbol, name, count, value) for name, count, value in results)
        if not results:
            no_scorable.append(symbol)
        elif not any(abs(value) >= 0.7 for _, _, value in results):
            no_qualifying.append(symbol)
        boundary("stocks", index, len(symbols))

    boundary("finalizing", 0, 1)
    correlations_path = output / "correlations.csv"
    temporary = correlations_path.with_name(correlations_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["symbol", "factor", "paired_sessions", "correlation", "qualifies"])
        writer.writerows(
            (symbol, factor, count, f"{value:.8f}", abs(value) >= 0.7)
            for symbol, factor, count, value in correlations
        )
    _replace_with_retry(temporary, correlations_path)
    candidates_path = output / "factor_candidates.csv"
    temporary = candidates_path.with_name(candidates_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["factor", "local_file"])
        writer.writerows((path.parent.name, str(path)) for path in factor_paths)
    _replace_with_retry(temporary, candidates_path)
    summary = {
        "membership_source_revision": SOURCE_REVISION,
        "train_end_exclusive": train_end.isoformat(),
        "minimum_observations": minimum_observations,
        "threshold_absolute": 0.7,
        "method": "same-session factor return vs next-session stock return; training period only",
        "factor_candidates": len(factor_paths),
        "historical_symbols": len(symbols),
        "missing_daily_files": missing,
        "no_scorable_factor": no_scorable,
        "no_qualifying_factor": no_qualifying,
        "limitations": [
            "Exploratory daily screening only; not a trained model.",
            ("All available local daily names are candidates; the target's local name is excluded."
             if include_peers else
             "Candidate names are a heuristic and include market indices and exchange-traded funds."),
            "The local file catalog is retrospective and can introduce survivorship and selection bias.",
            "Historical symbol renames and corporate actions require separate verification.",
            "Factors may contain the target stock and correlation is not evidence of causation.",
            "Source membership before April 2018 and after October 13, 2026 is not verified.",
            "RELIANCE dates are a session-calendar proxy, not an authoritative exchange calendar.",
            "Same-session factor availability assumes next-midnight IST and unverified publication latency.",
        ],
    }
    _write_json(output / "summary.json", summary)
    _write_json(state_path, {"stage": "finalizing", "completed": 1, "total": 1, "status": "completed"})
    print(f"Saved {len(correlations)} correlations to {output}")
    print(
        f"Missing stock files: {len(missing)}; no scorable factor: {len(no_scorable)}; "
        f"no qualifying factor: {len(no_qualifying)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--stocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--minimum-observations", type=int, default=60)
    parser.add_argument("--resume", action="store_true", help="Resume a stopped or interrupted audit")
    parser.add_argument("--include-peers", action="store_true",
                        help="Also screen all locally available daily stock and fund files")
    args = parser.parse_args()
    try:
        run_audit(
            args.membership,
            args.stocks,
            args.output,
            train_end=args.train_end,
            minimum_observations=args.minimum_observations,
            resume=args.resume,
            include_peers=args.include_peers,
        )
    except PreparationStopped as exc:
        print(exc)


if __name__ == "__main__":
    main()
