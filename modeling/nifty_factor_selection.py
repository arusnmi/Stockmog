"""Combine completed causal research audits into per-stock factor selections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path

from data_pipeline.huggingface_stocks import _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import PreparationStopped, _fingerprint, _write_json
from modeling.nifty_universe import COVERAGE_START, parse_nifty_membership


THRESHOLD = 0.7


def _validate_fingerprints(records: list[dict]) -> None:
    for record in records:
        if _fingerprint(Path(record["path"])) != record:
            raise ValueError(f"Audit input changed: {record['path']}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_audit(directory: Path, kind: str, membership_hash: str, cutoff: date) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if kind == "local":
        manifest_path = directory / "preparation-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 3:
            raise ValueError("Local audit must use consecutive-session format version 3")
        if json.loads((directory / "preparation-state.json").read_text(encoding="utf-8")).get("status") != "completed":
            raise ValueError("Local audit is not complete")
        inputs = [manifest["membership"], manifest["reference_calendar"],
                  *manifest["factor_inputs"], *manifest["stock_inputs"].values()]
        candidates = {Path(item["path"]).parent.name for item in manifest["factor_inputs"]}
        minimum = manifest["minimum_observations"]
        end = manifest["train_end"]
        source = "local"
        settings = None
    elif kind == "global":
        manifest_path = directory / "identity.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if summary.get("status") != "research_only" or manifest.get("version") != 1:
            raise ValueError("Global audit is incomplete or unsupported")
        inputs = manifest["inputs"]
        factor_info = json.loads(Path(inputs[2]["path"]).read_text(encoding="utf-8"))
        if factor_info.get("status") != "complete" or summary["unavailable_factors"]:
            raise ValueError("Global factor collection has unavailable sources")
        candidates = {item["symbol"] for item in factor_info["factors"]}
        minimum = manifest["minimum_observations"]
        end = manifest["train_end"]
        source = f"{factor_info['source']}@{factor_info['revision']}"
        settings = {"max_age_days": manifest["max_age_days"],
                    "max_close_age_minutes": manifest.get("max_close_age_minutes")}
    else:
        raise ValueError(f"Unknown audit kind: {kind}")
    _validate_fingerprints(inputs)
    if (manifest["membership_sha256"] != membership_hash or end != cutoff.isoformat()
            or summary["train_end_exclusive"] != end or summary["minimum_observations"] != minimum
            or summary["threshold_absolute"] != THRESHOLD or minimum < 3):
        raise ValueError("Audit membership, cutoff, minimum observations or threshold differs")
    if len(candidates) != summary["factor_candidates"]:
        raise ValueError("Audit factor inventory is inconsistent")
    return {"directory": directory, "summary": summary, "minimum": minimum,
            "candidates": candidates, "source": source, "manifest": manifest_path, "settings": settings}


def stock_rows(audit: dict, symbol: str) -> list[tuple[str, int, float]] | None:
    summary = audit["summary"]
    if symbol in summary["missing_daily_files"]:
        return None
    path = audit["directory"] / "stock_checkpoints" / f"{_safe_symbol(symbol)}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    parsed = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"Invalid audit checkpoint: {path}")
        name, count, value = row
        if (not isinstance(name, str) or name in seen or name not in audit["candidates"]
                or not isinstance(count, int) or count < audit["minimum"]
                or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1.000000001):
            raise ValueError(f"Invalid audit result in {path}")
        seen.add(name)
        parsed.append((name, count, float(value)))
    no_scorable = symbol in summary["no_scorable_factor"]
    no_qualifying = symbol in summary["no_qualifying_factor"]
    if no_scorable != (not parsed) or no_qualifying != (
        bool(parsed) and not any(abs(value) >= THRESHOLD for _, _, value in parsed)
    ):
        raise ValueError(f"Audit summary does not match checkpoint for {symbol}")
    return parsed


def run(membership: Path, local: Path, global_audits: list[Path], output: Path, *,
        train_end: date, resume: bool = False) -> dict:
    membership_hash = _sha(membership)
    intervals = parse_nifty_membership(membership.read_text(encoding="utf-8-sig"))
    symbols = sorted({entry.symbol for entry in intervals if entry.valid_from < train_end
                      and (entry.valid_to is None or entry.valid_to > COVERAGE_START)})
    audits = [load_audit(local, "local", membership_hash, train_end)]
    audits.extend(load_audit(path, "global", membership_hash, train_end) for path in global_audits)
    if any(audit["summary"]["historical_symbols"] != len(symbols) for audit in audits):
        raise ValueError("Audits do not cover the same historical universe")
    if len({audit["minimum"] for audit in audits}) != 1:
        raise ValueError("Audits must use the same minimum observations")
    if any(audit["settings"] != audits[1]["settings"] for audit in audits[2:]):
        raise ValueError("Global audits must use the same factor freshness and close policy")
    audit_files = [membership]
    for audit in audits:
        audit_files.extend([audit["manifest"], audit["directory"] / "summary.json"])
        audit_files.extend(audit["directory"] / "stock_checkpoints" / f"{_safe_symbol(symbol)}.json"
                           for symbol in symbols if symbol not in audit["summary"]["missing_daily_files"])
    identity = {"version": 1, "cutoff": train_end.isoformat(), "threshold_absolute": THRESHOLD,
                "files": [{"path": str(path.resolve()), "sha256": _sha(path)} for path in audit_files]}
    identity_path = output / "identity.json"
    if output.exists():
        if not resume:
            raise FileExistsError(f"Selection output exists: {output}")
        if not identity_path.is_file() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Selection inputs changed; use a new directory")
    else:
        if resume:
            raise FileNotFoundError(f"Selection output does not exist: {output}")
        output.mkdir(parents=True)
        _write_json(identity_path, identity)
    control = TrainingControl(output)
    if resume:
        control.request("continue")
    stocks = []
    for symbol in symbols:
        if not control.checkpoint_boundary(lambda: None):
            raise PreparationStopped("Selection stopped; rerun with --resume")
        scored: dict[tuple[str, str], tuple[int, float]] = {}
        missing = False
        for audit in audits:
            rows = stock_rows(audit, symbol)
            if rows is None:
                missing = True
                continue
            for name, count, value in rows:
                key = (audit["source"], name)
                previous = scored.get(key)
                if previous is not None and (previous[0] != count or not math.isclose(previous[1], value, abs_tol=1e-12)):
                    raise ValueError(f"Conflicting duplicate factor results for {symbol}: {name}")
                scored[key] = (count, value)
        selected = [
            {"source": source, "factor": factor, "paired_sessions": count, "correlation": value}
            for (source, factor), (count, value) in scored.items() if abs(value) >= THRESHOLD
        ]
        selected.sort(key=lambda item: (-abs(item["correlation"]), item["source"], item["factor"]))
        status = ("missing_stock_data" if missing else "selected" if selected else
                  "no_qualifying_factor" if scored else "insufficient_observations")
        stocks.append({"symbol": symbol, "status": status, "scored_factors": len(scored), "selected": selected})
    result = {
        "status": "research_only", "train_end_exclusive": train_end.isoformat(),
        "threshold_absolute": THRESHOLD, "minimum_observations": audits[0]["minimum"],
        "audits": [{"directory": str(audit["directory"].resolve()), "source": audit["source"],
                    "method": audit["summary"].get("method"), "settings": audit["settings"]}
                   for audit in audits],
        "stocks": stocks,
        "limitations": [
            "Selections use all observations before the cutoff and must not be used on earlier training rows; use chronological fold-specific selection.",
            "No factor is approved for production or training: source rights, timestamps and adjustment methods require verification.",
            "Local and US calendars are proxies; factor publication delays and special sessions are not verified.",
            "Multiple testing, overlapping factors and constituent inclusion can distort correlations.",
        ],
    }
    temporary = output / "stocks.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["symbol", "status", "scored_factors", "selected_factors"])
        writer.writerows((item["symbol"], item["status"], item["scored_factors"], len(item["selected"]))
                         for item in stocks)
    _replace_with_retry(temporary, output / "stocks.csv")
    _write_json(output / "selection.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, default=Path("data") / "external" / "nse_index_membership_0786cd915974.csv")
    parser.add_argument("--local-audit", type=Path, required=True)
    parser.add_argument("--global-audit", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.membership, args.local_audit, args.global_audit, args.output,
                     train_end=args.train_end, resume=args.resume)
        print(f"Selected factors for {sum(bool(stock['selected']) for stock in result['stocks'])}/{len(result['stocks'])} stocks")
    except PreparationStopped as exc:
        print(exc)


if __name__ == "__main__":
    main()
