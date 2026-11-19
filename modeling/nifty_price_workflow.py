"""Prepare, train and evaluate a resumable research-only Nifty price run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data_pipeline.huggingface_stocks import _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import PreparationStopped, _fingerprint, _write_json, consecutive_returns, load_closes
from modeling.global_factor_research import global_returns
from modeling.nifty_factor_selection import _validate_fingerprints, run as rebuild_selection
from modeling.nifty_preparation import TIMEFRAMES, _daily_dates
from modeling.nifty_price_evaluation import EvaluationStopped, _sha, evaluate_holdout
from modeling.nifty_price_model import NoPriceExamples, build_examples, load_prepared_candles, selected_factors
from modeling.nifty_price_training import (
    PriceTrainingConfig, _price_lock, prepare_price_dataset, train_pooled_price_model,
)
from modeling.nifty_universe import parse_nifty_membership
from modeling.training import TrainingStopped


SOURCE_FILES = (
    ("modeling", "nifty_price_workflow.py"), ("modeling", "nifty_price_model.py"),
    ("modeling", "nifty_price_training.py"), ("modeling", "nifty_preparation.py"),
    ("modeling", "nifty_universe.py"), ("modeling", "nifty_factor_selection.py"),
    ("modeling", "factor_research.py"), ("modeling", "global_factor_research.py"),
    ("modeling", "control.py"), ("modeling", "data.py"), ("data_pipeline", "huggingface_stocks.py"),
)


def _validate_selection(path: Path, membership: Path) -> dict:
    selection = json.loads(path.read_text(encoding="utf-8"))
    identity = json.loads((path.parent / "identity.json").read_text(encoding="utf-8"))
    audits = selection["audits"]
    if not audits or audits[0]["source"] != "local":
        raise ValueError("Selection requires a local audit followed by global audits")
    with tempfile.TemporaryDirectory(prefix="nifty-selection-check-") as directory:
        regenerated = Path(directory) / "selection"
        result = rebuild_selection(membership, Path(audits[0]["directory"]),
                                   [Path(item["directory"]) for item in audits[1:]], regenerated,
                                   train_end=date.fromisoformat(selection["train_end_exclusive"]))
        regenerated_identity = json.loads((regenerated / "identity.json").read_text(encoding="utf-8"))
    if result != selection or identity != regenerated_identity:
        raise ValueError("Selection or its recorded audit identity does not match the source audits")
    return selection


def _read_global_part(root: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    folder = root / "prepared" / symbol
    part = folder / f"{timeframe}.parquet"
    checkpoint = json.loads((folder / f"{timeframe}.json").read_text(encoding="utf-8"))
    if _sha(part) != checkpoint["sha256"]:
        raise ValueError(f"Corrupt global factor checkpoint: {part}")
    frame = pd.read_parquet(part)
    if len(frame) != checkpoint["rows"]:
        raise ValueError(f"Global factor row count mismatch: {part}")
    return frame


def _factor_inputs(selection: dict, chosen: set[tuple[str, str]], reference: list[date],
                   max_age_days: int, control: TrainingControl) -> dict:
    result = {}
    audits = selection["audits"]
    local_manifest = json.loads((Path(audits[0]["directory"]) / "preparation-manifest.json").read_text(encoding="utf-8"))
    local_paths = {Path(item["path"]).parent.name: Path(item["path"]) for item in local_manifest["factor_inputs"]}
    globals_ = []
    for audit in audits[1:]:
        settings = audit["settings"]
        if settings["max_age_days"] != max_age_days:
            raise ValueError("Factor maximum age must match the selection audit")
        manifest = json.loads((Path(audit["directory"]) / "identity.json").read_text(encoding="utf-8"))
        root = Path(manifest["inputs"][2]["path"]).parent
        info = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        globals_.append((audit["source"], root, {item["symbol"] for item in info["factors"]},
                         settings["max_close_age_minutes"]))
    calendars = {}
    for source, name in sorted(chosen):
        if not control.checkpoint_boundary(lambda: None):
            raise PreparationStopped("Stopped while preparing factor returns; rerun with --resume")
        if source == "local":
            if name not in local_paths:
                raise ValueError(f"Selected local factor is absent from the audit: {name}")
            returns = consecutive_returns(load_closes(local_paths[name]), reference).dropna()
            available = (pd.to_datetime(returns.index.astype(str)).tz_localize("Asia/Kolkata")
                         + timedelta(days=1)).tz_convert("UTC")
            result[(source, name)] = pd.DataFrame({"factor_available_at": available,
                                                   "factor_return": returns.to_numpy()})
            continue
        matches = [(root, close_age) for audit_source, root, names, close_age in globals_
                   if source == audit_source and name in names]
        if not matches:
            raise ValueError(f"Selected global factor is absent from the audits: {source}, {name}")
        candidate = None
        for root, close_age in matches:
            if root not in calendars:
                spy = _read_global_part(root, "SPY", "1d")
                sessions = sorted(spy["timestamp"].dt.tz_convert("America/New_York").dt.date.tolist())
                if len(sessions) != len(set(sessions)):
                    raise ValueError("SPY reference contains duplicate sessions")
                calendars[root] = sessions
            minutes = _read_global_part(root, name, "1m") if close_age is not None else None
            current = global_returns(_read_global_part(root, name, "1d"), calendars[root],
                                     minutes=minutes, max_close_age_minutes=close_age)
            if candidate is not None and not candidate.equals(current):
                raise ValueError(f"Conflicting duplicate global factor data: {name}")
            candidate = current
        result[(source, name)] = candidate
    return result


def _input_identity(config: dict) -> dict:
    membership = Path(config["membership"])
    paths = [membership, Path(config["selection"]), Path(config["selection"]).parent / "identity.json",
             Path(config["reference"])]
    for root_name in ("intraday", "daily", "multisession"):
        root = Path(config[root_name])
        identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
        if identity["membership_sha256"] != _sha(membership):
            raise ValueError(f"Prepared candle membership differs: {root}")
        _validate_fingerprints(identity["inputs"])
        paths.extend((root / "identity.json", root / "coverage.json"))
        coverage = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
        timeframes = (TIMEFRAMES[:6] if root_name == "intraday" else
                      ("1d",) if root_name == "daily" else ("2d", "trading_week"))
        for symbol in config["symbols"]:
            for tf in timeframes:
                matches = [job for job in coverage["jobs"] if job["kind"] == "stock"
                           and job["symbol"] == symbol and job["timeframe"] == tf]
                if len(matches) != 1:
                    raise ValueError(f"Expected one prepared job for {symbol} {tf} in {root}")
                job = matches[0]
                if job["status"] == "prepared":
                    folder = root / "stock" / _safe_symbol(symbol) / tf
                    for index in range(job["chunks"]):
                        part = folder / f"part-{index:06d}.parquet"
                        paths.extend((part, part.with_suffix(".json")))
    fingerprints = [_fingerprint(path) for path in paths]
    if any(not item["exists"] for item in fingerprints):
        raise FileNotFoundError("A required preparation input or checkpoint is missing")
    project = Path(__file__).resolve().parent.parent
    return {"version": 1, "config": config, "inputs": fingerprints,
            "small_input_sha256": {str(path): _sha(path) for path in paths if path.suffix != ".parquet"},
            "source_sha256": {".".join(name): _sha(project.joinpath(*name)) for name in SOURCE_FILES},
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow")}}


def _boundary(control: TrainingControl) -> None:
    if not control.checkpoint_boundary(lambda: None):
        raise PreparationStopped("Preparation stopped at a checkpoint; rerun with --resume")


def prepare_run(config: dict, output: Path, *, resume: bool = False) -> dict:
    output = output.resolve()
    selection = _validate_selection(Path(config["selection"]), Path(config["membership"]))
    local_manifest = json.loads((Path(selection["audits"][0]["directory"]) / "preparation-manifest.json")
                                .read_text(encoding="utf-8"))
    if _sha(Path(config["reference"])) != _sha(Path(local_manifest["reference_calendar"]["path"])):
        raise ValueError("Reference calendar differs from the factor-selection audit")
    for audit in selection["audits"][1:]:
        manifest = json.loads((Path(audit["directory"]) / "identity.json").read_text(encoding="utf-8"))
        if _sha(Path(config["reference"])) != _sha(Path(manifest["inputs"][1]["path"])):
            raise ValueError("Reference calendar differs from the global factor-selection audit")
    identity = _input_identity(config)
    if date.fromisoformat(config["start"]) < date.fromisoformat(selection["train_end_exclusive"]):
        raise ValueError("Example window starts before the factor-selection cutoff")
    if not date.fromisoformat(config["start"]) < date.fromisoformat(config["validation_start"]) < date.fromisoformat(config["holdout_start"]) < date.fromisoformat(config["end"]):
        raise ValueError("Start, validation, holdout and end dates must be strictly increasing")
    if output.exists():
        if not resume:
            raise FileExistsError(f"Run directory already exists: {output}")
    else:
        if resume:
            raise FileNotFoundError(f"Run directory does not exist: {output}")
        output.mkdir(parents=True)
    with _price_lock(output):
        manifest = output / "identity.json"
        if resume:
            if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("Preparation inputs or settings changed; use a new run directory")
        else:
            _write_json(manifest, identity)
        control = TrainingControl(output)
        if resume:
            control.request("continue")
        intervals = parse_nifty_membership(Path(config["membership"]).read_text(encoding="utf-8-sig"))
        reference = _daily_dates(Path(config["reference"]))
        stocks = {item["symbol"]: item for item in selection["stocks"]}
        if len(stocks) != len(selection["stocks"]) or set(config["symbols"]) - set(stocks):
            raise ValueError("Requested stocks must be unique members of the selection")
        if len({_safe_symbol(symbol) for symbol in config["symbols"]}) != len(config["symbols"]):
            raise ValueError("Requested symbols map to duplicate checkpoint filenames")
        chosen = {}
        skipped = []
        for symbol in config["symbols"]:
            status = stocks[symbol]["status"]
            if status in {"missing_stock_data", "insufficient_observations"} and config["skip_unusable"]:
                skipped.append({"symbol": symbol, "reason": status})
                continue
            chosen[symbol] = selected_factors(selection, symbol, date.fromisoformat(config["start"]))
            if not chosen[symbol] and not config["allow_no_factors"]:
                raise ValueError(f"{symbol} has no qualifying factors; use --allow-no-factors explicitly")
        factors = _factor_inputs(selection, {key for keys in chosen.values() for key in keys},
                                 reference, config["factor_age_days"], control)
        folder = output / "examples"
        folder.mkdir(exist_ok=True)
        ages = {tf: timedelta(hours=config["intraday_age_hours"]) for tf in TIMEFRAMES[:6]}
        ages["1d"] = timedelta(days=config["daily_age_days"])
        ages.update({tf: timedelta(days=config["multiday_age_days"]) for tf in ("2d", "trading_week")})
        items = []
        for symbol, keys in chosen.items():
            _boundary(control)
            path = folder / f"{_safe_symbol(symbol)}.parquet"
            meta_path = path.with_suffix(".json")
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not path.is_file() or _sha(path) != meta["sha256"]:
                    raise ValueError(f"Example checkpoint is missing or corrupt: {path}")
                items.append(meta)
                continue
            roots = {tf: Path(config["intraday"]) for tf in TIMEFRAMES[:6]}
            roots["1d"] = Path(config["daily"])
            roots.update({tf: Path(config["multisession"]) for tf in ("2d", "trading_week")})
            candles = {}
            for tf in TIMEFRAMES:
                _boundary(control)
                candles[tf] = load_prepared_candles(roots[tf], symbol, tf, max_rows=config["max_rows"])
            if candles["1d"] is None:
                if not config["skip_unusable"]:
                    raise ValueError(f"{symbol} has no daily candles; use --skip-unusable to report and omit")
                skipped.append({"symbol": symbol, "reason": "missing_daily_candles"})
                continue
            missing = [tf for tf in TIMEFRAMES if candles[tf] is None]
            if missing and not config["allow_missing_timeframes"]:
                raise ValueError(f"{symbol} is missing timeframes {missing}; use --allow-missing-timeframes explicitly")
            try:
                frame, _, missing = build_examples(
                    symbol, candles["1d"], candles, {key: factors[key] for key in keys},
                    selection, reference, tuple(item for item in intervals if item.symbol == symbol),
                    start=date.fromisoformat(config["start"]), end=date.fromisoformat(config["end"]),
                    max_candle_age=ages, max_factor_age=timedelta(days=config["factor_age_days"]),
                    allow_no_factors=config["allow_no_factors"])
            except NoPriceExamples:
                if not config["skip_unusable"]:
                    raise
                skipped.append({"symbol": symbol, "reason": "no_eligible_examples"})
                continue
            temporary = path.with_suffix(".tmp")
            frame.to_parquet(temporary, index=False)
            _replace_with_retry(temporary, path)
            meta = {"symbol": symbol, "rows": len(frame), "sha256": _sha(path),
                    "missing_timeframes": missing, "selected_factors": len(keys)}
            _write_json(meta_path, meta)
            items.append(meta)
            _write_json(output / "preparation-state.json",
                        {"status": "running", "completed": len(items) + len(skipped),
                         "total": len(config["symbols"])})
        _boundary(control)
        if not items:
            raise ValueError(f"No usable stock examples; omitted stocks: {skipped}")
        if _input_identity(config) != identity:
            raise ValueError("Preparation inputs changed during processing")
        result = {"status": "complete", "research_only": True, "stocks": items, "skipped": skipped}
        dataset = _assemble_dataset(output, config, selection, result, control)
        if _input_identity(config) != identity:
            raise ValueError("Preparation inputs changed during dataset assembly")
        result["split_counts"] = {"train": len(dataset.train), "validation": len(dataset.validation),
                                  "holdout": len(dataset.holdout)}
        _write_json(output / "preparation-summary.json", result)
        _write_json(output / "preparation-state.json",
                    {"status": "complete", "completed": len(items) + len(skipped), "total": len(config["symbols"])})
        return result


def _assemble_dataset(output: Path, config: dict, selection: dict, summary: dict,
                      control: TrainingControl):
    examples = {}
    for item in summary["stocks"]:
        if not control.checkpoint_boundary(lambda: None):
            raise PreparationStopped("Stopped during dataset assembly; restart the current command")
        path = output / "examples" / f"{_safe_symbol(item['symbol'])}.parquet"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if item != meta or _sha(path) != item["sha256"]:
            raise ValueError(f"Example checkpoint is missing or corrupt: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != item["rows"]:
            raise ValueError(f"Example row count is invalid: {path}")
        examples[item["symbol"]] = frame
    try:
        dataset = prepare_price_dataset(
            examples, selection,
            parse_nifty_membership(Path(config["membership"]).read_text(encoding="utf-8-sig")),
            _daily_dates(Path(config["reference"])),
            validation_start=date.fromisoformat(config["validation_start"]),
            holdout_start=date.fromisoformat(config["holdout_start"]),
            sequence_length=config["sequence_length"], allow_no_factors=config["allow_no_factors"],
            control=control)
    except TrainingStopped as exc:
        raise PreparationStopped("Stopped during dataset assembly; restart the current command") from exc
    return dataset


def load_run(output: Path):
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    config = identity["config"]
    if _input_identity(config) != identity:
        raise ValueError("Preparation inputs or settings changed")
    selection = _validate_selection(Path(config["selection"]), Path(config["membership"]))
    summary = json.loads((output / "preparation-summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Preparation is not complete")
    included = [item["symbol"] for item in summary["stocks"]]
    omitted = [item["symbol"] for item in summary["skipped"]]
    if (len(set(included + omitted)) != len(included) + len(omitted)
            or set(included + omitted) != set(config["symbols"])):
        raise ValueError("Preparation summary does not account for every requested stock")
    dataset = _assemble_dataset(output, config, selection, summary, TrainingControl(output))
    if summary["split_counts"] != {"train": len(dataset.train), "validation": len(dataset.validation),
                                    "holdout": len(dataset.holdout)}:
        raise ValueError("Prepared split counts changed")
    return dataset, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("run-dir", "membership", "selection", "reference", "intraday", "daily", "multisession"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    for name in ("start", "end", "validation-start", "holdout-start"):
        prepare.add_argument(f"--{name}", type=date.fromisoformat, required=True)
    prepare.add_argument("--symbols", nargs="+", required=True)
    for name in ("intraday-age-hours", "daily-age-days", "multiday-age-days", "factor-age-days",
                 "sequence-length", "max-rows"):
        prepare.add_argument(f"--{name}", type=int, required=True)
    for name in ("allow-no-factors", "allow-missing-timeframes", "skip-unusable", "resume"):
        prepare.add_argument(f"--{name}", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--research-only", action="store_true", required=True)
    train.add_argument("--tree-rounds", type=int, required=True)
    train.add_argument("--epochs", type=int, required=True)
    train.add_argument("--batch-size", type=int, required=True)
    train.add_argument("--learning-rate", type=float, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--checkpoint-batches", type=int, default=10)
    train.add_argument("--device", choices=("cpu", "cuda"), required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), required=True)
    evaluate.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        output = args.run_dir.resolve()
        if args.command == "prepare":
            config = vars(args).copy()
            for key in ("command", "run_dir", "resume"):
                del config[key]
            for key, value in config.items():
                if isinstance(value, Path):
                    config[key] = str(value.resolve())
                elif isinstance(value, date):
                    config[key] = value.isoformat()
            config["symbols"] = sorted(config["symbols"])
            if len(set(config["symbols"])) != len(config["symbols"]):
                raise ValueError("Duplicate requested stock")
            if min(config[key] for key in ("intraday_age_hours", "daily_age_days", "multiday_age_days",
                                           "factor_age_days", "sequence_length", "max_rows")) < 1:
                raise ValueError("Age limits, sequence length and max rows must be positive")
            result = prepare_run(config, output, resume=args.resume)
            print(f"Prepared {len(result['stocks'])} stocks; omitted {len(result['skipped'])}")
        else:
            with _price_lock(output):
                dataset, config = load_run(output)
                if args.command == "train":
                    training_config = PriceTrainingConfig(
                        sequence_length=config["sequence_length"], tree_rounds=args.tree_rounds,
                        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                        seed=args.seed, device=args.device, checkpoint_batches=args.checkpoint_batches)
                    result = train_pooled_price_model(dataset, output / "model", config=training_config,
                                                      research_only=args.research_only, resume=args.resume)
                else:
                    result = evaluate_holdout(dataset, output / "model", output / "evaluation",
                                              batch_size=args.batch_size, device=args.device, resume=args.resume)
                print(f"{args.command}: {result['status']}")
    except (PreparationStopped, TrainingStopped, EvaluationStopped) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
