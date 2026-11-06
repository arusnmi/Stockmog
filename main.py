from __future__ import annotations

import argparse
import hashlib
from importlib.util import find_spec
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from backtesting import BacktestConfig, BacktestResult, run_backtest
from backtesting.execution_simulator import ExecutionCosts
from config.settings import DATA_DIR, ensure_project_structure
from data_pipeline.csv_import import import_csv
from data_pipeline.csv_dataset import inspect_training_csv
from data_pipeline.huggingface_stocks import (
    NSE_HOLIDAYS_CSV,
    prepare_huggingface_dataset,
)
from data_pipeline.providers import YahooFinanceProvider
from data_pipeline.storage import CSVStore
from data_pipeline.validation import validate_ohlcv
from features.feature_engineering import build_features, write_features
from modeling.control import TrainingControl
from modeling.batch_plan import build_training_batch_plan, write_training_batch_plan
from modeling.batch_trainer import run_training_batch
from modeling.data import build_training_dataset, chronological_split
from modeling.hybrid import predict_hybrid_signals
from modeling.training import (
    TrainingStopped,
    calibrate_hybrid,
    classification_metrics,
    train_classical_models,
    train_lstm,
    train_xgboost,
)
from strategies import (
    CATALOG_STRATEGIES,
    STRATEGIES,
    generate_all_catalog_signals,
    generate_all_signals,
    generate_strategy_signals,
    strategies_for_style,
)


def _make_default_range(days: int = 365) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def cmd_download(args: argparse.Namespace) -> None:
    ensure_project_structure()
    provider = YahooFinanceProvider()
    start, end = _make_default_range(args.days)
    rows = provider.fetch_history(args.symbol, start, end, interval=args.interval)
    validation = validate_ohlcv(rows, interval=args.interval)

    store = CSVStore(DATA_DIR / "raw")
    path = store.save_rows(args.symbol, args.interval, rows)
    print(f"Saved {len(rows)} rows to {path}")
    print(f"Duplicate candles: {validation['duplicate_count']}")
    print(f"Missing candles: {validation['missing_count']}")
    print(f"Invalid rows: {validation['invalid_rows']}")


def cmd_inspect_csv(args: argparse.Namespace) -> None:
    ensure_project_structure()
    summary = inspect_training_csv(args.file)
    print(f"CSV: {summary['file']}")
    print(f"Rows: {summary['rows']}")
    if summary["rows"]:
        print(f"Range: {summary['first_timestamp']} to {summary['last_timestamp']}")
    else:
        print("Range: empty dataset")
    print(f"Columns: {', '.join(summary['columns'])}")


def cmd_import_csv(args: argparse.Namespace) -> None:
    ensure_project_structure()
    result = import_csv(
        args.file,
        symbol=args.symbol,
        interval=args.interval,
        output_path=args.output,
    )
    print(f"Saved {result['rows_written']} validated candles for {result['symbol']} ({result['interval']})")
    print(f"Training CSV: {result['output']}")
    print(f"Rows read: {result['rows_read']}")
    print(f"Duplicate rows skipped: {result['duplicate_rows']}")
    print(f"Invalid rows skipped: {result['invalid_rows']}")
    print(f"Missing intraday candles within sessions: {result['missing_candles']}")
    if result["errors_file"] is not None:
        print(f"Rejected rows log: {result['errors_file']}")


def cmd_features(args: argparse.Namespace) -> None:
    ensure_project_structure()
    output = write_features(
        args.input,
        style=args.style,
        output_path=args.output,
        box_window=args.box_window,
        relative_volume_window=args.relative_volume_window,
        opening_range_bars=args.opening_range_bars,
        market_index_path=args.market_index,
        volatility_index_path=args.volatility_index,
    )
    print(f"Saved {args.style} feature CSV to {output}")


def cmd_generate_signals(args: argparse.Namespace) -> None:
    ensure_project_structure()
    input_path = Path(args.input)
    if input_path.name.lower().endswith(".csv.gz"):
        source_name = input_path.name[:-7]
    else:
        source_name = input_path.stem
    symbol = source_name
    features = build_features(
        input_path,
        style=args.style,
        box_window=args.box_window,
        relative_volume_window=args.relative_volume_window,
        opening_range_bars=args.opening_range_bars,
    ).collect()
    if args.strategy == "all":
        signals = generate_all_signals(
            features,
            style=args.style,
            symbol=symbol,
        )
    elif args.strategy == "catalog-all":
        signals = generate_all_catalog_signals(
            features,
            style=args.style,
            symbol=symbol,
        )
    else:
        signals = generate_strategy_signals(
            features,
            strategy=args.strategy,
            style=args.style,
            symbol=symbol,
        )
    output = (
        Path(args.output)
        if args.output
        else DATA_DIR
        / "signals"
        / args.style
        / f"{source_name}_{args.strategy}_signals.csv"
    )
    if output.suffix.lower() != ".csv":
        raise ValueError(f"Signal output must be a CSV file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f"{output.name}.tmp")
    try:
        signals.write_csv(temporary_output)
        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    print(f"Generated {signals.height} {args.strategy} signal(s) to {output}")


def _write_backtest_outputs(
    result: BacktestResult,
    *,
    output_directory: Path,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, frame in (
        ("trades.csv", result.trades),
        ("equity.csv", result.equity_curve),
        ("yearly.csv", result.yearly),
    ):
        output = output_directory / filename
        temporary_output = output.with_name(f"{output.name}.tmp")
        try:
            frame.write_csv(temporary_output)
            temporary_output.replace(output)
        except Exception:
            temporary_output.unlink(missing_ok=True)
            raise

    summary = {
        key: value
        if not isinstance(value, float) or value not in (float("inf"), float("-inf"))
        else None
        for key, value in result.summary.items()
    }
    summary_output = output_directory / "summary.json"
    temporary_summary = summary_output.with_name(f"{summary_output.name}.tmp")
    try:
        temporary_summary.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_summary.replace(summary_output)
    except Exception:
        temporary_summary.unlink(missing_ok=True)
        raise
    return summary


def cmd_backtest(args: argparse.Namespace) -> None:
    ensure_project_structure()
    input_frames = []
    if args.strategy == "all":
        strategies_to_run = STRATEGIES
    elif args.strategy == "catalog-all":
        strategies_to_run = strategies_for_style(args.style)
    else:
        strategies_to_run = (args.strategy,)
    signals_by_strategy = {strategy: [] for strategy in strategies_to_run}
    for input_name in args.inputs:
        input_path = Path(input_name)
        source_name = (
            input_path.name[:-7]
            if input_path.name.lower().endswith(".csv.gz")
            else input_path.stem
        )
        symbol = input_path.parent.name
        if symbol.lower() in {"minute", "day", "intraday", "swing", "raw"}:
            symbol = source_name
        features = build_features(
            input_path,
            style=args.style,
            box_window=args.box_window,
            relative_volume_window=args.relative_volume_window,
            opening_range_bars=args.opening_range_bars,
        ).collect()
        if "symbol" not in features.columns:
            features = features.with_columns(pl.lit(symbol).alias("symbol"))
        input_frames.append(
            features.select("symbol", "timestamp", "open", "high", "low", "close")
        )
        for strategy in strategies_to_run:
            signals = generate_strategy_signals(
                features,
                strategy=strategy,
                style=args.style,
                symbol=symbol,
            )
            if args.hybrid_model_dir:
                hybrid_root = Path(args.hybrid_model_dir)
                strategy_model_dir = hybrid_root / strategy
                model_dir = (
                    strategy_model_dir
                    if (strategy_model_dir / "metadata.json").is_file()
                    else hybrid_root
                )
                signals = predict_hybrid_signals(
                    features,
                    signals,
                    model_dir=model_dir,
                    minimum_win_probability=args.minimum_win_probability,
                )
            signals_by_strategy[strategy].append(signals)

    bars = pl.concat(input_frames, how="vertical")
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_open_risk=args.max_open_risk,
        max_daily_loss=args.max_daily_loss,
        max_drawdown=args.max_drawdown,
        max_positions=args.max_positions,
        costs=ExecutionCosts(
            commission_bps=args.commission_bps,
            other_fees_bps=args.other_fees_bps,
            slippage_bps=args.slippage_bps,
        ),
    )
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else DATA_DIR / "backtests" / args.style
    )
    comparison = []
    for strategy in strategies_to_run:
        signals = pl.concat(signals_by_strategy[strategy], how="vertical")
        result = run_backtest(
            bars,
            signals,
            style=args.style,
            config=config,
        )
        output_directory = output_root / strategy
        summary = _write_backtest_outputs(result, output_directory=output_directory)
        comparison.append({"strategy": strategy, **summary})
        print(
            f"{strategy}: {summary['trade_count']} trades, "
            f"net return {summary['net_return_pct']:.2f}%, "
            f"max drawdown {summary['max_drawdown_pct']:.2f}% "
            f"-> {output_directory}"
        )

    comparison_path = output_root / "comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_comparison = comparison_path.with_name(f"{comparison_path.name}.tmp")
    try:
        pl.DataFrame(comparison).write_csv(temporary_comparison)
        temporary_comparison.replace(comparison_path)
    except Exception:
        temporary_comparison.unlink(missing_ok=True)
        raise
    print(f"Strategy comparison: {comparison_path}")


def _input_fingerprint(input_names: list[str]) -> list[dict[str, object]]:
    fingerprint = []
    for input_name in input_names:
        path = Path(input_name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training input does not exist: {path}")
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        fingerprint.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return fingerprint


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def cmd_train_models(args: argparse.Namespace) -> None:
    ensure_project_structure()
    required_packages = []
    if args.models in ("xgboost", "all"):
        required_packages.append("xgboost")
    if args.models in ("classical", "all"):
        required_packages.extend(("sklearn", "joblib"))
    if args.models in ("lstm", "all"):
        required_packages.extend(("torch", "psutil"))
    missing_packages = [
        package for package in required_packages if find_spec(package) is None
    ]
    if missing_packages:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing_packages)
            + ". Install requirements.txt and requirements-training.txt first."
        )
    if args.sequence_length < 1:
        raise ValueError("sequence_length must be at least one")
    if args.horizon is not None and args.horizon < 1:
        raise ValueError("horizon must be at least one")
    if min(
        args.xgboost_rounds,
        args.xgboost_patience,
        args.lstm_epochs,
        args.lstm_patience,
        args.batch_size,
        args.classical_trees,
    ) < 1:
        raise ValueError("Training rounds, patience, epochs, and batch size must be positive")
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else DATA_DIR / "models" / args.style / args.strategy
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training run directory: {run_dir}")
    control = TrainingControl(run_dir)
    manifest_path = run_dir / "run.json"
    manifest = {
        "format_version": 1,
        "inputs": _input_fingerprint(args.inputs),
        "style": args.style,
        "strategy": args.strategy,
        "horizon": args.horizon or (12 if args.style == "intraday" else 10),
        "sequence_length": args.sequence_length,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "commission_bps": args.commission_bps,
        "other_fees_bps": args.other_fees_bps,
        "slippage_bps": args.slippage_bps,
        "seed": args.seed,
        "xgboost_rounds": args.xgboost_rounds,
        "xgboost_patience": args.xgboost_patience,
        "lstm_epochs": args.lstm_epochs,
        "lstm_patience": args.lstm_patience,
        "batch_size": args.batch_size,
        "classical_trees": args.classical_trees,
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(
                f"Training run already exists at {run_dir}; pass --resume to continue "
                "from its latest checkpoints."
            )
        if existing_manifest != manifest:
            raise ValueError(
                "Resume settings or input files differ from the saved training run. "
                "Use the original inputs and parameters, or choose a new --run-dir."
            )
    elif args.resume:
        raise FileNotFoundError(f"No training run manifest exists at {manifest_path}")
    elif any(
        (run_dir / filename).exists()
        for filename in (
            "xgboost.checkpoint.ubj",
            "xgboost.state.json",
            "lstm.checkpoint.pt",
            "random_forest.checkpoint.joblib",
            "extra_trees.checkpoint.joblib",
            "xgboost.ubj",
            "lstm.pt",
        )
    ):
        raise FileExistsError(
            f"Checkpoint or model files already exist in {run_dir} without a run "
            "manifest; choose a fresh --run-dir or restore the matching run.json."
        )

    dataset = build_training_dataset(
        args.inputs,
        style=args.style,
        strategy=args.strategy,
        horizon=int(manifest["horizon"]),
        sequence_length=args.sequence_length,
        costs=ExecutionCosts(
            commission_bps=args.commission_bps,
            other_fees_bps=args.other_fees_bps,
            slippage_bps=args.slippage_bps,
        ),
    )
    split = chronological_split(
        dataset,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    mean = dataset.features[split.train].mean(axis=0).astype(np.float32)
    scale = dataset.features[split.train].std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    metadata = {
        "format_version": 1,
        "style": dataset.style,
        "strategy": dataset.strategy,
        "feature_names": list(dataset.feature_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "sequence_length": dataset.sequence_length,
        "horizon": dataset.horizon,
        "class_names": {
            "0": "loss",
            "1": "near_break_even",
            "2": "profitable",
        },
        "training_examples": len(split.train),
        "validation_examples": len(split.validation),
        "test_examples": len(split.test),
        "validation_start": split.validation_start.isoformat(),
        "test_start": split.test_start.isoformat(),
        "input_fingerprint": manifest["inputs"],
    }
    if manifest_path.exists():
        saved_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        if saved_metadata != metadata:
            raise ValueError(
                "Saved model metadata differs from the current input split or "
                "features; cannot resume safely."
            )
        if args.resume:
            control.request("continue")
    else:
        _write_json_atomic(run_dir / "metadata.json", metadata)
        _write_json_atomic(manifest_path, manifest)

    labels = dataset.labels
    class_counts = {
        "loss": int((labels == 0).sum()),
        "near_break_even": int((labels == 1).sum()),
        "profitable": int((labels == 2).sum()),
    }
    print(
        f"Prepared {len(labels)} labeled events "
        f"(train {len(split.train)}, validation {len(split.validation)}, "
        f"test {len(split.test)}); classes: {class_counts}"
    )
    if len(labels) < 500:
        print(
            "Warning: this is a small labeled dataset; treat model metrics as "
            "debugging output, not reliable evidence of predictive value."
        )
    try:
        model_results: dict[str, dict[str, np.ndarray]] = {}
        if args.models in ("xgboost", "all"):
            print("Training XGBoost on CPU.")
            model_results["xgboost"] = train_xgboost(
                dataset,
                split,
                run_dir=run_dir,
                control=control,
                max_rounds=args.xgboost_rounds,
                patience=args.xgboost_patience,
                seed=args.seed,
            )
            print(f"Saved XGBoost model to {run_dir / 'xgboost.ubj'}")

        if args.models in ("classical", "all"):
            print(
                "Training Random Forest, Extra Trees, and Logistic Regression "
                "on CPU."
            )
            model_results.update(
                train_classical_models(
                    dataset,
                    split,
                    run_dir=run_dir,
                    control=control,
                    trees=args.classical_trees,
                    patience=args.xgboost_patience,
                    seed=args.seed,
                )
            )

        if args.models in ("lstm", "all"):
            print("Training the small LSTM; CUDA is used automatically when available.")
            model_results["lstm"] = train_lstm(
                dataset,
                split,
                run_dir=run_dir,
                control=control,
                epochs=args.lstm_epochs,
                patience=args.lstm_patience,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            print(f"Saved LSTM model to {run_dir / 'lstm.pt'}")
    except TrainingStopped as exc:
        print(f"{exc}. Rerun the same command with --resume to continue.")
        return

    if args.models == "all":
        adx_index = (
            dataset.feature_names.index("adx_14")
            if "adx_14" in dataset.feature_names
            else None
        )
        atr_index = (
            dataset.feature_names.index("atr_pct")
            if "atr_pct" in dataset.feature_names
            else None
        )
        training_features = dataset.features[split.train]
        regime_thresholds = {
            "adx_trend_threshold": 25.0,
            "atr_pct_low_threshold": (
                float(np.nanpercentile(training_features[:, atr_index], 25))
                if atr_index is not None
                else None
            ),
            "atr_pct_high_threshold": (
                float(np.nanpercentile(training_features[:, atr_index], 75))
                if atr_index is not None
                else None
            ),
            "adx_feature": "adx_14" if adx_index is not None else None,
            "volatility_feature": "atr_pct" if atr_index is not None else None,
        }
        hybrid = calibrate_hybrid(
            labels=dataset.labels[split.validation],
            model_probabilities={
                name: result["validation_probabilities"]
                for name, result in model_results.items()
            },
            run_dir=run_dir,
            regime_thresholds=regime_thresholds,
        )
        blended_test = sum(
            weight * model_results[name]["test_probabilities"]
            for name, weight in hybrid["model_weights"].items()
        )
        from modeling.training import _probabilities_from_logits

        hybrid_test = _probabilities_from_logits(
            np.log(np.clip(blended_test, 1e-12, 1.0)),
            hybrid["temperature"],
        )
        lstm_result = model_results["lstm"]
        return_targets = np.clip(dataset.net_returns_r[split.test], -5.0, 5.0)
        return_errors = (
            lstm_result["test_expected_returns_r"] - return_targets
        )
        comparison = {
            "baseline_test": classification_metrics(
                dataset.labels[split.test],
                np.tile(
                    np.bincount(
                        dataset.labels[split.train],
                        minlength=3,
                    )
                    / len(split.train),
                    (len(split.test), 1),
                ),
            ),
            "hybrid_test": classification_metrics(
                dataset.labels[split.test],
                hybrid_test,
            ),
            "hybrid_validation": hybrid["validation_metrics"],
            "models_test": {
                name: classification_metrics(
                    dataset.labels[split.test],
                    result["test_probabilities"],
                )
                for name, result in model_results.items()
            },
            "lstm_test_expected_return_mae_r": float(
                np.abs(return_errors).mean()
            ),
            "lstm_test_expected_return_rmse_r": float(
                np.sqrt(np.square(return_errors).mean())
            ),
            "lstm_test_mean_return_uncertainty_r": float(
                lstm_result["test_return_uncertainty_r"].mean()
            ),
            "model_weights": hybrid["model_weights"],
        }
        _write_json_atomic(run_dir / "model-comparison.json", comparison)
        print(f"Calibrated hybrid model saved to {run_dir / 'hybrid.json'}")
        print(
            "Out-of-sample hybrid: "
            f"log loss {comparison['hybrid_test']['log_loss']:.4f}, "
            f"accuracy {comparison['hybrid_test']['accuracy_pct']:.1f}% "
            f"(majority baseline log loss {comparison['baseline_test']['log_loss']:.4f})"
        )
    print(f"Training artifacts: {run_dir}")


def cmd_training_control(args: argparse.Namespace) -> None:
    control = TrainingControl(args.run_dir)
    if args.action == "status":
        print(f"Training run {Path(args.run_dir)} status: {control.action()}")
        return
    action = {"resume": "continue"}.get(args.action, args.action)
    control.request(action)
    print(f"Requested {args.action} for training run {Path(args.run_dir)}")


def cmd_plan_training_batch(args: argparse.Namespace) -> None:
    ensure_project_structure()
    plan = build_training_batch_plan(
        stock_data_dir=args.stock_data_dir,
        run_root=args.run_root,
    )
    output_path = write_training_batch_plan(
        plan,
        output_path=args.output,
    )
    print(f"Saved training queue plan to {output_path}")
    print(f"Symbols: {len(plan['symbols'])}")
    print(f"Strategies: {plan['strategy_count']}")
    print(f"Intervals with available data: {plan['interval_count']}")
    print(f"Symbol/interval/strategy jobs: {plan['job_count']:,}")
    print(f"Model fits queued (XGBoost then LSTM): {plan['model_fit_count']:,}")
    print(f"Hybrid calibrations queued: {plan['hybrid_calibration_count']:,}")
    print(f"Sequential steps (including hybrid): {plan['total_steps']:,}")
    print(f"Stage order: {plan['execution_order']}")
    print("Training started: no")
    print(
        "This creates a compact matrix plan only; no model is trained and no "
        "per-job checkpoint artifacts are allocated."
    )


def cmd_train_batch(args: argparse.Namespace) -> None:
    ensure_project_structure()
    plan_path = Path(args.plan)
    if args.run_root:
        run_root = Path(args.run_root)
    elif plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        run_root = Path(plan["run_root"])
    else:
        raise FileNotFoundError(f"Training batch plan not found: {plan_path}")
    result = run_training_batch(
        plan_path=plan_path,
        run_root=run_root,
        resume=args.resume,
        max_jobs_per_stage=args.max_jobs_per_stage,
        allow_large_batch=args.allow_large_batch,
        sequence_length=args.sequence_length,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        commission_bps=args.commission_bps,
        other_fees_bps=args.other_fees_bps,
        slippage_bps=args.slippage_bps,
        seed=args.seed,
        xgboost_rounds=args.xgboost_rounds,
        xgboost_patience=args.xgboost_patience,
        lstm_epochs=args.lstm_epochs,
        lstm_patience=args.lstm_patience,
        batch_size=args.batch_size,
    )
    print(f"Batch status: {result['status']}")
    print(
        f"Next stage/job: {result['stage_index']}/"
        f"{result['job_index']}; failed jobs: {result['failed_jobs']}"
    )


def cmd_hybrid_signals(args: argparse.Namespace) -> None:
    ensure_project_structure()
    input_path = Path(args.input)
    source_name = (
        input_path.name[:-7]
        if input_path.name.lower().endswith(".csv.gz")
        else input_path.stem
    )
    symbol = input_path.parent.name
    if symbol.lower() in {"minute", "day", "intraday", "swing", "raw"}:
        symbol = source_name
    features = build_features(input_path, style=args.style).collect()
    if "symbol" not in features.columns:
        features = features.with_columns(pl.lit(symbol).alias("symbol"))
    signals = generate_strategy_signals(
        features,
        strategy=args.strategy,
        style=args.style,
        symbol=symbol,
    )
    scored = predict_hybrid_signals(
        features,
        signals,
        model_dir=args.model_dir,
        minimum_win_probability=args.minimum_win_probability,
    )
    output = (
        Path(args.output)
        if args.output
        else DATA_DIR
        / "signals"
        / "hybrid"
        / args.style
        / f"{source_name}_{args.strategy}_signals.csv"
    )
    if output.suffix.lower() != ".csv":
        raise ValueError(f"Hybrid signal output must be a CSV file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    try:
        scored.write_csv(temporary)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"Accepted {scored.height} AI-filtered {args.strategy} signal(s) "
        f"at win probability >= {args.minimum_win_probability:.2f}: {output}"
    )


def cmd_prepare_stock_data(args: argparse.Namespace) -> None:
    ensure_project_structure()
    symbols = (
        [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
        if args.symbols is not None
        else None
    )
    result = prepare_huggingface_dataset(
        repo_id=args.repo_id,
        split=args.split,
        symbols=symbols,
        output_root=args.output_root,
        download_dir=args.download_dir,
        keep_shards=args.keep_shards,
        batch_size=args.batch_size,
        holidays_csv=args.holidays_csv,
    )
    print(f"Processed shards: {result['shards']}")
    print(f"Processed symbols: {result['processed_symbols']}")
    print(f"Source rows processed: {result['source_rows']}")
    print(f"Output directory: {args.output_root}")
    if "minute_start_date" in result and "minute_end_date" in result:
        print(
            "Daily-source date range aligned to minute data: "
            f"{result['minute_start_date']} through {result['minute_end_date']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSV market-data and feature-engineering tools for the trading system."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and validate market data")
    download.add_argument("--symbol", required=True, help="Ticker symbol to download, for example INFY")
    download.add_argument("--interval", default="5m", help="OHLCV interval such as 5m or 1d")
    download.add_argument("--days", type=int, default=365, help="Lookback in days")
    download.set_defaults(func=cmd_download)

    inspect = subparsers.add_parser("inspect-csv", help="Inspect a CSV training dataset")
    inspect.add_argument("--file", required=True, help="Path to a normalized training CSV")
    inspect.set_defaults(func=cmd_inspect_csv)

    import_csv_parser = subparsers.add_parser(
        "import-csv",
        help="Validate and normalize a Kaggle OHLC or OHLCV CSV",
    )
    import_csv_parser.add_argument("--file", required=True, help="Path to an extracted Kaggle CSV file")
    import_csv_parser.add_argument("--symbol", help="Override the symbol inferred from the CSV filename")
    import_csv_parser.add_argument(
        "--interval",
        help="Override the interval inferred from the CSV filename (for example 1m, 5m, or 1d)",
    )
    import_csv_parser.add_argument(
        "--output",
        help="Destination for the normalized CSV (defaults under data\\raw)",
    )
    import_csv_parser.set_defaults(func=cmd_import_csv)

    features = subparsers.add_parser(
        "features",
        help="Generate causal intraday or swing features from a market CSV",
    )
    features.add_argument("--input", required=True, help="Path to a normalized OHLCV CSV")
    features.add_argument("--style", required=True, choices=("intraday", "swing"))
    features.add_argument("--output", help="Feature CSV destination (defaults under data\\features)")
    features.add_argument("--box-window", type=int, help="Prior bars used for box boundaries")
    features.add_argument(
        "--relative-volume-window",
        type=int,
        default=20,
        help="Prior bars used for the relative-volume baseline",
    )
    features.add_argument(
        "--opening-range-bars",
        type=int,
        default=3,
        help="Intraday opening-range length in candles",
    )
    features.add_argument(
        "--market-index",
        help="Optional index CSV used for strictly lagged market-context features",
    )
    features.add_argument(
        "--volatility-index",
        help="Optional VIX CSV used for strictly lagged market-context features",
    )
    features.set_defaults(func=cmd_features)

    signals = subparsers.add_parser(
        "generate-signals",
        help="Generate strategy entry, stop, and target signals from an OHLCV CSV",
    )
    signals.add_argument("--input", required=True, help="Input OHLCV CSV or CSV.GZ file")
    signals.add_argument("--style", required=True, choices=("intraday", "swing"))
    signals.add_argument(
        "--strategy",
        required=True,
        choices=(*STRATEGIES, *CATALOG_STRATEGIES, "all", "catalog-all"),
        metavar="STRATEGY",
        help="'all' runs the four core rules; 'catalog-all' runs every compatible catalog setup",
    )
    signals.add_argument("--output", help="Signal CSV path (defaults under data\\signals)")
    signals.add_argument("--box-window", type=int, help="Prior bars used for box boundaries")
    signals.add_argument(
        "--relative-volume-window",
        type=int,
        default=20,
        help="Prior bars used for the relative-volume baseline",
    )
    signals.add_argument(
        "--opening-range-bars",
        type=int,
        default=3,
        help="Intraday opening-range length in candles",
    )
    signals.set_defaults(func=cmd_generate_signals)

    backtest = subparsers.add_parser(
        "backtest",
        help="Simulate strategy signals with execution costs and portfolio risk limits",
    )
    backtest.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="OHLCV CSV/CSV.GZ input; repeat to backtest several symbols together",
    )
    backtest.add_argument("--style", required=True, choices=("intraday", "swing"))
    backtest.add_argument(
        "--strategy",
        required=True,
        choices=(*STRATEGIES, *CATALOG_STRATEGIES, "all", "catalog-all"),
        metavar="STRATEGY",
        help="'all' runs four core rules; 'catalog-all' benchmarks compatible catalog setups",
    )
    backtest.add_argument("--output-dir", help="Output directory (defaults under data\\backtests)")
    backtest.add_argument("--initial-capital", type=float, default=1_000_000.0)
    backtest.add_argument("--risk-per-trade", type=float, default=0.0025)
    backtest.add_argument("--max-open-risk", type=float, default=0.01)
    backtest.add_argument("--max-daily-loss", type=float, default=0.01)
    backtest.add_argument("--max-drawdown", type=float, default=0.05)
    backtest.add_argument("--max-positions", type=int)
    backtest.add_argument("--commission-bps", type=float, default=3.0)
    backtest.add_argument("--other-fees-bps", type=float, default=7.0)
    backtest.add_argument("--slippage-bps", type=float, default=5.0)
    backtest.add_argument("--box-window", type=int)
    backtest.add_argument("--relative-volume-window", type=int, default=20)
    backtest.add_argument("--opening-range-bars", type=int, default=3)
    backtest.add_argument(
        "--hybrid-model-dir",
        help=(
            "Optional trained hybrid model directory; for all/catalog-all, provide "
            "the parent containing one model directory per strategy"
        ),
    )
    backtest.add_argument("--minimum-win-probability", type=float, default=0.45)
    backtest.set_defaults(func=cmd_backtest)

    train_models = subparsers.add_parser(
        "train-models",
        help="Train XGBoost/LSTM models, calibrate the hybrid, and save checkpoints",
    )
    train_models.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="OHLCV CSV/CSV.GZ input; repeat for additional symbols",
    )
    train_models.add_argument("--style", required=True, choices=("intraday", "swing"))
    train_models.add_argument(
        "--strategy",
        required=True,
        choices=(*STRATEGIES, *CATALOG_STRATEGIES),
        metavar="STRATEGY",
    )
    train_models.add_argument(
        "--models",
        choices=("xgboost", "classical", "lstm", "all"),
        default="all",
        help=(
            "Model stage(s) to train sequentially; classical trains Random "
            "Forest, Extra Trees, and Logistic Regression"
        ),
    )
    train_models.add_argument(
        "--run-dir",
        help="Checkpoint/model directory (defaults under data\\models\\<style>\\<strategy>)",
    )
    train_models.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching existing run from its last checkpoint",
    )
    train_models.add_argument("--horizon", type=int, help="Forward outcome horizon in bars")
    train_models.add_argument("--sequence-length", type=int, default=60)
    train_models.add_argument("--validation-fraction", type=float, default=0.15)
    train_models.add_argument("--test-fraction", type=float, default=0.15)
    train_models.add_argument("--commission-bps", type=float, default=3.0)
    train_models.add_argument("--other-fees-bps", type=float, default=7.0)
    train_models.add_argument("--slippage-bps", type=float, default=5.0)
    train_models.add_argument("--seed", type=int, default=42)
    train_models.add_argument("--xgboost-rounds", type=int, default=500)
    train_models.add_argument("--xgboost-patience", type=int, default=30)
    train_models.add_argument("--classical-trees", type=int, default=120)
    train_models.add_argument("--lstm-epochs", type=int, default=40)
    train_models.add_argument("--lstm-patience", type=int, default=8)
    train_models.add_argument("--batch-size", type=int, default=32)
    train_models.set_defaults(func=cmd_train_models)

    training_control = subparsers.add_parser(
        "training-control",
        help="Pause, resume, or stop a training run at a checkpoint boundary",
    )
    training_control.add_argument(
        "action",
        choices=("pause", "resume", "stop", "status"),
        help="pause saves and waits; resume continues; stop saves and exits; status checks state",
    )
    training_control.add_argument("--run-dir", required=True)
    training_control.set_defaults(func=cmd_training_control)

    plan_training_batch = subparsers.add_parser(
        "plan-training-batch",
        help="Create a compact, non-executing all-symbol training queue plan",
    )
    plan_training_batch.add_argument(
        "--stock-data-dir",
        default="data\\stocks",
        help="Root containing minute and day per-symbol CSV directories",
    )
    plan_training_batch.add_argument(
        "--run-root",
        default="data\\models\\batch-runs",
        help="Reserved root for future per-symbol model artifacts",
    )
    plan_training_batch.add_argument(
        "--output",
        default="data\\models\\batch-plans\\all-symbols-all-intervals.json",
        help="New plan JSON path; existing files are never overwritten",
    )
    plan_training_batch.set_defaults(func=cmd_plan_training_batch)

    train_batch = subparsers.add_parser(
        "train-batch",
        help="Sequentially train all planned XGBoost/LSTM pairs and calibrate hybrids",
    )
    train_batch.add_argument(
        "--plan",
        default="data\\models\\batch-plans\\all-symbols-all-intervals.json",
        help="Plan generated by plan-training-batch",
    )
    train_batch.add_argument(
        "--run-root",
        help="Override the run root recorded in the plan",
    )
    train_batch.add_argument(
        "--resume",
        action="store_true",
        help="Resume at the saved stage/job and model checkpoint",
    )
    train_batch.add_argument(
        "--max-jobs-per-stage",
        type=int,
        help="Stop after this many jobs per stage; rerun with --resume to continue",
    )
    train_batch.add_argument(
        "--allow-large-batch",
        action="store_true",
        help="Required to execute a plan containing more than 10,000 jobs without a per-stage limit",
    )
    train_batch.add_argument("--sequence-length", type=int, default=60)
    train_batch.add_argument("--validation-fraction", type=float, default=0.15)
    train_batch.add_argument("--test-fraction", type=float, default=0.15)
    train_batch.add_argument("--commission-bps", type=float, default=3.0)
    train_batch.add_argument("--other-fees-bps", type=float, default=7.0)
    train_batch.add_argument("--slippage-bps", type=float, default=5.0)
    train_batch.add_argument("--seed", type=int, default=42)
    train_batch.add_argument("--xgboost-rounds", type=int, default=500)
    train_batch.add_argument("--xgboost-patience", type=int, default=30)
    train_batch.add_argument("--lstm-epochs", type=int, default=40)
    train_batch.add_argument("--lstm-patience", type=int, default=8)
    train_batch.add_argument("--batch-size", type=int, default=32)
    train_batch.set_defaults(func=cmd_train_batch)

    hybrid_signals = subparsers.add_parser(
        "hybrid-signals",
        help="Filter rule-based strategy candidates with a calibrated hybrid model",
    )
    hybrid_signals.add_argument("--input", required=True, help="OHLCV CSV or CSV.GZ file")
    hybrid_signals.add_argument("--style", required=True, choices=("intraday", "swing"))
    hybrid_signals.add_argument(
        "--strategy",
        required=True,
        choices=(*STRATEGIES, *CATALOG_STRATEGIES),
        metavar="STRATEGY",
    )
    hybrid_signals.add_argument("--model-dir", required=True, help="Trained model run directory")
    hybrid_signals.add_argument(
        "--minimum-win-probability",
        type=float,
        default=0.45,
        help="Minimum calibrated profitable-outcome probability for an accepted signal",
    )
    hybrid_signals.add_argument("--output", help="Accepted signal CSV destination")
    hybrid_signals.set_defaults(func=cmd_hybrid_signals)

    stock_data = subparsers.add_parser(
        "prepare-stock-data",
        help="Download Hugging Face stock Parquet shards and write per-symbol feature CSVs",
    )
    stock_data.add_argument(
        "--repo-id",
        default="Saintforest/indian-stock-market-minute-data",
        help="Hugging Face dataset repository ID",
    )
    stock_data.add_argument(
        "--split",
        choices=("minute", "day", "both"),
        default="both",
        help=(
            "minute creates 1m/5m/15m/20m/30m/1h plus 1d, 2d, and trading_week; "
            "day creates 1d, 2d, and trading_week"
        ),
    )
    stock_data.add_argument(
        "--symbols",
        help="Optional comma-separated NSE symbol list, for example RELIANCE,TCS,INFY",
    )
    stock_data.add_argument(
        "--holidays-csv",
        default=str(NSE_HOLIDAYS_CSV),
        help="CSV of official NSE closure dates (requires a YYYY-MM-DD `date` column)",
    )
    stock_data.add_argument(
        "--output-root",
        default="data\\stocks",
        help="Output root for per-symbol compressed CSVs",
    )
    stock_data.add_argument(
        "--download-dir",
        default="data\\downloads\\huggingface",
        help="Temporary/local directory for Hugging Face Parquet shards",
    )
    stock_data.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep each downloaded Parquet shard after it has been processed",
    )
    stock_data.add_argument(
        "--batch-size",
        type=int,
        default=65_536,
        help="Rows per streaming Parquet batch",
    )
    stock_data.set_defaults(func=cmd_prepare_stock_data)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
