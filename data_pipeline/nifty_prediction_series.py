"""Generate one-day-ahead rolling predictions through the latest complete session."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import _safe_symbol
from data_pipeline.providers import YahooFinanceProvider
from modeling.control import _replace_with_retry
from modeling.nifty_price_evaluation import _sha
from modeling.nifty_price_training import _price_lock, load_price_predictor
from modeling.nifty_timeframe_data import FactorStore, attach_factor, strategy_features
from modeling.nifty_timeframe_workflow import (
    DEFAULT_RUN, PROJECT, _job_dir, _load_job, _run_summary,
)
from modeling.nifty_universe import parse_nifty_membership
from modeling.nifty_preparation import available_candles


IST = ZoneInfo("Asia/Kolkata")
CORE = ["timestamp", "open", "high", "low", "close", "volume",
        "source_bars", "trading_days", "is_partial"]


def _session_day(value: object) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.date()
    return timestamp.tz_convert(IST).date()


def normalize_provider_rows(rows: list[dict], *, now: datetime | None = None) -> pd.DataFrame:
    if not rows:
        raise ValueError("Yahoo Finance returned no daily candles")
    current = datetime.now(IST) if now is None else now.astimezone(IST)
    frame = pd.DataFrame(rows)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Provider rows are missing columns: {sorted(required - set(frame.columns))}")
    days = pd.to_datetime(frame["timestamp"], errors="raise").map(_session_day)
    if days.duplicated().any():
        raise ValueError("Yahoo Finance returned duplicate daily sessions")
    complete_today = current.time() >= time(15, 45)
    eligible = days < current.date() if not complete_today else days <= current.date()
    frame = frame.loc[eligible].copy()
    days = days.loc[eligible]
    if frame.empty:
        raise ValueError("No complete daily candles are available yet")
    frame["timestamp"] = pd.to_datetime(days.astype(str)).dt.tz_localize(
        "Asia/Kolkata").dt.tz_convert("UTC")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    prices = frame[["open", "high", "low", "close"]]
    if (frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing
            or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()
            or not np.isfinite(frame["volume"]).all() or (frame["volume"] < 0).any()
            or (frame["high"] < prices[["open", "low", "close"]].max(axis=1)).any()
            or (frame["low"] > prices[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError("Yahoo Finance returned invalid or duplicate daily OHLCV candles")
    frame["source_bars"] = 1
    frame["trading_days"] = 1
    frame["is_partial"] = False
    return frame[CORE].reset_index(drop=True)


def _read_local(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=CORE)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if (frame["timestamp"].duplicated().any()
            or frame["timestamp"].map(_session_day).duplicated().any()):
        raise ValueError(f"Duplicate local daily candles: {path}")
    return frame[CORE]


def merge_daily_sources(local: pd.DataFrame, provider: pd.DataFrame, symbol: str) -> pd.DataFrame:
    local_days = local["timestamp"].map(_session_day)
    provider_days = provider["timestamp"].map(_session_day)
    last_local = max(local_days)
    addition = provider.loc[provider_days > last_local]
    result = pd.concat([local, addition], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate sessions after merging daily sources")
    return result[CORE]


def rolling_daily_predictions(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    reference: list[date],
    predictor,
    feature_names: list[str],
    sequence_length: int,
    start_target: date,
) -> pd.DataFrame:
    if sequence_length < 1 or not reference or reference != sorted(set(reference)):
        raise ValueError("Invalid sequence length or reference sessions")
    if (len(candles) != len(features)
            or not candles["timestamp"].reset_index(drop=True).equals(
                features["timestamp"].reset_index(drop=True))):
        raise ValueError("Prediction feature rows do not align with daily candles")
    if not set(feature_names).issubset(features.columns):
        raise ValueError("Latest feature rows do not match the trained model feature set")
    days = candles["timestamp"].dt.tz_convert(IST).dt.date.tolist()
    positions = {day: index for index, day in enumerate(reference)}
    matrix = features[feature_names].to_numpy(dtype=np.float32)
    if np.isinf(matrix).any():
        raise ValueError("Prediction features contain infinite values")
    records = []
    for decision_index in range(sequence_length - 1, len(candles) - 1):
        decision_day, target_day = days[decision_index:decision_index + 2]
        if target_day < start_target:
            continue
        if (decision_day not in positions or target_day not in positions
                or positions[target_day] != positions[decision_day] + 1):
            continue
        window_days = days[decision_index - sequence_length + 1:decision_index + 1]
        if any(positions.get(right, -2) != positions.get(left, -1) + 1
               for left, right in zip(window_days, window_days[1:])):
            continue
        window = matrix[decision_index - sequence_length + 1:decision_index + 1]
        flat = matrix[decision_index:decision_index + 1]
        anchor = np.asarray([float(candles["close"].iloc[decision_index])])
        forecast = np.asarray(predictor.predict(
            flat, window[None, :, :], anchor, feature_names=feature_names))
        if forecast.shape != (1,) or not np.isfinite(forecast).all() or forecast[0] <= 0:
            raise ValueError(f"Model produced an invalid forecast for {target_day}")
        price = float(forecast[0])
        actual = float(candles["close"].iloc[decision_index + 1])
        factor_values = [name for name in feature_names
                         if name.startswith("factor:") and not name.endswith(":missing")]
        missing_factors = sum(pd.isna(features[name].iloc[decision_index]) for name in factor_values)
        records.append({
            "decision_session": decision_day.isoformat(),
            "target_session": target_day.isoformat(),
            "available_at": features["available_at"].iloc[decision_index].isoformat(),
            "anchor_close": float(anchor[0]),
            "forecast_close": price,
            "actual_close": actual,
            "predicted_trend": "up" if price > anchor[0] else "down" if price < anchor[0] else "flat",
            "missing_factor_count": int(missing_factors),
        })
    if not records:
        raise ValueError(f"No consecutive daily predictions after {start_target}")
    result = pd.DataFrame(records)
    if result["target_session"].duplicated().any():
        raise ValueError("Duplicate target sessions in rolling prediction series")
    return result


def predict_series(
    symbol: str,
    *,
    run_dir: Path = DEFAULT_RUN,
    device: str = "cpu",
    now: datetime | None = None,
    sessions: int | None = None,
    provider: YahooFinanceProvider | None = None,
) -> dict:
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    if sessions is not None and sessions < 1:
        raise ValueError("Sessions must be positive")
    output = run_dir.resolve()
    with _price_lock(output):
        summary = _run_summary(output)
        if not any(job["symbol"] == symbol and job["timeframe"] == "1d"
                   for job in summary["jobs"]):
            raise ValueError(f"No prepared daily model job for {symbol}")
        folder = _job_dir(output, symbol, "1d")
        meta, _, _ = _load_job(folder)
        identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
        membership_path = Path(identity["config"]["membership"])
        membership = parse_nifty_membership(membership_path.read_text(encoding="utf-8-sig"))
        from modeling.nifty_universe import local_symbol

        local = local_symbol(symbol, membership)
        stock_root = Path(identity["config"]["stocks"])
        local_path = stock_root / "day" / _safe_symbol(local) / "1d.csv.gz"
        local_target = _read_local(local_path)

        current = datetime.now(IST) if now is None else now.astimezone(IST)
        end = current.date() + timedelta(days=1)
        provider = provider if provider is not None else YahooFinanceProvider()

        def fetch(ticker: str, local_frame: pd.DataFrame) -> pd.DataFrame:
            start = max(_session_day(value) for value in local_frame["timestamp"]) + timedelta(days=1)
            if start >= end:
                return local_frame
            rows = provider.fetch_history(ticker, start, end, interval="1d")
            return merge_daily_sources(local_frame, normalize_provider_rows(rows, now=current), ticker)

        ticker_symbol = symbol if symbol == "LTM" else local
        ticker = ticker_symbol if ticker_symbol.endswith(".NS") else ticker_symbol + ".NS"
        target_raw = fetch(ticker, local_target)
        reference_path = stock_root / "day" / "RELIANCE" / "1d.csv.gz"
        if local == "RELIANCE":
            reference_raw = target_raw
        else:
            reference_raw = fetch("RELIANCE.NS", _read_local(reference_path))
        target_last = _session_day(target_raw["timestamp"].iloc[-1])
        reference_last = _session_day(reference_raw["timestamp"].iloc[-1])
        if target_last != reference_last:
            raise ValueError(
                f"{symbol} data ends {target_last}, but the latest reference session is {reference_last}; "
                "cannot claim predictions through the last complete market session"
            )
        reference = [_session_day(value) for value in reference_raw["timestamp"]]
        candles = available_candles(target_raw, "1d", daily_source=True)
        features = strategy_features(candles, "1d")

        selected = meta["selection"]["selected"]
        factor_updates = []
        if selected:
            from data_pipeline.nifty_factor_updates import extend_factor

            factors = FactorStore(identity["config"], reference)
            for item in selected:
                key = (item["source"], item["factor"])
                _, values = factors.load(key, "1d")
                if values is None:
                    raise ValueError(f"Selected factor is unavailable for daily prediction: {key}")
                values, update = extend_factor(
                    key, values, factors, identity["config"], stock_root, reference, provider, current)
                factor_updates.append(update)
                features = attach_factor(
                    features, values, "factor:" + json.dumps(key), "1d")
        model = folder / "model"
        training = json.loads((model / "training-summary.json").read_text(encoding="utf-8"))
        if training["dataset_sha256"] != meta["dataset_sha256"]:
            raise ValueError("Model dataset differs from prepared data")
        predictor = load_price_predictor(model, device=device)
        start_target = date.fromisoformat(meta["holdout_start"])
        if sessions is not None:
            if len(reference) < sessions:
                raise ValueError(f"{symbol}: fewer than {sessions} reference sessions available")
            start_target = max(start_target, reference[-sessions])
        predictions = rolling_daily_predictions(
            candles, features, reference, predictor, meta["feature_names"],
            meta["sequence_length"], start_target)
        if sessions is not None:
            expected = [day.isoformat() for day in reference[-sessions:]]
            if predictions["target_session"].tolist() != expected:
                raise ValueError(f"{symbol}: missing predictions for one or more of the latest "
                                 f"{sessions} reference sessions")
        suffix = "" if sessions is None else f"-{sessions}"
        csv_path = folder / f"prediction-series{suffix}.csv"
        temporary = csv_path.with_suffix(".tmp")
        predictions.to_csv(temporary, index=False)
        _replace_with_retry(temporary, csv_path)
        report = {
            "status": "complete", "research_only": True, "symbol": symbol, "timeframe": "1d",
            "source": "Yahoo Finance via the repository's YahooFinanceProvider",
            "first_target_session": predictions["target_session"].iloc[0],
            "last_complete_trading_day": target_last.isoformat(),
            "predictions": len(predictions), "csv": str(csv_path),
            "csv_sha256": _sha(csv_path),
            "prepared_sha256": _sha(folder / "prepared.json"),
            "model_summary_sha256": _sha(model / "training-summary.json"),
            "selected_factors": len(selected),
            "factor_updates": factor_updates,
            "missing_factor_prediction_rows": int((predictions["missing_factor_count"] > 0).sum()),
            "warning": "Yahoo extensions are not verified against saved history; differing adjustments "
                       "can cause false jumps. Rolling historical forecasts are research only.",
        }
        report_path = folder / f"prediction-series{suffix}-summary.json"
        report_temp = report_path.with_suffix(".tmp")
        report_temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _replace_with_retry(report_temp, report_path)
        return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0].lower() == "predict-series":
        arguments = arguments[1:]
    args = parser.parse_args(arguments)
    print(json.dumps(predict_series(args.symbol, run_dir=args.run_dir, device=args.device),
                     indent=2))


if __name__ == "__main__":
    main()
