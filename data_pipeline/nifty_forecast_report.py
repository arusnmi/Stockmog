"""Generate a 20-session historical forecast report for trained daily models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import _safe_symbol
from data_pipeline.nifty_prediction_series import IST, predict_series
from data_pipeline.providers import YahooFinanceProvider
from modeling.factor_research import _write_json
from modeling.nifty_price_evaluation import _metrics, _sha
from modeling.nifty_price_training import _price_lock
from modeling.nifty_timeframe_workflow import DEFAULT_RUN, PROJECT, _job_dir, _run_summary


SESSIONS = 20


class ReportProvider(YahooFinanceProvider):
    def __init__(self) -> None:
        self.cache: dict[tuple, list[dict]] = {}

    def fetch_history(self, symbol, start, end, interval="1d"):
        key = (symbol, start, end, interval)
        if key not in self.cache:
            self.cache[key] = super().fetch_history(symbol, start, end, interval)
        return self.cache[key]


def _read_result(run: Path, symbol: str, series: dict) -> tuple[pd.DataFrame, list[dict]]:
    folder = _job_dir(run, symbol, "1d")
    csv_path = folder / f"prediction-series-{SESSIONS}.csv"
    with _price_lock(run):
        if (series["csv_sha256"] != _sha(csv_path)
                or series["prepared_sha256"] != _sha(folder / "prepared.json")
                or series["model_summary_sha256"] != _sha(folder / "model" / "training-summary.json")):
            raise ValueError(f"{symbol}: prediction data or model metadata changed during report generation")
        meta = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path)
    if (series["symbol"] != symbol or series["timeframe"] != "1d"
            or series["predictions"] != SESSIONS or len(frame) != SESSIONS):
        raise ValueError(f"{symbol}: invalid prediction series metadata or row count")
    required = {"target_session", "anchor_close", "actual_close", "forecast_close",
                "predicted_trend", "missing_factor_count"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{symbol}: prediction CSV is missing required columns")
    dates = pd.to_datetime(frame["target_session"], format="%Y-%m-%d", errors="raise")
    if (dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing
            or frame["target_session"].iloc[-1] != series["last_complete_trading_day"]):
        raise ValueError(f"{symbol}: prediction dates are invalid or incomplete")
    _metrics(frame, "forecast_close")
    if (frame["missing_factor_count"].isna().any()
            or (frame["missing_factor_count"] < 0).any()):
        raise ValueError(f"{symbol}: invalid factor missing counts")
    selected = meta["selection"]["selected"]
    if len(selected) != series["selected_factors"]:
        raise ValueError(f"{symbol}: selected factors differ from prediction metadata")
    updates = {(item["source"], item["name"]): item for item in series["factor_updates"]}
    if len(updates) != len(selected):
        raise ValueError(f"{symbol}: factor update metadata is incomplete")
    return frame, [{"source": item["source"], "name": item["factor"],
                    "update": updates[(item["source"], item["factor"])]} for item in selected]


def _plot_stock(frame: pd.DataFrame, symbol: str, factors: list[dict]):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 1, height_ratios=(4, 1.4))
    ax = fig.add_subplot(grid[0])
    notes = fig.add_subplot(grid[1])
    dates = pd.to_datetime(frame["target_session"])
    x = np.arange(len(frame))
    for column, label, color in (("actual_close", "Actual close", "tab:blue"),
                                 ("forecast_close", "Predicted close", "tab:orange")):
        values = frame[column].to_numpy(dtype=float)
        ax.plot(dates, values, label=label, color=color, marker="o", markersize=3)
        trend = np.polyval(np.polyfit(x, values, 1), x)
        ax.plot(dates, trend, label=f"{label} trend (linear fit)", color=color, linestyle="--")
    ax.set(title=f"{symbol}: latest {SESSIONS} completed sessions",
           xlabel="Target session", ylabel="Closing price (INR)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis="x", labelrotation=30)
    def label(item):
        update = item.get("update")
        if update is None:
            return f"{item['name']} [{item['source']}]"
        ticker = update.get("yahoo_ticker") or "no Yahoo mapping"
        return f"{item['name']} [{item['source']}; {ticker}: {update['status']}]"

    external = [label(item) for item in factors if item["source"] != "local"]
    local = [label(item) for item in factors if item["source"] == "local"]
    factor_text = "External factors: " + ("; ".join(external) if external else "None selected")
    local_text = "Local stock/index factors: " + ("; ".join(local) if local else "None selected")
    missing = int((frame["missing_factor_count"] > 0).sum())
    text = "\n".join((textwrap.fill(factor_text, width=125),
                      textwrap.fill(local_text, width=125),
                      f"Sessions with missing selected factor values: {missing}/{len(frame)}",
                      "Unverified Yahoo joins may introduce false price jumps or factor returns.",
                      "Trend lines are retrospective linear fits, not additional forecasts."))
    notes.axis("off")
    notes.text(0, 0.96, text, va="top", fontsize=8, transform=notes.transAxes, wrap=True)
    fig.tight_layout()
    return fig


def create_report(run: Path, output: Path, *, symbols: list[str] | None = None,
                  device: str = "cpu") -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Graphing requires matplotlib from requirements-forecast.txt") from exc

    run, output = run.resolve(), output.resolve()
    summary = _run_summary(run)
    available = {job["symbol"] for job in summary["jobs"] if job["timeframe"] == "1d"}
    requested = sorted(available if symbols is None else set(symbols))
    if not requested:
        raise ValueError("No daily models requested or available")
    unknown = set(requested) - available
    if unknown:
        raise ValueError(f"No prepared daily model job for: {', '.join(sorted(unknown))}")
    output.mkdir(parents=True, exist_ok=False)
    graphs, series_dir = output / "graphs", output / "series"
    graphs.mkdir()
    series_dir.mkdir()
    provider = ReportProvider()
    now = datetime.now(IST)
    stocks, failures = [], []
    pdf_path = output / "stock_forecast_report.pdf"
    for index, symbol in enumerate(requested, 1):
        print(f"Generating {symbol} ({index}/{len(requested)})", file=sys.stderr, flush=True)
        try:
            series = predict_series(symbol, run_dir=run, device=device, now=now,
                                    sessions=SESSIONS, provider=provider)
            frame, factors = _read_result(run, symbol, series)
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            failure = {"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"FAILED {symbol}: {failure['error']}", file=sys.stderr, flush=True)
            continue
        name = f"{_safe_symbol(symbol)}-{hashlib.sha256(symbol.encode()).hexdigest()[:10]}"
        graph = graphs / f"{name}.png"
        csv = series_dir / f"{name}.csv"
        frame.to_csv(csv, index=False)
        fig = _plot_stock(frame, symbol, factors)
        try:
            fig.savefig(graph, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        stocks.append({
            "symbol": symbol, "graph": str(graph.relative_to(output)),
            "series": str(csv.relative_to(output)), "series_sha256": _sha(csv),
            "first_target_session": frame["target_session"].iloc[0],
            "last_target_session": frame["target_session"].iloc[-1],
            "selected_factors": factors,
            "missing_factor_prediction_rows": series["missing_factor_prediction_rows"],
            "metrics": _metrics(frame, "forecast_close"),
        })
    if stocks:
        from data_pipeline.nifty_report_document import build_document

        build_document(output, stocks, failures, pdf_path, _plot_stock)
    result = {
        "status": "complete" if not failures else "partial" if stocks else "failed",
        "research_only": True, "run": str(run), "sessions_per_stock": SESSIONS,
        "graph_pdf": str(pdf_path) if stocks else None,
        "stocks": stocks, "failures": failures,
        "limitations": [
            "Historical one-day-ahead predictions, not forecasts of future sessions.",
            "Trend lines are retrospective linear fits over the displayed sessions.",
            "Factor observations may be stale or missing; missing counts refer to decision rows.",
            "Yahoo extensions are not checked against saved history; differing adjustments may cause false jumps.",
            "Yahoo Finance is not exchange-verified and may not provide the latest market session.",
        ],
    }
    _write_json(output / "report.json", result)
    return result


def render_saved_report(folder: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    from data_pipeline.nifty_report_document import build_document

    folder = folder.resolve()
    saved = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if saved.get("status") not in {"complete", "partial"} or saved.get("research_only") is not True:
        raise ValueError("A completed or partial research report is required")
    pdf_path = folder / "stock_forecast_report.pdf"
    build_document(folder, saved["stocks"], saved["failures"], pdf_path, _plot_stock)
    return pdf_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--symbols", nargs="+", help="Limit the report to these symbols")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--from-report", type=Path,
                        help="Build a detailed PDF from a saved forecast-report folder without downloading")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0].lower() == "forecast-report":
        arguments = arguments[1:]
    args = parser.parse_args(arguments)
    if args.from_report is not None:
        if args.output is not None or args.symbols is not None or args.device != "cpu":
            parser.error("--from-report cannot be combined with --output, --symbols, or --device")
        print(json.dumps({"pdf": str(render_saved_report(args.from_report))}, indent=2))
        return
    output = args.output or (PROJECT / "stock_forecasts" /
                             datetime.now(IST).strftime("xgboost-lstm-external-factors-%Y%m%d-%H%M%S-%f"))
    result = create_report(args.run_dir, output, symbols=args.symbols, device=args.device)
    print(json.dumps({"status": result["status"], "report": str(output.resolve() / "report.json"),
                      "pdf": result["graph_pdf"], "stocks": len(result["stocks"]),
                      "failures": result["failures"]}, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
