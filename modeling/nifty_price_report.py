"""Generate per-stock holdout price comparison graphs and metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

from data_pipeline.huggingface_stocks import _safe_symbol
from modeling.control import TrainingControl, _replace_with_retry
from modeling.factor_research import _write_json
from modeling.nifty_price_evaluation import _metrics, _overlap, _sha
from modeling.nifty_price_training import _price_lock


class ReportStopped(RuntimeError):
    """Report generation stopped after a completed stock."""


def create_report(evaluation: Path, output: Path, *, resume: bool = False) -> dict:
    evaluation, output = evaluation.resolve(), output.resolve()
    if _overlap(evaluation, output):
        raise ValueError("Evaluation and report directories must not overlap")
    if not evaluation.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError("Graphing requires matplotlib from requirements-forecast.txt") from exc
    with _price_lock(evaluation):
        summary_path = evaluation / "holdout-summary.json"
        predictions_path = evaluation / "holdout-predictions.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or summary.get("research_only") is not True:
            raise ValueError("A completed research holdout evaluation is required")
        if _sha(predictions_path) != summary["predictions_sha256"]:
            raise ValueError("Holdout prediction CSV has changed")
        source_hashes = [_sha(summary_path), _sha(predictions_path)]
        identity = {"version": 1, "evaluation": str(evaluation), "source_sha256": source_hashes,
                    "matplotlib_version": matplotlib.__version__}
        frame = pd.read_csv(predictions_path, dtype={"symbol": str, "target_session": str},
                            float_precision="round_trip")
        required = {"symbol", "target_session", "anchor_close", "actual_close", "forecast_close", "baseline_close"}
        if not required.issubset(frame.columns) or frame.empty or frame["symbol"].isna().any():
            raise ValueError("Holdout CSV is empty or missing required columns")
        dates = pd.to_datetime(frame["target_session"], format="%Y-%m-%d", errors="raise")
        if dates.isna().any() or frame.duplicated(["symbol", "target_session"]).any():
            raise ValueError("Holdout target dates must be unique for each stock")
        frame["target_session"] = dates
        if set(frame["symbol"]) != set(summary["stocks"]):
            raise ValueError("Holdout stocks do not match evaluation summary")
        for symbol, group in frame.groupby("symbol", sort=True):
            for kind, column in (("model", "forecast_close"), ("last_close_baseline", "baseline_close")):
                actual = _metrics(group, column)
                expected = summary["stocks"][symbol][kind]
                if (actual["count"] != expected["count"] or any(
                    not math.isclose(actual[key], expected[key], rel_tol=1e-10, abs_tol=1e-10)
                    for key in ("mae", "rmse", "mape_percent", "directional_accuracy_percent")
                )):
                    raise ValueError(f"Holdout metrics do not match CSV for {symbol}")
        if output.exists():
            if not resume:
                raise FileExistsError(f"Report output already exists: {output}")
        else:
            if resume:
                raise FileNotFoundError(f"Report output does not exist: {output}")
            output.mkdir(parents=True)
        with _price_lock(output):
            manifest = output / "identity.json"
            if resume:
                if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
                    raise ValueError("Report inputs or settings changed; use a new directory")
            else:
                _write_json(manifest, identity)
            control = TrainingControl(output)
            if resume:
                control.request("continue")
            graphs = output / "graphs"
            graphs.mkdir(exist_ok=True)
            items = []
            for symbol, group in frame.groupby("symbol", sort=True):
                if not control.checkpoint_boundary(lambda: None):
                    raise ReportStopped("Report stopped; rerun with --resume")
                filename = f"{_safe_symbol(symbol)}-{hashlib.sha256(symbol.encode()).hexdigest()[:10]}.png"
                target = graphs / filename
                checkpoint = target.with_suffix(".json")
                if checkpoint.is_file():
                    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if not target.is_file() or _sha(target) != saved["sha256"]:
                        raise ValueError(f"Graph checkpoint is missing or corrupt: {target}")
                else:
                    group = group.sort_values("target_session")
                    fig, ax = plt.subplots(figsize=(11, 5))
                    try:
                        marker = "o" if len(group) == 1 else None
                        ax.plot(group["target_session"], group["actual_close"], label="Actual close", marker=marker)
                        ax.plot(group["target_session"], group["forecast_close"], label="AI forecast", marker=marker)
                        ax.set(title=f"{symbol}: holdout daily close", xlabel="Target session",
                               ylabel="Price (INR)")
                        ax.grid(alpha=0.25)
                        ax.legend()
                        fig.autofmt_xdate()
                        temporary = graphs / f"{filename}.tmp.png"
                        fig.savefig(temporary, dpi=150, bbox_inches="tight")
                        _replace_with_retry(temporary, target)
                    finally:
                        plt.close(fig)
                    _write_json(checkpoint, {"sha256": _sha(target)})
                items.append({"symbol": symbol, "graph": str(Path("graphs") / filename),
                              **summary["stocks"][symbol]})
            if [_sha(summary_path), _sha(predictions_path)] != source_hashes:
                raise ValueError("Holdout evaluation changed during report generation")
            pdf_path = output / "stock-graphs.pdf"
            temporary_pdf = output / "stock-graphs.pdf.tmp"
            try:
                with PdfPages(temporary_pdf) as pdf:
                    for item in items:
                        if not control.checkpoint_boundary(lambda: None):
                            raise ReportStopped("Report stopped; rerun with --resume")
                        fig, ax = plt.subplots(figsize=(11, 6))
                        try:
                            ax.imshow(plt.imread(output / item["graph"]))
                            ax.axis("off")
                            pdf.savefig(fig, bbox_inches="tight")
                        finally:
                            plt.close(fig)
                _replace_with_retry(temporary_pdf, pdf_path)
            finally:
                if temporary_pdf.exists():
                    temporary_pdf.unlink()
            metrics_path = output / "stock-metrics.csv"
            temporary = output / "stock-metrics.csv.tmp"
            records = []
            for item in items:
                for kind in ("model", "last_close_baseline"):
                    records.append({"symbol": item["symbol"], "method": kind, **item[kind]})
            pd.DataFrame(records).to_csv(temporary, index=False)
            _replace_with_retry(temporary, metrics_path)
            if [_sha(summary_path), _sha(predictions_path)] != source_hashes:
                raise ValueError("Holdout evaluation changed during report generation")
            result = {"status": "complete", "research_only": True, "evaluation": str(evaluation),
                      "graph_pdf": pdf_path.name, "graph_pdf_sha256": _sha(pdf_path),
                      "overall": summary["metrics"], "stocks": items,
                      "stocks_without_holdout": summary["stocks_without_holdout"],
                      "limitations": summary["limitations"]}
            _write_json(output / "report.json", result)
            return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = create_report(args.evaluation, args.output, resume=args.resume)
        print(f"Generated {len(result['stocks'])} stock graphs in {args.output}")
    except ReportStopped as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
