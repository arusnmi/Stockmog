"""Build a detailed PDF from a saved daily forecast report."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.control import _replace_with_retry
from modeling.nifty_price_evaluation import _metrics, _sha


def report_stats(frame: pd.DataFrame) -> dict:
    model = _metrics(frame, "forecast_close")
    baseline = _metrics(frame, "anchor_close")
    actual = frame["actual_close"].to_numpy(dtype=float)
    forecast = frame["forecast_close"].to_numpy(dtype=float)
    error = (forecast - actual) / actual * 100
    return {"model": model, "baseline": baseline,
            "mean_signed_percent": float(np.mean(error)),
            "within_five_percent": float(np.mean(np.abs(error) <= 5) * 100)}


def _text_page(pdf, title: str, paragraphs: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.7, 8.3))
    try:
        fig.text(.06, .95, "RESEARCH REPORT  |  NOT INVESTMENT ADVICE", fontsize=9, color="#555555")
        fig.text(.06, .89, title, fontsize=22, weight="bold", color="#174878")
        y = .82
        for paragraph in paragraphs:
            lines = textwrap.wrap(paragraph, width=115, break_long_words=False,
                                  break_on_hyphens=False) or [""]
            fig.text(.06, y, "\n".join(lines), fontsize=11, va="top", linespacing=1.5)
            y -= .032 * len(lines) + .025
        pdf.savefig(fig)
    finally:
        plt.close(fig)


def _table_page(pdf, title: str, columns: list[str], rows: list[list[str]], note: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    try:
        ax.axis("off")
        fig.text(.06, .95, "RESEARCH REPORT  |  NOT INVESTMENT ADVICE", fontsize=9, color="#555555")
        fig.text(.06, .89, title, fontsize=19, weight="bold", color="#174878")
        table = ax.table(cellText=rows, colLabels=columns, loc="upper center", cellLoc="center",
                         bbox=[0, .08, 1, .77])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#174878")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f1f5f8")
        fig.text(.06, .06, "\n".join(textwrap.wrap(note, width=135)), fontsize=8, va="bottom")
        pdf.savefig(fig)
    finally:
        plt.close(fig)


def build_document(root: Path, stocks: list[dict], failures: list[dict], output: Path, plot_stock) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if output.exists():
        raise FileExistsError(f"Report PDF already exists: {output}")
    if not stocks:
        raise ValueError("No completed stock forecasts to report")
    frames = []
    seen = set()
    for stock in stocks:
        symbol = stock["symbol"]
        if symbol in seen:
            raise ValueError(f"Duplicate report symbol: {symbol}")
        seen.add(symbol)
        path = (root / stock["series"]).resolve()
        if not path.is_relative_to(root.resolve()) or _sha(path) != stock["series_sha256"]:
            raise ValueError(f"Saved prediction series changed or is outside the report: {symbol}")
        frame = pd.read_csv(path, dtype={"target_session": str})
        dates = pd.to_datetime(frame["target_session"], format="%Y-%m-%d", errors="raise")
        if (len(frame) != 20 or dates.isna().any() or dates.duplicated().any()
                or not dates.is_monotonic_increasing
                or frame["target_session"].iloc[0] != stock["first_target_session"]
                or frame["target_session"].iloc[-1] != stock["last_target_session"]):
            raise ValueError(f"Invalid saved prediction sessions for {symbol}")
        frames.append((stock, frame, report_stats(frame)))
    all_data = pd.concat([frame for _, frame, _ in frames], ignore_index=True)
    total = report_stats(all_data)
    first = min(frame["target_session"].iloc[0] for _, frame, _ in frames)
    last = max(frame["target_session"].iloc[-1] for _, frame, _ in frames)
    missing_stocks = sum((frame["missing_factor_count"] > 0).any() for _, frame, _ in frames)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with PdfPages(temporary) as pdf:
            _text_page(pdf, "Stock forecast retrospective", [
                f"{len(frames)} stocks | {first} to {last} | 20 completed sessions per stock",
                f"The trained daily models produced {len(all_data)} paired one-session-ahead historical "
                f"predictions. Model MAPE was {total['model']['mape_percent']:.2f}%; the previous-close "
                f"baseline was {total['baseline']['mape_percent']:.2f}%. Lower is better.",
                f"Across stock-sessions, {total['within_five_percent']:.2f}% of model forecasts were within "
                f"5% of the actual close. {missing_stocks} stocks had missing selected factors in at least one "
                f"decision row. {len(failures)} requested stocks could not be reported.",
                "The following pages contain the methodology, stock-by-stock metrics and charts, and the "
                "complete 20-session prediction/actual appendix.",
                "Prices and factors include unverified Yahoo extensions. Historical comparisons are not "
                "prospective performance or a trading recommendation.",
            ])
            _text_page(pdf, "Scope, data and methodology", [
                "Each prediction uses the saved stock's trained daily model, using features through the "
                "preceding session. Actual target closes are used only for comparison. Models are not retrained "
                "during this report; the target period starts no earlier than the model's saved holdout.",
                "Saved stock and selected factor history is retained; later observations are appended from "
                "Yahoo Finance. The price scales and ticker mappings have not been independently verified. "
                "A mismatched join can introduce false returns and distort indicators and predictions.",
                "MAPE is the mean absolute percentage error against actual close. The baseline uses the "
                "previous session's close as the forecast. Signed error is (forecast - actual) / actual; "
                "within 5% is the fraction with absolute percentage error at most 5%. Overall metrics "
                "weight each stock-session equally.",
                "Charts show actual and model-predicted closes with dashed linear trend fits over the displayed "
                "sessions. These trend fits use the full displayed period and are descriptive, not forecasts.",
                "Factor availability follows the saved workflow's lag and missing-value handling. A graph "
                "identifies selected factors and Yahoo extension status; missing counts refer to decision rows.",
            ])
            summary_rows = []
            for stock, frame, stats in frames:
                summary_rows.append([
                    stock["symbol"], str(len(frame)), f"{stats['model']['mape_percent']:.2f}%",
                    f"{stats['baseline']['mape_percent']:.2f}%",
                    f"{stats['mean_signed_percent']:+.2f}%", f"{stats['within_five_percent']:.2f}%",
                    str(int((frame["missing_factor_count"] > 0).sum())),
                ])
            for offset in range(0, len(summary_rows), 22):
                _table_page(pdf, f"Stock results | {offset + 1}-{min(offset + 22, len(summary_rows))}",
                            ["Symbol", "Pairs", "Model MAPE", "Baseline MAPE", "Signed error",
                             "Within 5%", "Missing rows"], summary_rows[offset:offset + 22],
                            "Baseline = previous close. Missing rows have at least one unavailable selected factor.")
            if failures:
                for offset in range(0, len(failures), 8):
                    _text_page(pdf, "Stocks not included", [
                        f"{item['symbol']}: {item['error']}" for item in failures[offset:offset + 8]])
            for stock, frame, stats in frames:
                fig = plot_stock(frame, stock["symbol"], stock["selected_factors"])
                try:
                    ax = fig.axes[0]
                    ax.set_title(
                        f"{stock['symbol']} | Model MAPE {stats['model']['mape_percent']:.2f}%  "
                        f"| Previous-close baseline {stats['baseline']['mape_percent']:.2f}%  "
                        f"| Within 5% {stats['within_five_percent']:.2f}%")
                    pdf.savefig(fig, bbox_inches="tight")
                finally:
                    plt.close(fig)
            for stock, frame, _ in frames:
                daily = []
                for row in frame.itertuples(index=False):
                    error = (row.forecast_close - row.actual_close) / row.actual_close * 100
                    daily.append([row.target_session, f"{row.anchor_close:.2f}",
                                  f"{row.forecast_close:.2f}", f"{row.actual_close:.2f}",
                                  f"{error:+.2f}%", row.predicted_trend,
                                  str(row.missing_factor_count)])
                _table_page(pdf, f"{stock['symbol']} | Daily results",
                            ["Target session", "Previous close", "Forecast", "Actual",
                             "Signed error", "Predicted trend", "Missing factors"], daily,
                            "Prices in INR. Trend is predicted close versus previous close; "
                            "missing factors are counted at the decision session.")
            _text_page(pdf, "Sources and reproducibility", [
                "Stock prices: the existing saved daily stock history, extended with Yahoo Finance daily "
                "candles after the saved history ends. Yahoo Finance: https://finance.yahoo.com/",
                "Selected factors: names, saved source identifiers, Yahoo ticker mappings and update status "
                "are printed on each stock's chart and stored in report.json. Indian index source snapshot: "
                "https://github.com/aswinv90/indian-market-historical-data",
                "Each report folder contains the exact per-stock series CSV, chart PNG and report.json. "
                "The PDF is derived from these saved CSVs and their recorded SHA-256 hashes.",
                "Sources, trading calendars, corporate actions and publication times are not independently "
                "verified. Results should not be used as investment advice.",
            ])
        _replace_with_retry(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
