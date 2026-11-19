import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_pipeline.nifty_forecast_report import _plot_stock, create_report, render_saved_report
from data_pipeline.nifty_report_document import report_stats


def report_frame():
    actual = np.arange(20, dtype=float) + 100
    return pd.DataFrame({
        "target_session": pd.bdate_range("2026-01-02", periods=20).strftime("%Y-%m-%d"),
        "anchor_close": actual - 1,
        "actual_close": actual,
        "forecast_close": actual + 2,
        "predicted_trend": ["up"] * 20,
        "missing_factor_count": [1] * 20,
    })


class ForecastReportTests(unittest.TestCase):
    def test_graph_uses_actual_predicted_trends_and_factor_references(self):
        import matplotlib.pyplot as plt

        frame = report_frame()
        factors = [{"name": "NIFTY_50:Close", "source": "india:provider"},
                   {"name": "OTHER", "source": "local"}]
        fig = _plot_stock(frame, "TEST", factors)
        try:
            lines = fig.axes[0].lines
            self.assertEqual(len(lines), 4)
            np.testing.assert_array_equal(lines[0].get_ydata(), frame["actual_close"])
            np.testing.assert_array_equal(lines[2].get_ydata(), frame["forecast_close"])
            np.testing.assert_allclose(lines[1].get_ydata(), frame["actual_close"])
            np.testing.assert_allclose(lines[3].get_ydata(), frame["forecast_close"])
            text = fig.axes[1].texts[0].get_text()
            self.assertIn("NIFTY_50:Close [india:provider]", text)
            self.assertIn("OTHER", text)
            self.assertIn("20/20", text)
        finally:
            plt.close(fig)

    def test_partial_report_keeps_successes_and_records_failure(self):
        summary = {"jobs": [{"symbol": "AAA", "timeframe": "1d"},
                            {"symbol": "BBB", "timeframe": "1d"}]}
        frame = report_frame()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("data_pipeline.nifty_forecast_report._run_summary", return_value=summary), \
                 patch("data_pipeline.nifty_forecast_report.predict_series",
                       side_effect=[{"missing_factor_prediction_rows": 20},
                                    ValueError("provider unavailable")]), \
                 patch("data_pipeline.nifty_forecast_report._read_result",
                       return_value=(frame, [])):
                result = create_report(root, root / "report")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["failures"][0]["symbol"], "BBB")
            self.assertIn("provider unavailable", result["failures"][0]["error"])
            self.assertEqual(len(result["stocks"]), 1)
            self.assertTrue((root / "report" / result["stocks"][0]["graph"]).is_file())
            self.assertTrue((root / "report" / "stock_forecast_report.pdf").is_file())
            self.assertEqual(len(pd.read_csv(root / "report" / result["stocks"][0]["series"])), 20)
            self.assertEqual(json.loads((root / "report" / "report.json").read_text())["status"],
                             "partial")

    def test_summary_metrics_compare_model_to_previous_close(self):
        frame = report_frame()
        stats = report_stats(frame)
        expected = ((frame["forecast_close"] - frame["actual_close"]).abs()
                    / frame["actual_close"]).mean() * 100
        baseline = ((frame["anchor_close"] - frame["actual_close"]).abs()
                    / frame["actual_close"]).mean() * 100
        self.assertAlmostEqual(stats["model"]["mape_percent"], expected)
        self.assertAlmostEqual(stats["baseline"]["mape_percent"], baseline)
        self.assertAlmostEqual(stats["mean_signed_percent"], expected)
        self.assertEqual(stats["within_five_percent"], 100)

    def test_saved_report_rejects_changed_series(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "series.csv").write_text("changed", encoding="utf-8")
            (root / "report.json").write_text(json.dumps({
                "status": "complete", "research_only": True, "failures": [],
                "stocks": [{"symbol": "TEST", "series": "series.csv", "series_sha256": "wrong"}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction series changed"):
                render_saved_report(root)
            self.assertFalse((root / "stock_forecast_report.pdf").exists())


if __name__ == "__main__":
    unittest.main()
