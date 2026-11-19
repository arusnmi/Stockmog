import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from PIL import Image

from modeling.control import TrainingControl
from modeling.nifty_price_evaluation import _metrics, _sha
from modeling.nifty_price_report import ReportStopped, create_report


def evaluation_fixture(root):
    root.mkdir()
    frame = pd.DataFrame([
        ("A&B", "2025-01-03", 100.0, 101.0, 102.0, 100.0),
        ("A&B", "2025-01-02", 98.0, 100.0, 99.0, 98.0),
        ("A_B", "2025-01-02", 20.0, 21.0, 22.0, 20.0),
    ], columns=["symbol", "target_session", "anchor_close", "actual_close", "forecast_close", "baseline_close"])
    path = root / "holdout-predictions.csv"
    frame.to_csv(path, index=False)
    metrics = lambda group: {"model": _metrics(group, "forecast_close"),
                             "last_close_baseline": _metrics(group, "baseline_close")}
    summary = {"status": "complete", "research_only": True, "predictions_sha256": _sha(path),
               "metrics": metrics(frame), "stocks": {name: metrics(group) for name, group in frame.groupby("symbol")},
               "stocks_without_holdout": ["CCC"], "limitations": ["synthetic"]}
    (root / "holdout-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return frame, summary


class PriceReportTests(unittest.TestCase):
    def test_graph_command_from_completed_evaluation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_fixture(root / "evaluation")
            command = [sys.executable, "-m", "modeling.nifty_price_report",
                       "--evaluation", str(root / "evaluation"), "--output", str(root / "report")]
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue((root / "report" / "stock-graphs.pdf").is_file())

    def test_graphs_use_exact_csv_values_sorted_by_target_date_and_have_distinct_names(self):
        from matplotlib.axes import Axes
        original = Axes.plot
        lines = []

        def capture(ax, x, y, *args, **kwargs):
            lines.append((kwargs.get("label"), list(y), kwargs.get("marker")))
            return original(ax, x, y, *args, **kwargs)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_fixture(root / "evaluation")
            with patch.object(Axes, "plot", capture):
                result = create_report(root / "evaluation", root / "report")
            self.assertEqual(lines, [("Actual close", [100.0, 101.0], None),
                                     ("AI forecast", [99.0, 102.0], None),
                                     ("Actual close", [21.0], "o"), ("AI forecast", [22.0], "o")])
            self.assertEqual(len(result["stocks"]), 2)
            paths = [root / "report" / item["graph"] for item in result["stocks"]]
            self.assertNotEqual(paths[0], paths[1])
            for path in paths:
                with Image.open(path) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 100)
            self.assertEqual(len(pd.read_csv(root / "report" / "stock-metrics.csv")), 4)
            self.assertEqual(result["stocks_without_holdout"], ["CCC"])
            pdf = (root / "report" / result["graph_pdf"]).read_bytes()
            self.assertTrue(pdf.startswith(b"%PDF-"))
            self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", pdf)), 2)

    def test_stop_resume_and_detect_modified_graph_or_input(self):
        from modeling import nifty_price_report
        write = nifty_price_report._write_json
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_fixture(root / "evaluation")
            output = root / "report"

            def stop_after_first_graph(path, value):
                write(path, value)
                if path.parent.name == "graphs":
                    TrainingControl(output).request("stop")

            with patch.object(nifty_price_report, "_write_json", stop_after_first_graph):
                with self.assertRaises(ReportStopped):
                    create_report(root / "evaluation", output)
            checkpoint = next((output / "graphs").glob("*.png"))
            initial = _sha(checkpoint)
            create_report(root / "evaluation", output, resume=True)
            self.assertEqual(_sha(checkpoint), initial)
            with checkpoint.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                create_report(root / "evaluation", output, resume=True)
            with (root / "evaluation" / "holdout-predictions.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                create_report(root / "evaluation", root / "other")

    def test_rejects_mismatched_metrics_and_duplicate_dates(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame, summary = evaluation_fixture(root / "evaluation")
            summary["stocks"]["A&B"]["model"]["mae"] = 100
            (root / "evaluation" / "holdout-summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metrics"):
                create_report(root / "evaluation", root / "report")
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
            path = root / "evaluation" / "holdout-predictions.csv"
            frame.to_csv(path, index=False)
            summary["predictions_sha256"] = _sha(path)
            (root / "evaluation" / "holdout-summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                create_report(root / "evaluation", root / "report")


if __name__ == "__main__":
    unittest.main()
