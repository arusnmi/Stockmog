import hashlib
import json
import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from modeling.nifty_preparation import TIMEFRAMES
from modeling.nifty_price_model import (
    build_examples, build_forecast_features, build_price_lstm, build_price_tree, combine_price_predictions,
    load_prepared_candles, predict_hybrid_prices,
)
from modeling.nifty_universe import Membership


def bars(timestamps, availability, closes, partial=False):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "available_at": pd.to_datetime(availability, utc=True),
        "open": closes, "high": np.array(closes) + 1, "low": np.array(closes) - 1,
        "close": closes, "volume": 10.0, "is_partial": partial,
    })


class NiftyPriceModelTests(unittest.TestCase):
    def setUp(self):
        self.days = [date(2022, 1, d) for d in (3, 4, 5, 6)]
        self.daily = bars(
            ["2022-01-02T18:30Z", "2022-01-03T18:30Z", "2022-01-04T18:30Z",
             "2022-01-05T18:30Z"],
            ["2022-01-03T18:30Z", "2022-01-04T18:30Z", "2022-01-05T18:30Z",
             "2022-01-06T18:30Z"], [100, 110, 120, 130])
        self.frame = bars(
            ["2022-01-03T09:00Z", "2022-01-04T09:00Z"],
            ["2022-01-03T10:00Z", "2022-01-04T10:00Z"], [105, 115])
        self.candles = {tf: self.frame.copy() for tf in TIMEFRAMES}
        self.ages = {tf: timedelta(days=3) for tf in TIMEFRAMES}
        self.members = (Membership("AAA", self.days[0], None, "test", ""),)
        self.selection = {"status": "research_only", "threshold_absolute": 0.7,
                          "train_end_exclusive": "2022-01-03",
                          "stocks": [{"symbol": "AAA", "status": "selected", "selected": [
                              {"source": "local", "factor": "F", "correlation": -0.7}]}]}
        self.factors = {("local", "F"): pd.DataFrame({
            "factor_available_at": pd.to_datetime(["2022-01-03T18:30Z", "2022-01-03T19:00Z"], utc=True),
            "factor_return": [0.1, 0.9]})}

    def make(self, **overrides):
        values = dict(symbol="AAA", daily=self.daily, candles=self.candles,
                      factor_returns=self.factors, selection=self.selection,
                      reference_sessions=self.days, membership=self.members,
                      start=self.days[0], end=date(2022, 1, 7),
                      max_candle_age=self.ages, max_factor_age=timedelta(days=3))
        values.update(overrides)
        return build_examples(**values)

    def test_features_use_only_available_bars_and_future_label_is_not_a_feature(self):
        result, names, missing = self.make()
        self.assertEqual(len(result), 3)
        self.assertEqual(len(names), 37)
        self.assertEqual(missing, [])
        self.assertAlmostEqual(result.loc[0, "1m_close_ratio"], 0.05)
        self.assertAlmostEqual(result.loc[0, "factor_0_return"], 0.1)
        self.assertAlmostEqual(result.loc[0, "target_log_return"], np.log(1.1))
        self.assertEqual(result.loc[0, "target_close"], 110.0)
        self.assertNotIn("target_close", names)
        changed = self.frame.copy()
        changed.loc[1, ["open", "close", "high", "low"]] = [400, 400, 401, 399]
        modified = dict(self.candles, **{"1m": changed})
        updated, _, _ = self.make(candles=modified)
        self.assertEqual(result.loc[0, "1m_close_ratio"], updated.loc[0, "1m_close_ratio"])
        self.assertNotEqual(result.loc[1, "1m_close_ratio"], updated.loc[1, "1m_close_ratio"])
        self.assertTrue(pd.isna(result.loc[1, "factor_0_return"]))

    def test_selection_cutoff_missing_sources_and_membership(self):
        selection = dict(self.selection, train_end_exclusive="2022-01-04")
        with self.assertRaisesRegex(ValueError, "cutoff"):
            self.make(selection=selection)
        self.selection["stocks"][0]["selected"][0]["correlation"] = 0.6999999
        with self.assertRaisesRegex(ValueError, "threshold"):
            self.make()
        self.selection["stocks"][0].update(status="no_qualifying_factor", selected=[])
        with self.assertRaisesRegex(ValueError, "stock-only"):
            self.make(factor_returns={})
        candles = dict(self.candles, **{"1m": None})
        result, names, missing = self.make(factor_returns={}, candles=candles, allow_no_factors=True,
                                           membership=(Membership("AAA", self.days[0], self.days[2], "test", ""),))
        self.assertEqual(len(result), 1)
        self.assertEqual(missing, ["1m"])
        self.assertTrue(result["1m_close_ratio"].isna().all())
        self.assertEqual(len(names), 36)

    def test_missing_reference_session_cannot_create_label_across_gap(self):
        daily = self.daily.drop(index=1)
        result, _, _ = self.make(daily=daily)
        self.assertEqual(result["session"].tolist(), [self.days[2]])
        self.assertEqual(result["target_session"].tolist(), [self.days[3]])

    def test_old_membership_candle_cannot_be_used_after_reentry(self):
        membership = (Membership("AAA", self.days[0], self.days[1], "test", ""),
                      Membership("AAA", self.days[2], None, "test", ""))
        candles = dict(self.candles, **{"1m": self.frame.iloc[:1]})
        result, _, _ = self.make(membership=membership, candles=candles)
        self.assertEqual(result["session"].tolist(), [self.days[2]])
        self.assertTrue(pd.isna(result.loc[0, "1m_close_ratio"]))

    def test_forecast_does_not_require_future_daily_close_and_matches_example_features(self):
        expected, names, _ = self.make()
        forecast, actual_names, _ = build_forecast_features(
            "AAA", self.daily.iloc[:1], self.candles, self.factors, self.selection, self.members,
            session=self.days[0], target_session=self.days[1], max_candle_age=self.ages,
            max_factor_age=timedelta(days=3))
        self.assertEqual(actual_names, names)
        pd.testing.assert_series_equal(expected.loc[0, names], forecast.loc[0, names])
        self.assertNotIn("target_log_return", forecast)
        with self.assertRaisesRegex(ValueError, "membership interval"):
            build_forecast_features(
                "AAA", self.daily.iloc[:1], self.candles, self.factors, self.selection,
                (Membership("AAA", self.days[0], self.days[1], "test", ""),),
                session=self.days[0], target_session=self.days[1],
                max_candle_age=self.ages, max_factor_age=timedelta(days=3))

    def test_prepared_loader_rejects_corrupted_and_oversized_chunks(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "stock" / "AAA" / "1m"
            folder.mkdir(parents=True)
            path = folder / "part-000000.parquet"
            self.frame.to_parquet(path, index=False)
            path.with_suffix(".json").write_text(json.dumps({
                "rows": 2, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), encoding="utf-8")
            (root / "coverage.json").write_text(json.dumps({"jobs": [
                {"kind": "stock", "symbol": "AAA", "timeframe": "1m",
                 "status": "prepared", "chunks": 1, "rows": 2}]}), encoding="utf-8")
            self.assertEqual(len(load_prepared_candles(root, "AAA", "1m", max_rows=2)), 2)
            with self.assertRaisesRegex(ValueError, "oversized"):
                load_prepared_candles(root, "AAA", "1m", max_rows=1)
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                load_prepared_candles(root, "AAA", "1m", max_rows=2)

    def test_regression_components_and_positive_price_conversion(self):
        tree = build_price_tree()
        self.assertEqual(tree.get_params()["objective"], "reg:squarederror")
        network = build_price_lstm(torch, 3)
        self.assertEqual(tuple(network(torch.zeros((2, 4, 3))).shape), (2,))
        actual = combine_price_predictions(np.array([100.0]), np.log([1.1]),
                                           np.log([0.9]), tree_weight=0.5)
        self.assertAlmostEqual(actual[0], 100 * np.sqrt(1.1 * 0.9))
        with self.assertRaises(ValueError):
            combine_price_predictions(np.array([0.0]), np.array([0.0]), np.array([0.0]), tree_weight=0.5)

        class Tree:
            def predict(self, values):
                return np.full(len(values), np.log(1.1))

        class Network(torch.nn.Module):
            def forward(self, values):
                return torch.full((len(values),), float(np.log(0.9)))

        model = Network()
        predicted = predict_hybrid_prices(Tree(), model, torch, np.zeros((1, 3)),
                                          np.zeros((1, 4, 3)), np.array([100]), tree_weight=0.5)
        self.assertAlmostEqual(predicted[0], actual[0], places=5)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
