import unittest
import warnings
from unittest.mock import patch

import pandas as pd

from data_pipeline.providers import YahooFinanceProvider


class YahooProviderTests(unittest.TestCase):
    def test_single_ticker_multiindex_prices_are_extracted_as_scalars(self):
        frame = pd.DataFrame(
            [[10., 12., 9., 11., 100.]],
            index=pd.to_datetime(["2026-01-05"]),
            columns=pd.MultiIndex.from_product(
                [["Open", "High", "Low", "Close", "Volume"], ["TEST.NS"]],
                names=["Price", "Ticker"]))
        with patch("yfinance.download", return_value=frame), warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            rows = YahooFinanceProvider().fetch_history("TEST.NS", "2026-01-05", "2026-01-06")
        self.assertEqual(rows[0]["close"], 11.)
        self.assertEqual(rows[0]["volume"], 100.)


if __name__ == "__main__":
    unittest.main()
