from __future__ import annotations

from datetime import date
from typing import Any


class MarketDataProvider:
    """Abstract interface for any data source connected to the trading system."""

    def fetch_history(self, symbol: str, start: date | str, end: date | str, interval: str = "1d") -> list[dict[str, Any]]:
        raise NotImplementedError


class YahooFinanceProvider(MarketDataProvider):
    """Provider wrapper for Indian listed instruments using Yahoo Finance as the initial connector."""

    def fetch_history(self, symbol: str, start: date | str, end: date | str, interval: str = "1d") -> list[dict[str, Any]]:
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing is expected before install
            raise RuntimeError("Install project dependencies with `pip install -r requirements.txt` before downloading data.") from exc

        ticker = str(symbol).strip()
        if not ticker:
            raise ValueError("A valid symbol is required.")

        df = yf.download(
            tickers=ticker,
            start=str(start),
            end=str(end),
            interval=str(interval),
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            raise ValueError(f"No data returned for symbol {ticker} with interval {interval}.")

        records: list[dict[str, Any]] = []
        for index, row in df.iterrows():
            open_value = row.get("Open")
            high_value = row.get("High")
            low_value = row.get("Low")
            close_value = row.get("Close")
            volume_value = row.get("Volume")

            records.append(
                {
                    "timestamp": index.isoformat() if hasattr(index, "isoformat") else str(index),
                    "open": float(open_value) if open_value is not None and not isinstance(open_value, (list, tuple, set)) else 0.0,
                    "high": float(high_value) if high_value is not None and not isinstance(high_value, (list, tuple, set)) else 0.0,
                    "low": float(low_value) if low_value is not None and not isinstance(low_value, (list, tuple, set)) else 0.0,
                    "close": float(close_value) if close_value is not None and not isinstance(close_value, (list, tuple, set)) else 0.0,
                    "volume": float(volume_value) if volume_value is not None and not isinstance(volume_value, (list, tuple, set)) else 0.0,
                }
            )
        return records
