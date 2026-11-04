from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


class CSVStore:
    """Canonical local CSV store for market data used by the training pipeline."""

    fieldnames = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_rows(self, symbol: str, interval: str, rows: Iterable[dict[str, Any]]) -> Path:
        symbol_key = str(symbol).upper()
        destination_dir = self.root_dir / symbol_key
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{symbol_key}_{interval}.csv"
        temporary_destination = destination.with_name(f"{destination.name}.tmp")

        try:
            with temporary_destination.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writeheader()
                for row in rows:
                    timestamp = row.get("timestamp") or row.get("datetime") or row.get("time")
                    if timestamp is None:
                        raise ValueError("Each market-data row must include a timestamp.")
                    writer.writerow(
                        {
                            "timestamp": str(timestamp),
                            "open": row.get("open"),
                            "high": row.get("high"),
                            "low": row.get("low"),
                            "close": row.get("close"),
                            "volume": row.get("volume"),
                        }
                    )
            temporary_destination.replace(destination)
        except Exception:
            temporary_destination.unlink(missing_ok=True)
            raise

        return destination
