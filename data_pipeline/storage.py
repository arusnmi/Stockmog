from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class ParquetStore:
    """Local file store that writes OHLCV rows to Parquet when the dependency is available."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_rows(self, symbol: str, interval: str, rows: Iterable[dict[str, Any]]) -> Path:
        symbol_key = str(symbol).upper()
        destination_dir = self.root_dir / symbol_key
        destination_dir.mkdir(parents=True, exist_ok=True)

        payload = list(rows)
        parquet_path = destination_dir / f"{symbol_key}_{interval}.parquet"
        try:
            import pandas as pd
            import pyarrow  # noqa: F401 - validates parquet support is installed

            dataframe = pd.DataFrame(payload)
            if not dataframe.empty and "timestamp" in dataframe.columns:
                dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
            if "timestamp" in dataframe.columns:
                dataframe.sort_values("timestamp", inplace=True, kind="mergesort")
            dataframe.to_parquet(parquet_path, index=False)
            return parquet_path
        except (ModuleNotFoundError, ImportError, OSError):
            fallback_path = destination_dir / f"{symbol_key}_{interval}.json"
            with fallback_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
            return fallback_path
