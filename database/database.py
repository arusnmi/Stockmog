from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from config.settings import DB_PATH


class SQLiteMarketDatabase:
    """Small SQLite utility layer for storing OHLCV candles and data-quality checks."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = None
        self.initialize()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT,
                    UNIQUE(symbol, interval, timestamp)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_interval_ts ON market_data(symbol, interval, timestamp)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS data_quality_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    issue_type TEXT NOT NULL,
                    issue_message TEXT NOT NULL,
                    payload TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def insert_rows(self, symbol: str, interval: str, rows: Iterable[dict[str, Any]], source: str = "unknown") -> int:
        count = 0
        connection = self._connect()
        try:
            for row in rows:
                cleaned = {
                    "symbol": str(symbol).upper(),
                    "interval": str(interval),
                    "timestamp": str(row.get("timestamp") or row.get("datetime") or row.get("time")),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "source": source,
                }
                connection.execute(
                    """
                    INSERT OR REPLACE INTO market_data (
                        symbol, interval, timestamp, open, high, low, close, volume, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cleaned["symbol"],
                        cleaned["interval"],
                        cleaned["timestamp"],
                        cleaned["open"],
                        cleaned["high"],
                        cleaned["low"],
                        cleaned["close"],
                        cleaned["volume"],
                        cleaned["source"],
                    ),
                )
                count += 1
            connection.commit()
            return count
        finally:
            connection.close()

    def record_quality_issue(
        self,
        symbol: str,
        interval: str,
        issue_type: str,
        issue_message: str,
        payload: str | None = None,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO data_quality_issues (symbol, interval, issue_type, issue_message, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(symbol).upper(), str(interval), issue_type, issue_message, payload),
            )
            connection.commit()
        finally:
            connection.close()

    def count_rows(self, symbol: str | None = None, interval: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM market_data WHERE 1 = 1"
        params: list[str] = []
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(str(symbol).upper())
        if interval is not None:
            query += " AND interval = ?"
            params.append(str(interval))
        connection = self._connect()
        try:
            row = connection.execute(query, params).fetchone()
            return int(row[0]) if row else 0
        finally:
            connection.close()

    def fetch_rows(self, symbol: str, interval: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT symbol, interval, timestamp, open, high, low, close, volume, source
                FROM market_data
                WHERE symbol = ? AND interval = ?
                ORDER BY timestamp ASC
                """,
                (str(symbol).upper(), str(interval)),
            ).fetchall()
            return [
                {
                    "symbol": row[0],
                    "interval": row[1],
                    "timestamp": row[2],
                    "open": row[3],
                    "high": row[4],
                    "low": row[5],
                    "close": row[6],
                    "volume": row[7],
                    "source": row[8],
                }
                for row in rows
            ]
        finally:
            connection.close()
