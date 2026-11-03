import unittest
from pathlib import Path

from database.database import SQLiteMarketDatabase


class SQLiteDatabaseTests(unittest.TestCase):
    def test_database_creates_schema_and_persists_rows(self):
        path = Path(".tmp-trading.db")
        if path.exists():
            path.unlink()

        database = SQLiteMarketDatabase(path)
        rows = [
            {"timestamp": "2024-01-01", "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5, "volume": 100.0},
            {"timestamp": "2024-01-02", "open": 10.5, "high": 12.0, "low": 10.0, "close": 11.0, "volume": 110.0},
        ]

        count = database.insert_rows("INFY", "1d", rows, source="test")
        self.assertEqual(count, 2)
        self.assertEqual(database.count_rows("INFY", "1d"), 2)
        self.assertEqual(len(database.fetch_rows("INFY", "1d")), 2)

        database.close()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
