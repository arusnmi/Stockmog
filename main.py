from __future__ import annotations

import argparse
from datetime import date, timedelta

from config.settings import DATA_DIR, ensure_project_structure
from data_pipeline.providers import YahooFinanceProvider
from data_pipeline.storage import ParquetStore
from data_pipeline.validation import validate_ohlcv
from database.database import SQLiteMarketDatabase


def _make_default_range(days: int = 365) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def cmd_download(args: argparse.Namespace) -> None:
    ensure_project_structure()
    provider = YahooFinanceProvider()
    start, end = _make_default_range(args.days)
    rows = provider.fetch_history(args.symbol, start, end, interval=args.interval)
    validation = validate_ohlcv(rows, interval=args.interval)

    db = SQLiteMarketDatabase()
    db.insert_rows(args.symbol, args.interval, rows, source="yfinance")
    for issue in validation["issues"]:
        db.record_quality_issue(args.symbol, args.interval, "validation", issue, payload=str(len(rows)))

    store = ParquetStore(DATA_DIR / "raw")
    path = store.save_rows(args.symbol, args.interval, rows)
    print(f"Saved {len(rows)} rows to {path}")
    print(f"Duplicate candles: {validation['duplicate_count']}")
    print(f"Missing candles: {validation['missing_count']}")
    print(f"Invalid rows: {validation['invalid_rows']}")


def cmd_inspect(args: argparse.Namespace) -> None:
    ensure_project_structure()
    db = SQLiteMarketDatabase()
    total = db.count_rows(args.symbol, args.interval)
    print(f"{args.symbol} {args.interval}: {total} rows in SQLite")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1: market-data infrastructure for the trading system.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and validate market data")
    download.add_argument("--symbol", required=True, help="Ticker symbol to download, for example INFY")
    download.add_argument("--interval", default="5m", help="OHLCV interval such as 5m or 1d")
    download.add_argument("--days", type=int, default=365, help="Lookback in days")
    download.set_defaults(func=cmd_download)

    inspect = subparsers.add_parser("inspect", help="Check what has already been saved locally")
    inspect.add_argument("--symbol", required=True)
    inspect.add_argument("--interval", default="5m")
    inspect.set_defaults(func=cmd_inspect)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
