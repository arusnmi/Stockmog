from __future__ import annotations

import argparse
from datetime import date, timedelta

from config.settings import DATA_DIR, ensure_project_structure
from data_pipeline.csv_import import import_csv
from data_pipeline.csv_dataset import inspect_training_csv
from data_pipeline.providers import YahooFinanceProvider
from data_pipeline.storage import CSVStore
from data_pipeline.validation import validate_ohlcv


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

    store = CSVStore(DATA_DIR / "raw")
    path = store.save_rows(args.symbol, args.interval, rows)
    print(f"Saved {len(rows)} rows to {path}")
    print(f"Duplicate candles: {validation['duplicate_count']}")
    print(f"Missing candles: {validation['missing_count']}")
    print(f"Invalid rows: {validation['invalid_rows']}")


def cmd_inspect_csv(args: argparse.Namespace) -> None:
    ensure_project_structure()
    summary = inspect_training_csv(args.file)
    print(f"CSV: {summary['file']}")
    print(f"Rows: {summary['rows']}")
    if summary["rows"]:
        print(f"Range: {summary['first_timestamp']} to {summary['last_timestamp']}")
    else:
        print("Range: empty dataset")
    print(f"Columns: {', '.join(summary['columns'])}")


def cmd_import_csv(args: argparse.Namespace) -> None:
    ensure_project_structure()
    result = import_csv(
        args.file,
        symbol=args.symbol,
        interval=args.interval,
        output_path=args.output,
    )
    print(f"Saved {result['rows_written']} validated candles for {result['symbol']} ({result['interval']})")
    print(f"Training CSV: {result['output']}")
    print(f"Rows read: {result['rows_read']}")
    print(f"Duplicate rows skipped: {result['duplicate_rows']}")
    print(f"Invalid rows skipped: {result['invalid_rows']}")
    print(f"Missing intraday candles within sessions: {result['missing_candles']}")
    if result["errors_file"] is not None:
        print(f"Rejected rows log: {result['errors_file']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1: market-data infrastructure for the trading system.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and validate market data")
    download.add_argument("--symbol", required=True, help="Ticker symbol to download, for example INFY")
    download.add_argument("--interval", default="5m", help="OHLCV interval such as 5m or 1d")
    download.add_argument("--days", type=int, default=365, help="Lookback in days")
    download.set_defaults(func=cmd_download)

    inspect = subparsers.add_parser("inspect-csv", help="Inspect a CSV training dataset")
    inspect.add_argument("--file", required=True, help="Path to a normalized training CSV")
    inspect.set_defaults(func=cmd_inspect_csv)

    import_csv_parser = subparsers.add_parser(
        "import-csv",
        help="Validate and normalize a Kaggle OHLC or OHLCV CSV",
    )
    import_csv_parser.add_argument("--file", required=True, help="Path to an extracted Kaggle CSV file")
    import_csv_parser.add_argument("--symbol", help="Override the symbol inferred from the CSV filename")
    import_csv_parser.add_argument(
        "--interval",
        help="Override the interval inferred from the CSV filename (for example 1m, 5m, or 1d)",
    )
    import_csv_parser.add_argument(
        "--output",
        help="Destination for the normalized CSV (defaults under data\\raw)",
    )
    import_csv_parser.set_defaults(func=cmd_import_csv)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
