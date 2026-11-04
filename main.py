from __future__ import annotations

import argparse
from datetime import date, timedelta

from config.settings import DATA_DIR, ensure_project_structure
from data_pipeline.csv_import import import_csv
from data_pipeline.csv_dataset import inspect_training_csv
from data_pipeline.huggingface_stocks import (
    NSE_HOLIDAYS_CSV,
    prepare_huggingface_dataset,
)
from data_pipeline.providers import YahooFinanceProvider
from data_pipeline.storage import CSVStore
from data_pipeline.validation import validate_ohlcv
from features.feature_engineering import write_features


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


def cmd_features(args: argparse.Namespace) -> None:
    ensure_project_structure()
    output = write_features(
        args.input,
        style=args.style,
        output_path=args.output,
        box_window=args.box_window,
        relative_volume_window=args.relative_volume_window,
        opening_range_bars=args.opening_range_bars,
        market_index_path=args.market_index,
        volatility_index_path=args.volatility_index,
    )
    print(f"Saved {args.style} feature CSV to {output}")


def cmd_prepare_stock_data(args: argparse.Namespace) -> None:
    ensure_project_structure()
    symbols = (
        [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
        if args.symbols is not None
        else None
    )
    result = prepare_huggingface_dataset(
        repo_id=args.repo_id,
        split=args.split,
        symbols=symbols,
        output_root=args.output_root,
        download_dir=args.download_dir,
        keep_shards=args.keep_shards,
        batch_size=args.batch_size,
        holidays_csv=args.holidays_csv,
    )
    print(f"Processed shards: {result['shards']}")
    print(f"Processed symbols: {result['processed_symbols']}")
    print(f"Source rows processed: {result['source_rows']}")
    print(f"Output directory: {args.output_root}")
    if "minute_start_date" in result and "minute_end_date" in result:
        print(
            "Daily-source date range aligned to minute data: "
            f"{result['minute_start_date']} through {result['minute_end_date']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSV market-data and feature-engineering tools for the trading system."
    )
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

    features = subparsers.add_parser(
        "features",
        help="Generate causal intraday or swing features from a market CSV",
    )
    features.add_argument("--input", required=True, help="Path to a normalized OHLCV CSV")
    features.add_argument("--style", required=True, choices=("intraday", "swing"))
    features.add_argument("--output", help="Feature CSV destination (defaults under data\\features)")
    features.add_argument("--box-window", type=int, help="Prior bars used for box boundaries")
    features.add_argument(
        "--relative-volume-window",
        type=int,
        default=20,
        help="Prior bars used for the relative-volume baseline",
    )
    features.add_argument(
        "--opening-range-bars",
        type=int,
        default=3,
        help="Intraday opening-range length in candles",
    )
    features.add_argument(
        "--market-index",
        help="Optional index CSV used for strictly lagged market-context features",
    )
    features.add_argument(
        "--volatility-index",
        help="Optional VIX CSV used for strictly lagged market-context features",
    )
    features.set_defaults(func=cmd_features)

    stock_data = subparsers.add_parser(
        "prepare-stock-data",
        help="Download Hugging Face stock Parquet shards and write per-symbol feature CSVs",
    )
    stock_data.add_argument(
        "--repo-id",
        default="Saintforest/indian-stock-market-minute-data",
        help="Hugging Face dataset repository ID",
    )
    stock_data.add_argument(
        "--split",
        choices=("minute", "day", "both"),
        default="both",
        help=(
            "minute creates 5m/15m/30m/1h plus 1d, 2d, and trading_week; "
            "day creates 1d, 2d, and trading_week"
        ),
    )
    stock_data.add_argument(
        "--symbols",
        help="Optional comma-separated NSE symbol list, for example RELIANCE,TCS,INFY",
    )
    stock_data.add_argument(
        "--holidays-csv",
        default=str(NSE_HOLIDAYS_CSV),
        help="CSV of official NSE closure dates (requires a YYYY-MM-DD `date` column)",
    )
    stock_data.add_argument(
        "--output-root",
        default="data\\stocks",
        help="Output root for per-symbol compressed CSVs",
    )
    stock_data.add_argument(
        "--download-dir",
        default="data\\downloads\\huggingface",
        help="Temporary/local directory for Hugging Face Parquet shards",
    )
    stock_data.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep each downloaded Parquet shard after it has been processed",
    )
    stock_data.add_argument(
        "--batch-size",
        type=int,
        default=65_536,
        help="Rows per streaming Parquet batch",
    )
    stock_data.set_defaults(func=cmd_prepare_stock_data)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
