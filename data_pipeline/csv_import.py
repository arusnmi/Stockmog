from __future__ import annotations

import csv
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import RAW_DIR

_INTERVAL_SUFFIXES = {
    "minute": "1m",
    "1minute": "1m",
    "3minute": "3m",
    "5minute": "5m",
    "15minute": "15m",
    "60minute": "60m",
    "day": "1d",
    "daily": "1d",
}
_TIMESTAMP_COLUMNS = {"timestamp", "datetime", "datetimestamp", "datetimeindex"}
_DATE_COLUMNS = {"date", "tradingdate"}
_TIME_COLUMNS = {"time", "tradingtime"}
_PRICE_COLUMNS = ("open", "high", "low", "close")


def _normalize_header(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def infer_symbol_and_interval(path: str | Path) -> tuple[str, str]:
    """Infer the instrument and candle interval from Kaggle-style CSV filenames."""
    file_path = Path(path)
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {file_path.name}")

    stem = file_path.stem
    if "_" not in stem:
        raise ValueError(
            f"Cannot infer the interval from {file_path.name}; use --symbol and --interval."
        )

    symbol, suffix = stem.rsplit("_", 1)
    interval = _INTERVAL_SUFFIXES.get(_normalize_header(suffix))
    if not symbol.strip() or interval is None:
        raise ValueError(
            f"Cannot infer symbol and interval from {file_path.name}; use --symbol and --interval."
        )
    return symbol.strip().upper(), interval


def _parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp is empty")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass

    for pattern in (
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    raise ValueError(f"unsupported timestamp format: {text!r}")


def _parse_number(value: str | None, field: str, *, optional: bool = False) -> float | None:
    text = (value or "").strip().replace(",", "")
    if not text:
        if optional:
            return None
        raise ValueError(f"{field} is empty")

    try:
        number = float(text)
    except ValueError as exc:
        if optional and text.lower() in {"na", "n/a", "null", "none"}:
            return None
        raise ValueError(f"{field} is not numeric: {text!r}") from exc

    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def _parse_required_number(value: str | None, field: str) -> float:
    number = _parse_number(value, field)
    if number is None:
        raise ValueError(f"{field} is required")
    return number


def _interval_delta(interval: str) -> timedelta | None:
    match = re.fullmatch(r"(\d+)(m|h|d)", interval.lower())
    if match is None:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    return None


def import_csv(
    file_path: str | Path,
    *,
    symbol: str | None = None,
    interval: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and normalize a source CSV into a training-ready CSV without loading it into memory."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {path.name}")

    inferred_symbol = ""
    inferred_interval = ""
    if symbol is None or interval is None:
        inferred_symbol, inferred_interval = infer_symbol_and_interval(path)
    symbol = str(symbol or inferred_symbol).strip().upper()
    interval = str(interval or inferred_interval).strip().lower()
    if not symbol:
        raise ValueError("symbol must not be empty")
    if not interval:
        raise ValueError("interval must not be empty")

    output = (
        Path(output_path)
        if output_path is not None
        else RAW_DIR / symbol / f"{symbol}_{interval}.csv"
    )
    if output.suffix.lower() != ".csv":
        raise ValueError(f"Normalized training output must be a CSV file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f"{output.name}.tmp")
    errors_output = output.with_name(f"{output.stem}.errors.csv")
    temporary_errors = errors_output.with_name(f"{errors_output.name}.tmp")
    if path.resolve() == temporary_output.resolve():
        raise ValueError("Source file conflicts with the temporary output path.")

    rows_read = 0
    invalid_rows = 0
    duplicate_rows = 0
    rows_written = 0
    missing_candles = 0
    previous_timestamp: datetime | None = None

    try:
        with (
            path.open("r", encoding="utf-8-sig", newline="") as csv_file,
            temporary_output.open("w", encoding="utf-8", newline="") as output_file,
            temporary_errors.open("w", encoding="utf-8", newline="") as errors_file,
        ):
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                raise ValueError(f"CSV file has no header: {path.name}")

            columns = {
                _normalize_header(name): name
                for name in reader.fieldnames
                if name is not None
            }
            timestamp_column = next(
                (columns[name] for name in _TIMESTAMP_COLUMNS if name in columns),
                None,
            )
            date_column = next(
                (columns[name] for name in _DATE_COLUMNS if name in columns),
                None,
            )
            time_column = next(
                (columns[name] for name in _TIME_COLUMNS if name in columns),
                None,
            )
            price_columns = {name: columns.get(name) for name in _PRICE_COLUMNS}
            missing_columns = [name for name, column in price_columns.items() if column is None]
            if timestamp_column is None and date_column is None:
                missing_columns.append("timestamp/date")
            if missing_columns:
                raise ValueError(
                    f"CSV is missing required columns: {', '.join(missing_columns)}"
                )
            volume_column = columns.get("volume") or columns.get("vol")

            writer = csv.DictWriter(
                output_file,
                fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
            )
            writer.writeheader()
            error_writer = csv.DictWriter(
                errors_file,
                fieldnames=["source_line", "error", "row"],
            )
            error_writer.writeheader()

            for line_number, csv_row in enumerate(reader, start=2):
                rows_read += 1
                try:
                    if timestamp_column is not None:
                        timestamp_text = csv_row.get(timestamp_column, "")
                    else:
                        timestamp_text = csv_row.get(date_column or "", "")
                        if time_column is not None:
                            timestamp_text = (
                                f"{timestamp_text} {csv_row.get(time_column, '')}".strip()
                            )
                    timestamp = _parse_timestamp(timestamp_text or "")
                    prices = {
                        field: _parse_required_number(csv_row.get(column), field)
                        for field, column in price_columns.items()
                        if column is not None
                    }
                    open_price = prices["open"]
                    high_price = prices["high"]
                    low_price = prices["low"]
                    close_price = prices["close"]
                    if min(open_price, high_price, low_price, close_price) <= 0:
                        raise ValueError("OHLC prices must be greater than zero")
                    if high_price < max(open_price, low_price, close_price):
                        raise ValueError("high must be greater than or equal to open, low and close")
                    if low_price > min(open_price, high_price, close_price):
                        raise ValueError("low must be less than or equal to open, high and close")
                    if previous_timestamp is not None and timestamp < previous_timestamp:
                        raise ValueError("timestamps are out of order")
                    if previous_timestamp is not None and timestamp == previous_timestamp:
                        duplicate_rows += 1
                        continue

                    volume = (
                        _parse_number(csv_row.get(volume_column), "volume", optional=True)
                        if volume_column is not None
                        else None
                    )
                    expected_delta = _interval_delta(interval)
                    if (
                        expected_delta is not None
                        and previous_timestamp is not None
                        and timestamp.date() == previous_timestamp.date()
                    ):
                        difference = timestamp - previous_timestamp
                        if difference > expected_delta:
                            missing_candles += max(
                                int(difference.total_seconds() // expected_delta.total_seconds()) - 1,
                                0,
                            )
                    writer.writerow(
                        {
                            "timestamp": timestamp.isoformat(sep=" "),
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": "" if volume is None else volume,
                        }
                    )
                    rows_written += 1
                    previous_timestamp = timestamp
                except (TypeError, ValueError) as exc:
                    invalid_rows += 1
                    error_writer.writerow(
                        {
                            "source_line": line_number,
                            "error": str(exc),
                            "row": repr(csv_row),
                        }
                    )

        temporary_output.replace(output)
        if invalid_rows:
            temporary_errors.replace(errors_output)
        else:
            temporary_errors.unlink(missing_ok=True)
            errors_output.unlink(missing_ok=True)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        temporary_errors.unlink(missing_ok=True)
        raise

    return {
        "file": str(path),
        "output": str(output),
        "errors_file": str(errors_output) if invalid_rows else None,
        "symbol": symbol,
        "interval": interval,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "duplicate_rows": duplicate_rows,
        "invalid_rows": invalid_rows,
        "missing_candles": missing_candles,
    }
