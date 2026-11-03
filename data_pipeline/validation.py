from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()

    value_text = str(value).strip()
    if not value_text:
        return None

    for pattern in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M%z",
        "%Y-%m-%dT%H:%M",
    ):
        try:
            return datetime.strptime(value_text, pattern)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value_text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _interval_to_timedelta(interval: str) -> timedelta:
    interval_text = str(interval).strip().lower()
    if interval_text.endswith("min"):
        return timedelta(minutes=int(interval_text[:-3]))
    if interval_text.endswith("m"):
        return timedelta(minutes=int(interval_text[:-1]))
    if interval_text.endswith("h"):
        return timedelta(hours=int(interval_text[:-1]))
    if interval_text.endswith("d"):
        return timedelta(days=int(interval_text[:-1]) or 1)
    if interval_text in {"1d", "daily"}:
        return timedelta(days=1)
    if interval_text == "5m":
        return timedelta(minutes=5)
    return timedelta(minutes=1)


def validate_ohlcv(rows: list[dict[str, Any]], interval: str = "1d") -> dict[str, Any]:
    """Detect invalid, duplicated and missing candles in OHLCV rows."""
    if not rows:
        return {
            "total_rows": 0,
            "valid_rows": 0,
            "duplicate_count": 0,
            "missing_count": 0,
            "invalid_rows": 0,
            "issues": [],
            "first_timestamp": None,
            "last_timestamp": None,
        }

    seen: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    invalid_rows = 0
    issues: list[str] = []

    for index, row in enumerate(rows):
        timestamp = _coerce_timestamp(row.get("timestamp") or row.get("datetime") or row.get("time"))
        if timestamp is None:
            invalid_rows += 1
            issues.append(f"Row {index}: invalid timestamp")
            continue

        key = timestamp.isoformat()
        if key in seen:
            issues.append(f"Row {index}: duplicate timestamp {key}")
            continue
        seen.add(key)

        normalized_rows.append({**row, "timestamp": timestamp})

    normalized_rows.sort(key=lambda item: item["timestamp"])
    timestamps = [item["timestamp"] for item in normalized_rows]

    missing_count = 0
    expected_delta = _interval_to_timedelta(interval)
    for previous, current in zip(timestamps, timestamps[1:]):
        delta = current - previous
        if delta > expected_delta:
            gaps = int(delta.total_seconds() // expected_delta.total_seconds()) - 1
            missing_count += max(gaps, 0)
            issues.append(f"Missing window between {previous.isoformat()} and {current.isoformat()}: {gaps} candle(s)")

    summary = {
        "total_rows": len(rows),
        "valid_rows": len(normalized_rows),
        "duplicate_count": len(rows) - len(normalized_rows) - invalid_rows,
        "missing_count": missing_count,
        "invalid_rows": invalid_rows,
        "issues": issues,
        "first_timestamp": timestamps[0].isoformat() if timestamps else None,
        "last_timestamp": timestamps[-1].isoformat() if timestamps else None,
    }
    return summary
