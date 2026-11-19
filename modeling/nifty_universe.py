"""Point-in-time Nifty 50 membership from a pinned, public dataset."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.request import urlopen


SOURCE_REVISION = "0786cd9159742f4f2012dcd5f01c1f224b0b019b"
SOURCE_URL = (
    "https://raw.githubusercontent.com/aditya-jha/nse-historical-membership/"
    f"{SOURCE_REVISION}/index_history/data/index_membership_history.csv"
)
COVERAGE_START = date(2018, 4, 1)
COVERAGE_END = date(2026, 10, 13)


@dataclass(frozen=True)
class Membership:
    symbol: str
    valid_from: date
    valid_to: date | None
    source: str
    source_url: str

    def contains(self, day: date) -> bool:
        return self.valid_from <= day and (self.valid_to is None or day < self.valid_to)


def parse_nifty_membership(content: str) -> tuple[Membership, ...]:
    reader = csv.DictReader(io.StringIO(content))
    required = {"index_name", "symbol", "valid_from", "valid_to", "source", "source_url"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("Membership CSV is missing required columns")
    intervals: list[Membership] = []
    for row in reader:
        if row["index_name"] != "Nifty 50":
            continue
        symbol = row["symbol"].strip()
        if not symbol:
            raise ValueError("Nifty 50 membership has an empty symbol")
        start = date.fromisoformat(row["valid_from"])
        end = date.fromisoformat(row["valid_to"]) if row["valid_to"] else None
        if end is not None and end <= start:
            raise ValueError(f"Invalid membership interval for {symbol}")
        intervals.append(Membership(symbol, start, end, row["source"], row["source_url"]))
    if not intervals:
        raise ValueError("No Nifty 50 memberships found")
    return tuple(intervals)


def members_on(intervals: tuple[Membership, ...], day: date) -> set[str]:
    if not COVERAGE_START <= day <= COVERAGE_END:
        raise ValueError(
            f"Membership date {day} is outside verified source coverage "
            f"{COVERAGE_START} to {COVERAGE_END}"
        )
    return {entry.symbol for entry in intervals if entry.contains(day)}


def local_symbol(symbol: str, intervals: tuple[Membership, ...]) -> str:
    """Resolve the documented LTIM rename only for its verified index interval."""
    if symbol != "LTM":
        return symbol
    entries = [entry for entry in intervals if entry.symbol == symbol]
    if not entries or any(
        entry.valid_from < date(2023, 7, 13)
        or entry.valid_to is None
        or entry.valid_to > date(2024, 9, 30)
        for entry in entries
    ):
        raise ValueError("LTM membership falls outside the verified LTIM mapping")
    return "LTIM"


def download_membership(destination: Path) -> None:
    with urlopen(SOURCE_URL, timeout=60) as response:
        content = response.read().decode("utf-8-sig")
    intervals = parse_nifty_membership(content)
    if len(members_on(intervals, date(2022, 1, 3))) != 50:
        raise ValueError("Nifty 50 membership validation failed for 2022-01-03")
    if len(members_on(intervals, date(2026, 1, 21))) != 50:
        raise ValueError("Nifty 50 membership validation failed for 2026-01-21")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as output:
        output.write(content)
