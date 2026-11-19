"""Causal next-candle price examples for each prepared Nifty timeframe."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import INTRADAY_TIMEFRAMES
from modeling.nifty_preparation import TIMEFRAMES
from modeling.nifty_price_model import _validate_candles
from modeling.nifty_universe import COVERAGE_END, COVERAGE_START, Membership


class NoCandleExamples(ValueError):
    """No eligible complete consecutive candles exist for this stock and window."""


def build_candle_examples(
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    reference_sessions: list[date],
    membership: tuple[Membership, ...],
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Create one next-close label per consecutive, complete pair of candles.

    The exclusive date window applies to both source and target sessions.
    Intraday pairs never cross sessions. Multi-session pairs require complete
    reference-calendar groups, including the target group's final session.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if not COVERAGE_START <= start < end <= COVERAGE_END:
        raise ValueError("Example window is outside verified membership coverage")
    if reference_sessions != sorted(set(reference_sessions)):
        raise ValueError("Reference sessions must be sorted and unique")
    if any(item.symbol != symbol for item in membership):
        raise ValueError("Membership intervals do not match the stock")
    bars = _validate_candles(candles)
    if len(bars) < 2:
        raise NoCandleExamples(f"No consecutive complete {timeframe} candles for {symbol}")
    sessions = bars["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
    target_session = sessions.shift(-1)
    target_timestamp = bars["timestamp"].shift(-1)
    target_available = bars["available_at"].shift(-1)
    target_close = bars["close"].shift(-1)
    complete = ~bars["is_partial"].astype(bool)
    eligible = complete & complete.shift(-1, fill_value=False)
    eligible &= (sessions >= start) & (target_session < end)
    eligible &= target_available > bars["available_at"]
    eligible &= bars["available_at"] <= target_timestamp
    reference_index = {day: index for index, day in enumerate(reference_sessions)}
    if timeframe in INTRADAY_TIMEFRAMES:
        eligible &= sessions.isin(reference_index)
        eligible &= sessions == target_session
        eligible &= target_timestamp == bars["timestamp"] + timedelta(
            minutes=INTRADAY_TIMEFRAMES[timeframe])
    else:
        if sessions.duplicated().any():
            raise ValueError(f"Duplicate {timeframe} candle session for {symbol}")
        if timeframe == "1d":
            next_session = dict(zip(reference_sessions, reference_sessions[1:]))
            expected_next = sessions.map(next_session)
            final_sessions = sessions
        else:
            if "trading_days" not in candles.columns:
                raise ValueError(f"{timeframe} candles require trading_days")
            counts = pd.to_numeric(candles["trading_days"].reset_index(drop=True), errors="raise")
            if counts.isna().any() or (counts <= 0).any() or (counts % 1 != 0).any():
                raise ValueError("Invalid multi-session candle trading-day counts")
            if timeframe == "2d":
                final_by_start = dict(zip(reference_sessions, reference_sessions[1:]))
                next_session = dict(zip(reference_sessions, reference_sessions[2:]))
                expected_counts = pd.Series(2, index=bars.index)
            else:
                groups_by_week: dict[date, list[date]] = {}
                for day in reference_sessions:
                    groups_by_week.setdefault(day - timedelta(days=day.weekday()), []).append(day)
                groups = list(groups_by_week.values())
                final_by_start = {}
                count_by_start = {}
                next_session = {}
                for group, following in zip(groups, groups[1:]):
                    monday = group[0] - timedelta(days=group[0].weekday())
                    next_monday = following[0] - timedelta(days=following[0].weekday())
                    for label in {monday, group[0]}:
                        final_by_start[label] = group[-1]
                        count_by_start[label] = len(group)
                        next_session[label] = {next_monday, following[0]}
                if groups:
                    last = groups[-1]
                    for label in {last[0], last[0] - timedelta(days=last[0].weekday())}:
                        final_by_start[label] = last[-1]
                        count_by_start[label] = len(last)
                expected_counts = sessions.map(count_by_start)
            expected_next = sessions.map(next_session)
            final_sessions = sessions.map(final_by_start)
            eligible &= counts == expected_counts
            eligible &= counts.shift(-1) == expected_counts.shift(-1)
            if timeframe == "trading_week":
                eligible &= pd.Series([isinstance(expected, set) and target in expected
                                       for expected, target in zip(expected_next, target_session)])
                eligible &= (target_session - sessions).map(
                    lambda delta: pd.notna(delta) and 1 <= delta.days <= 7)
            else:
                eligible &= target_session == expected_next
        if timeframe == "1d":
            eligible &= target_session == expected_next
        target_final = final_sessions.shift(-1)
        eligible &= target_final.notna() & (target_final < end)
        if timeframe != "1d":
            earliest = pd.to_datetime(final_sessions, errors="coerce").dt.tz_localize(
                "Asia/Kolkata").add(timedelta(hours=15, minutes=30)).dt.tz_convert("UTC")
            eligible &= bars["available_at"] >= earliest
            eligible &= target_available >= earliest.shift(-1)
    target_final = sessions.shift(-1) if timeframe in INTRADAY_TIMEFRAMES else final_sessions.shift(-1)
    in_membership = pd.Series(False, index=bars.index)
    for item in membership:
        in_membership |= (sessions >= item.valid_from) & (
            True if item.valid_to is None else target_final < item.valid_to)
    eligible &= in_membership
    positions = np.flatnonzero(eligible.to_numpy())
    if len(positions) == 0:
        raise NoCandleExamples(f"No consecutive complete {timeframe} candles for {symbol}")
    current = bars.iloc[positions]
    future = bars.iloc[positions + 1]
    anchor = current["close"].to_numpy(dtype=np.float64)
    target = future["close"].to_numpy(dtype=np.float64)
    return pd.DataFrame({
        "symbol": symbol,
        "timeframe": timeframe,
        "session": sessions.iloc[positions].to_numpy(),
        "target_session": target_session.iloc[positions].to_numpy(),
        "timestamp": current["timestamp"].reset_index(drop=True),
        "available_at": current["available_at"].reset_index(drop=True),
        "target_timestamp": future["timestamp"].reset_index(drop=True),
        "label_available_at": future["available_at"].reset_index(drop=True),
        "anchor_close": anchor,
        "target_close": target,
        "target_log_return": np.log(target / anchor),
        "body": (current["close"] / current["open"] - 1).to_numpy(),
        "range": ((current["high"] - current["low"]) / current["close"]).to_numpy(),
        "volume": current["volume"].to_numpy(dtype=np.float64),
    })
