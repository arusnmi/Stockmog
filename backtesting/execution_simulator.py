from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import floor, isfinite
from typing import Literal
from zoneinfo import ZoneInfo

ExitReason = Literal["stop_loss", "take_profit", "session_end", "end_of_data", "daily_loss_limit"]


@dataclass(frozen=True)
class ExecutionCosts:
    """Simple configurable per-side costs; bps are applied to traded notional."""

    commission_bps: float = 3.0
    other_fees_bps: float = 7.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        for name, value in (
            ("commission_bps", self.commission_bps),
            ("other_fees_bps", self.other_fees_bps),
            ("slippage_bps", self.slippage_bps),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and not negative")
        if self.slippage_bps >= 10_000:
            raise ValueError("slippage_bps must be less than 10,000 per side")

    @property
    def fee_rate(self) -> float:
        return (self.commission_bps + self.other_fees_bps) / 10_000.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    def entry_fill(self, price: float, direction: int) -> float:
        return price * (1.0 + direction * self.slippage_rate)

    def exit_fill(self, price: float, direction: int) -> float:
        return price * (1.0 - direction * self.slippage_rate)

    def fees(self, entry_notional: float, exit_notional: float) -> float:
        return (entry_notional + exit_notional) * self.fee_rate

    def estimated_round_trip_cost_per_unit(self, price: float) -> float:
        return price * (2.0 * self.fee_rate + 2.0 * self.slippage_rate)


@dataclass(frozen=True)
class ExitFill:
    price: float
    reason: ExitReason


def resolve_exit(
    *,
    direction: int,
    open_price: float,
    high: float,
    low: float,
    stop_price: float,
    target_price: float,
) -> ExitFill | None:
    """Resolve gaps and intrabar barriers; when both are touched, stop wins."""
    if direction == 1:
        if open_price <= stop_price:
            return ExitFill(open_price, "stop_loss")
        if open_price >= target_price:
            return ExitFill(open_price, "take_profit")
        stop_hit = low <= stop_price
        target_hit = high >= target_price
    else:
        if open_price >= stop_price:
            return ExitFill(open_price, "stop_loss")
        if open_price <= target_price:
            return ExitFill(open_price, "take_profit")
        stop_hit = high >= stop_price
        target_hit = low <= target_price

    if stop_hit:
        return ExitFill(stop_price, "stop_loss")
    if target_hit:
        return ExitFill(target_price, "take_profit")
    return None


def calculate_quantity(
    *,
    equity: float,
    risk_fraction: float,
    entry_price: float,
    stop_distance: float,
    costs: ExecutionCosts,
) -> int:
    """Size whole shares for risk and cash, including estimated round-trip costs."""
    if equity <= 0 or risk_fraction <= 0 or entry_price <= 0 or stop_distance <= 0:
        return 0
    risk_budget = equity * risk_fraction
    risk_per_unit = stop_distance + costs.estimated_round_trip_cost_per_unit(entry_price)
    cash_per_unit = entry_price * (1.0 + costs.fee_rate)
    return max(0, min(floor(risk_budget / risk_per_unit), floor(equity / cash_per_unit)))


def session_date(timestamp: datetime) -> datetime.date:
    if timestamp.tzinfo is None:
        return timestamp.date()
    return timestamp.astimezone(ZoneInfo("Asia/Kolkata")).date()
