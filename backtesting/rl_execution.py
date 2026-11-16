from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from math import floor, isfinite
from typing import Callable, Literal

import backtrader as bt
import numpy as np
import pandas as pd

from .execution_simulator import ExecutionCosts

Action = Literal["wait", "hold", "open_long", "close_long", "open_short", "cover_short"]
_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "20m": 20, "30m": 30, "1h": 60}


@dataclass(frozen=True)
class ExecutionConfig:
    style: Literal["intraday", "swing"]
    interval: str
    initial_cash: float = 50_000.0
    exposure_fraction: float = 0.1
    max_drawdown: float = 0.5
    costs: ExecutionCosts = field(default_factory=ExecutionCosts)

    def __post_init__(self) -> None:
        if self.style not in ("intraday", "swing"):
            raise ValueError("style must be intraday or swing")
        if self.style == "intraday" and self.interval not in _INTERVAL_MINUTES:
            raise ValueError("Unsupported intraday interval")
        if self.style == "swing" and self.interval not in ("1d", "2d", "trading_week"):
            raise ValueError("Unsupported swing interval")
        if not isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if not isfinite(self.exposure_fraction) or not 0 < self.exposure_fraction <= 1:
            raise ValueError("exposure_fraction must be between zero and one")
        if not isfinite(self.max_drawdown) or not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be between zero and one")


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    position: int
    equity: float
    peak_equity: float


@dataclass(frozen=True)
class Execution:
    action: str
    submitted_at: datetime
    executed_at: datetime
    size: int
    price: float
    commission: float


@dataclass(frozen=True)
class EpisodeResult:
    initial_cash: float
    final_equity: float
    final_position: int
    peak_equity: float
    drawdown_triggered: bool
    executions: tuple[Execution, ...]

    def formatted_summary(self) -> dict[str, str]:
        profit = self.final_equity - self.initial_cash
        return {
            "initial_capital": f"{self.initial_cash:.2f}",
            "final_equity": f"{self.final_equity:.2f}",
            "profit_loss": f"{profit:.2f}",
            "profit_loss_percent": f"{100 * profit / self.initial_cash:.2f}",
            "peak_equity": f"{self.peak_equity:.2f}",
            "open_position_shares": f"{self.final_position:.2f}",
        }


def _short_exit_cutoff(interval: str) -> time:
    minutes = _INTERVAL_MINUTES[interval]
    session_open = datetime(2000, 1, 1, 9, 15)
    last_bar = session_open + timedelta(minutes=((375 - 1) // minutes) * minutes)
    return (last_bar - timedelta(minutes=minutes)).time()


class _ExecutionStrategy(bt.Strategy):
    params = (("config", None), ("policy", None), ("last_entry_timestamp", None))

    def __init__(self) -> None:
        self.pending = None
        self.executions: list[Execution] = []
        self.peak_equity = float(self.p.config.initial_cash)
        self.drawdown_triggered = False

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status != order.Completed:
            raise RuntimeError(
                f"Backtrader order {order.info.action} was {order.getstatusname()}"
            )
        executed_at = bt.num2date(order.executed.dt)
        submitted_at = order.info.submitted_at
        if executed_at <= submitted_at:
            raise RuntimeError("Market order did not execute after its decision bar")
        if order.info.short_order and executed_at.date() != submitted_at.date():
            raise RuntimeError("An intraday short order would execute in a later session")
        self.executions.append(
            Execution(
                action=order.info.action,
                submitted_at=submitted_at,
                executed_at=executed_at,
                size=int(order.executed.size),
                price=float(order.executed.price),
                commission=float(order.executed.comm),
            )
        )
        self.pending = None
        if self.drawdown_triggered and not self.position:
            self.env.runstop()

    def _submit(self, action: str, now: datetime, size: int | None = None) -> None:
        if action == "open_long":
            order = self.buy(size=size)
        elif action == "open_short":
            order = self.sell(size=size)
        else:
            order = self.close()
        if order is None:
            raise RuntimeError(f"Backtrader did not create an order for {action}")
        order.addinfo(
            action=action,
            submitted_at=now,
            short_order=action in ("open_short", "cover_short"),
        )
        self.pending = order

    def next(self) -> None:
        now = self.data.datetime.datetime(0)
        config: ExecutionConfig = self.p.config
        position = int(self.position.size)
        equity = float(self.broker.getvalue())
        self.peak_equity = max(self.peak_equity, equity)
        if position < 0 and config.style == "intraday":
            cutoff = _short_exit_cutoff(config.interval)
            if now.time() > cutoff:
                raise RuntimeError(
                    "Missing session-end bar prevented intraday short liquidation"
                )
        if equity <= self.peak_equity * (1.0 - config.max_drawdown):
            self.drawdown_triggered = True
            if position:
                self._submit("cover_short" if position < 0 else "close_long", now)
            else:
                self.env.runstop()
            return
        if position < 0 and config.style == "intraday":
            if now >= self.p.last_entry_timestamp:
                self._submit("cover_short", now)
                return
        if position < 0 and now.time() == _short_exit_cutoff(config.interval):
            self._submit("cover_short", now)
            return

        observation = Observation(
            timestamp=now,
            open=float(self.data.open[0]),
            high=float(self.data.high[0]),
            low=float(self.data.low[0]),
            close=float(self.data.close[0]),
            volume=float(self.data.volume[0]),
            position=position,
            equity=equity,
            peak_equity=self.peak_equity,
        )
        action: Action = self.p.policy(observation)
        valid = (
            {"wait", "open_long", "open_short"} if position == 0
            else {"hold", "close_long"} if position > 0
            else {"hold", "cover_short"}
        )
        if action not in valid:
            raise ValueError(f"Action {action!r} is invalid for position {position}")
        if action == "open_short":
            if config.style != "intraday":
                raise ValueError("Short positions are allowed only for intraday bars")
            if now >= self.p.last_entry_timestamp:
                raise ValueError("Not enough episode bars remain to close an intraday short")
            if now.time() >= _short_exit_cutoff(config.interval):
                raise ValueError("Not enough session time remains to open a short")
        if action in ("open_long", "open_short"):
            size = floor(max(equity, 0.0) * config.exposure_fraction / observation.close)
            if size < 1:
                raise ValueError("Insufficient equity to buy or short one whole share")
            self._submit(action, now, size)
        elif action in ("close_long", "cover_short"):
            self._submit(action, now)


def simulate_episode(
    bars: pd.DataFrame,
    *,
    config: ExecutionConfig,
    policy: Callable[[Observation], Action],
) -> EpisodeResult:
    """Run one symbol using Backtrader; decisions use completed-bar observations."""
    required = ("open", "high", "low", "close", "volume")
    if not isinstance(bars.index, pd.DatetimeIndex) or len(bars) < 2:
        raise ValueError("Bars must contain at least two rows and a DatetimeIndex")
    if not set(required).issubset(bars.columns):
        raise ValueError("Bars must include open, high, low, close and volume")
    frame = bars.loc[:, required].copy()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("Asia/Kolkata").tz_localize(None)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Bar timestamps must be unique and strictly increasing")
    if config.style == "intraday":
        minutes = _INTERVAL_MINUTES[config.interval]
        minute_of_day = frame.index.hour * 60 + frame.index.minute
        since_open = minute_of_day - (9 * 60 + 15)
        if (
            (since_open < 0).any()
            or (since_open >= 375).any()
            or (since_open % minutes != 0).any()
            or (frame.index.second != 0).any()
            or (frame.index.microsecond != 0).any()
            or (frame.index.nanosecond != 0).any()
        ):
            raise ValueError("Intraday timestamps must align with the 09:15-15:30 session")
    values = frame.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (values[:, :4] <= 0).any()
        or (values[:, 4] < 0).any()
    ):
        raise ValueError("Bars contain non-finite or invalid prices or volume")
    if (
        (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any()
        or (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any()
    ):
        raise ValueError("OHLC bars have inconsistent high or low prices")
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=frame))
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.costs.fee_rate)
    cerebro.broker.set_slippage_perc(
        config.costs.slippage_rate, slip_open=True, slip_out=True
    )
    cerebro.addstrategy(
        _ExecutionStrategy,
        config=config,
        policy=policy,
        last_entry_timestamp=frame.index[-2].to_pydatetime(),
    )
    strategy = cerebro.run(runonce=False, preload=False)[0]
    if strategy.pending is not None:
        raise RuntimeError("Data ended before a submitted market order could execute")
    if strategy.position.size < 0:
        raise RuntimeError("Data ended before the intraday short could be covered")
    if strategy.drawdown_triggered and strategy.position:
        raise RuntimeError("Data ended before the drawdown liquidation could execute")
    return EpisodeResult(
        initial_cash=config.initial_cash,
        final_equity=float(strategy.broker.getvalue()),
        final_position=int(strategy.position.size),
        peak_equity=strategy.peak_equity,
        drawdown_triggered=strategy.drawdown_triggered,
        executions=tuple(strategy.executions),
    )
