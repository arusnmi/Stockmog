from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any, Literal

import polars as pl

from .execution_simulator import (
    ExecutionCosts,
    ExitReason,
    calculate_quantity,
    resolve_exit,
    session_date,
)
from .performance import summarize_by_year, summarize_performance

BacktestStyle = Literal["intraday", "swing"]

TRADE_COLUMNS = {
    "symbol": pl.String,
    "strategy": pl.String,
    "style": pl.String,
    "direction": pl.String,
    "signal_timestamp": pl.Datetime("us"),
    "entry_timestamp": pl.Datetime("us"),
    "exit_timestamp": pl.Datetime("us"),
    "entry_price": pl.Float64,
    "stop_price": pl.Float64,
    "target_price": pl.Float64,
    "exit_price": pl.Float64,
    "quantity": pl.Int64,
    "exit_reason": pl.String,
    "gross_pnl": pl.Float64,
    "fees": pl.Float64,
    "slippage_cost": pl.Float64,
    "transaction_cost": pl.Float64,
    "net_pnl": pl.Float64,
    "entry_notional": pl.Float64,
    "exit_notional": pl.Float64,
    "initial_risk": pl.Float64,
    "return_on_risk": pl.Float64,
}

EQUITY_COLUMNS = {
    "timestamp": pl.Datetime("us"),
    "date": pl.Date,
    "equity": pl.Float64,
    "cash": pl.Float64,
    "open_positions": pl.UInt32,
    "open_risk": pl.Float64,
}


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    risk_per_trade: float = 0.0025
    max_open_risk: float = 0.01
    max_daily_loss: float = 0.01
    max_drawdown: float = 0.05
    max_positions: int | None = None
    costs: ExecutionCosts = ExecutionCosts()

    def __post_init__(self) -> None:
        if not isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be finite and greater than zero")
        for name in ("risk_per_trade", "max_open_risk", "max_daily_loss", "max_drawdown"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0 or value > 1:
                raise ValueError(f"{name} must be greater than zero and at most one")
        if self.risk_per_trade > self.max_open_risk:
            raise ValueError("risk_per_trade must not exceed max_open_risk")
        if self.max_positions is not None and self.max_positions < 1:
            raise ValueError("max_positions must be at least one")


@dataclass
class Position:
    symbol: str
    strategy: str
    style: str
    direction: int
    signal_timestamp: datetime
    entry_timestamp: datetime
    entry_price: float
    stop_price: float
    target_price: float
    quantity: int
    initial_risk: float


@dataclass(frozen=True)
class BacktestResult:
    trades: pl.DataFrame
    equity_curve: pl.DataFrame
    summary: dict[str, Any]
    yearly: pl.DataFrame


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _validate_inputs(
    bars: pl.DataFrame,
    signals: pl.DataFrame,
    style: BacktestStyle,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if style not in ("intraday", "swing"):
        raise ValueError("style must be 'intraday' or 'swing'")
    required_bars = {"timestamp", "open", "high", "low", "close"}
    missing_bars = required_bars - set(bars.columns)
    if missing_bars:
        raise ValueError(
            "Backtest bars are missing columns: " + ", ".join(sorted(missing_bars))
        )
    required_signals = {
        "symbol",
        "timestamp",
        "strategy",
        "direction",
        "entry_price",
        "stop_price",
        "target_price",
        "reward_risk",
    }
    missing_signals = required_signals - set(signals.columns)
    if missing_signals:
        raise ValueError(
            "Backtest signals are missing columns: "
            + ", ".join(sorted(missing_signals))
        )
    if not isinstance(bars.schema["timestamp"], pl.Datetime):
        raise ValueError("Backtest bar timestamps must be datetimes")
    if not isinstance(signals.schema["timestamp"], pl.Datetime):
        raise ValueError("Signal timestamps must be datetimes")
    bar_timezone = bars.schema["timestamp"].time_zone
    signal_timezone = signals.schema["timestamp"].time_zone
    if bar_timezone != signal_timezone:
        raise ValueError("Backtest bars and signals must use the same timestamp timezone")

    bars = bars.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", bar_timezone)),
        *(
            [pl.col("symbol").cast(pl.String)]
            if "symbol" in bars.columns
            else [pl.lit("UNKNOWN").alias("symbol")]
        ),
        *[
            pl.col(column).cast(pl.Float64, strict=True)
            for column in ("open", "high", "low", "close")
        ],
    ).select("timestamp", "symbol", "open", "high", "low", "close")
    signals = signals.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", signal_timezone)),
        pl.col("symbol").cast(pl.String),
        pl.col("strategy").cast(pl.String),
        pl.col("direction").cast(pl.String),
        *[
            pl.col(column).cast(pl.Float64, strict=True)
            for column in (
                "entry_price",
                "stop_price",
                "target_price",
                "reward_risk",
            )
        ],
    )
    if bars.select(
        pl.any_horizontal(
            pl.col("timestamp").is_null(),
            pl.col("symbol").is_null(),
            pl.col("symbol").str.strip_chars().eq(""),
        ).any()
    ).item():
        raise ValueError("Backtest bars contain a null timestamp or empty symbol")
    invalid_signals = signals.filter(
        pl.any_horizontal(
            pl.col("timestamp").is_null(),
            pl.col("symbol").is_null(),
            pl.col("symbol").str.strip_chars().eq(""),
            pl.col("strategy").is_null(),
            ~pl.col("direction").is_in(["long", "short"]),
            *[
                pl.col(column).is_null() | ~pl.col(column).is_finite()
                for column in (
                    "entry_price",
                    "stop_price",
                    "target_price",
                    "reward_risk",
                )
            ],
            pl.col("entry_price") <= 0,
            pl.col("stop_price") <= 0,
            pl.col("target_price") <= 0,
            pl.col("reward_risk") <= 0,
        )
    )
    if invalid_signals.height:
        raise ValueError(
            f"Signals contain {invalid_signals.height} invalid row(s)"
        )
    has_null_ohlc = bars.select(
        pl.any_horizontal(
            *[
                pl.col(column).is_null()
                for column in ("open", "high", "low", "close")
            ]
        ).any()
    ).item()
    if has_null_ohlc:
        raise ValueError("Backtest bars contain null OHLC prices")
    invalid_ohlc = bars.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") < pl.max_horizontal("open", "low", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "high", "close"))
        | pl.any_horizontal(
            *[~pl.col(column).is_finite() for column in ("open", "high", "low", "close")]
        )
    )
    if invalid_ohlc.height:
        raise ValueError(f"Backtest bars contain {invalid_ohlc.height} invalid OHLC row(s)")
    if bars.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item():
        raise ValueError("Backtest bars contain duplicate symbol/timestamp rows")
    if signals.select(pl.struct("symbol", "timestamp", "strategy").is_duplicated().any()).item():
        raise ValueError("Signals contain duplicate symbol/timestamp/strategy rows")
    bar_keys = set(
        bars.select("symbol", "timestamp").iter_rows()
    )
    unmatched_signals = [
        (row["symbol"], row["timestamp"])
        for row in signals.select("symbol", "timestamp").iter_rows(named=True)
        if (row["symbol"], row["timestamp"]) not in bar_keys
    ]
    if unmatched_signals:
        raise ValueError(
            f"Signals contain {len(unmatched_signals)} timestamp(s) without matching bars"
        )
    return bars.sort(["timestamp", "symbol"]), signals.sort(
        ["timestamp", "symbol", "strategy"]
    )


def run_backtest(
    bars: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    style: BacktestStyle,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Simulate close-generated signals at the next bar with portfolio risk limits."""
    config = config or BacktestConfig()
    bars, signals = _validate_inputs(bars, signals, style)
    if bars.is_empty():
        raise ValueError("Backtest requires at least one OHLC bar")

    bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in bars.iter_rows(named=True):
        bars_by_symbol.setdefault(row["symbol"], []).append(row)
    signal_rows: dict[tuple[str, datetime], list[dict[str, Any]]] = {}
    for row in signals.iter_rows(named=True):
        signal_rows.setdefault((row["symbol"], row["timestamp"]), []).append(row)

    last_session_bar = {
        (symbol, session_date(rows[index]["timestamp"])): index
        for symbol, rows in bars_by_symbol.items()
        for index in range(len(rows))
        if index + 1 == len(rows)
        or session_date(rows[index + 1]["timestamp"]) != session_date(rows[index]["timestamp"])
    }
    positions: dict[str, Position] = {}
    pending: dict[str, tuple[dict[str, Any], datetime]] = {}
    trades: list[dict[str, Any]] = []
    equity_records: list[dict[str, Any]] = []
    equity = config.initial_capital
    peak_equity = equity
    current_day: date | None = None
    day_start_equity = equity
    last_marked_equity = equity
    halted_days: set[date] = set()
    drawdown_halted = False
    last_prices: dict[str, tuple[datetime, float]] = {}

    def close_position(
        position: Position,
        *,
        timestamp: datetime,
        raw_exit_price: float,
        reason: ExitReason,
    ) -> None:
        nonlocal equity
        exit_price = config.costs.exit_fill(raw_exit_price, position.direction)
        raw_entry_price = position.entry_price / (
            1.0 + position.direction * config.costs.slippage_rate
        )
        entry_notional = position.entry_price * position.quantity
        exit_notional = exit_price * position.quantity
        gross_pnl = position.direction * position.quantity * (
            raw_exit_price - raw_entry_price
        )
        slippage_cost = position.direction * position.quantity * (
            (position.entry_price - raw_entry_price)
            + (raw_exit_price - exit_price)
        )
        fees = config.costs.fees(entry_notional, exit_notional)
        transaction_cost = fees + slippage_cost
        net_pnl = gross_pnl - transaction_cost
        equity += net_pnl
        trades.append(
            {
                "symbol": position.symbol,
                "strategy": position.strategy,
                "style": position.style,
                "direction": "long" if position.direction == 1 else "short",
                "signal_timestamp": position.signal_timestamp,
                "entry_timestamp": position.entry_timestamp,
                "exit_timestamp": timestamp,
                "entry_price": position.entry_price,
                "stop_price": position.stop_price,
                "target_price": position.target_price,
                "exit_price": exit_price,
                "quantity": position.quantity,
                "exit_reason": reason,
                "gross_pnl": gross_pnl,
                "fees": fees,
                "slippage_cost": slippage_cost,
                "transaction_cost": transaction_cost,
                "net_pnl": net_pnl,
                "entry_notional": entry_notional,
                "exit_notional": exit_notional,
                "initial_risk": position.initial_risk,
                "return_on_risk": net_pnl / position.initial_risk
                if position.initial_risk
                else 0.0,
            }
        )

    def mark_to_market(
        timestamp: datetime,
        row_by_symbol: dict[str, dict[str, Any]],
        *,
        record: bool = True,
    ) -> float:
        unrealized = 0.0
        open_risk = 0.0
        open_notional = 0.0
        reserved_entry_fees = 0.0
        for symbol, position in positions.items():
            bar = row_by_symbol.get(symbol)
            market_price = (
                bar["close"]
                if bar is not None
                else last_prices.get(symbol, (None, 0.0))[1]
            )
            if market_price:
                estimated_exit_price = config.costs.exit_fill(
                    market_price,
                    position.direction,
                )
                entry_notional = position.entry_price * position.quantity
                exit_notional = estimated_exit_price * position.quantity
                unrealized += (
                    position.direction
                    * position.quantity
                    * (estimated_exit_price - position.entry_price)
                    - config.costs.fees(entry_notional, exit_notional)
                )
            open_notional += position.entry_price * position.quantity
            reserved_entry_fees += (
                position.entry_price
                * position.quantity
                * config.costs.fee_rate
            )
            open_risk += (
                abs(position.entry_price - position.stop_price) * position.quantity
                + config.costs.estimated_round_trip_cost_per_unit(position.entry_price)
                * position.quantity
            )
        marked_equity = equity + unrealized
        if record:
            equity_records.append(
                {
                    "timestamp": timestamp,
                    "date": session_date(timestamp),
                    "equity": marked_equity,
                    "cash": max(0.0, equity - open_notional - reserved_entry_fees),
                    "open_positions": len(positions),
                    "open_risk": open_risk,
                }
            )
        return marked_equity

    def close_all_positions(
        *,
        timestamp: datetime,
        row_by_symbol: dict[str, dict[str, Any]],
        reason: ExitReason,
    ) -> None:
        for symbol, position in list(positions.items()):
            bar = row_by_symbol.get(symbol)
            last_price = last_prices.get(symbol)
            if bar is None and last_price is None:
                continue
            close_position(
                position,
                timestamp=timestamp,
                raw_exit_price=bar["close"] if bar is not None else last_price[1],
                reason=reason,
            )
            del positions[symbol]

    timeline = sorted(
        {
            row["timestamp"]
            for symbol_rows in bars_by_symbol.values()
            for row in symbol_rows
        }
    )
    bar_indices = {
        symbol: {row["timestamp"]: index for index, row in enumerate(rows)}
        for symbol, rows in bars_by_symbol.items()
    }

    for timestamp in timeline:
        day = session_date(timestamp)
        if day != current_day:
            current_day = day
            day_start_equity = last_marked_equity

        current_rows = {
            symbol: rows[index]
            for symbol, rows in bars_by_symbol.items()
            if (index := bar_indices[symbol].get(timestamp)) is not None
        }
        for symbol, bar in current_rows.items():
            last_prices[symbol] = (timestamp, bar["close"])

        for symbol, position in list(positions.items()):
            bar = current_rows.get(symbol)
            if bar is None:
                continue
            exit_fill = resolve_exit(
                direction=position.direction,
                open_price=bar["open"],
                high=bar["high"],
                low=bar["low"],
                stop_price=position.stop_price,
                target_price=position.target_price,
            )
            if exit_fill is not None:
                close_position(
                    position,
                    timestamp=timestamp,
                    raw_exit_price=exit_fill.price,
                    reason=exit_fill.reason,
                )
                del positions[symbol]

        pre_entry_equity = mark_to_market(timestamp, current_rows, record=False)
        peak_equity = max(peak_equity, pre_entry_equity)
        if pre_entry_equity <= day_start_equity * (1.0 - config.max_daily_loss):
            halted_days.add(day)
            close_all_positions(
                timestamp=timestamp,
                row_by_symbol=current_rows,
                reason="daily_loss_limit",
            )
        if pre_entry_equity <= peak_equity * (1.0 - config.max_drawdown):
            drawdown_halted = True

        for symbol, bar in current_rows.items():
            pending_signal = pending.pop(symbol, None)
            if pending_signal is None or symbol in positions:
                continue
            signal, signal_day = pending_signal
            if style == "intraday" and signal_day != day:
                continue
            if day in halted_days or drawdown_halted:
                continue
            if len(positions) >= (
                config.max_positions
                if config.max_positions is not None
                else (3 if style == "intraday" else 5)
            ):
                continue

            signal_direction = signal["direction"]
            if signal_direction not in ("long", "short"):
                continue
            direction = 1 if signal_direction == "long" else -1
            reference = float(signal["entry_price"])
            stop_reference = float(signal["stop_price"])
            target_reference = float(signal["target_price"])
            reward_risk = float(signal["reward_risk"])
            values = (reference, stop_reference, target_reference, reward_risk)
            if not all(isfinite(value) for value in values) or reward_risk <= 0:
                continue
            stop_distance = abs(reference - stop_reference)
            if stop_distance <= 0:
                continue
            if (
                direction == 1
                and not (stop_reference < reference < target_reference)
            ) or (
                direction == -1
                and not (target_reference < reference < stop_reference)
            ):
                continue

            risk_equity = mark_to_market(timestamp, current_rows, record=False)
            entry_price = config.costs.entry_fill(bar["open"], direction)
            quantity = calculate_quantity(
                equity=risk_equity,
                risk_fraction=config.risk_per_trade,
                entry_price=entry_price,
                stop_distance=stop_distance,
                costs=config.costs,
            )
            open_notional = sum(
                position.entry_price * position.quantity
                for position in positions.values()
            )
            available_cash = max(0.0, equity - open_notional)
            available_cash -= sum(
                position.entry_price
                * position.quantity
                * config.costs.fee_rate
                for position in positions.values()
            )
            cash_per_unit = entry_price * (1.0 + config.costs.fee_rate)
            quantity = min(quantity, int(available_cash / cash_per_unit))
            if quantity < 1:
                continue
            planned_risk = (
                stop_distance
                + config.costs.estimated_round_trip_cost_per_unit(entry_price)
            ) * quantity
            open_risk = sum(
                (
                    abs(position.entry_price - position.stop_price)
                    + config.costs.estimated_round_trip_cost_per_unit(
                        position.entry_price
                    )
                )
                * position.quantity
                for position in positions.values()
            )
            risk_limit = risk_equity * config.max_open_risk
            remaining_risk = risk_limit - open_risk
            if planned_risk > remaining_risk:
                allowed_quantity = int(remaining_risk / max(planned_risk / quantity, 1e-12))
                quantity = min(quantity, allowed_quantity)
                if quantity < 1:
                    continue
                planned_risk = (
                    stop_distance
                    + config.costs.estimated_round_trip_cost_per_unit(entry_price)
                ) * quantity

            stop_price = entry_price - direction * stop_distance
            target_price = entry_price + direction * stop_distance * reward_risk
            position = Position(
                symbol=symbol,
                strategy=str(signal["strategy"]),
                style=style,
                direction=direction,
                signal_timestamp=signal["timestamp"],
                entry_timestamp=timestamp,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quantity=quantity,
                initial_risk=planned_risk,
            )
            positions[symbol] = position
            entry_exit = resolve_exit(
                direction=position.direction,
                open_price=bar["open"],
                high=bar["high"],
                low=bar["low"],
                stop_price=position.stop_price,
                target_price=position.target_price,
            )
            if entry_exit is not None:
                close_position(
                    position,
                    timestamp=timestamp,
                    raw_exit_price=entry_exit.price,
                    reason=entry_exit.reason,
                )
                del positions[symbol]
                post_exit_equity = mark_to_market(
                    timestamp,
                    current_rows,
                    record=False,
                )
                if post_exit_equity <= day_start_equity * (
                    1.0 - config.max_daily_loss
                ):
                    halted_days.add(day)
                    close_all_positions(
                        timestamp=timestamp,
                        row_by_symbol=current_rows,
                        reason="daily_loss_limit",
                    )
                peak_equity = max(peak_equity, post_exit_equity)
                if post_exit_equity <= peak_equity * (1.0 - config.max_drawdown):
                    drawdown_halted = True

        if style == "intraday":
            for symbol, position in list(positions.items()):
                bar = current_rows.get(symbol)
                if bar is None:
                    continue
                index = bar_indices[symbol][timestamp]
                if last_session_bar[(symbol, day)] == index:
                    close_position(
                        position,
                        timestamp=timestamp,
                        raw_exit_price=bar["close"],
                        reason="session_end",
                    )
                    del positions[symbol]
        marked_equity = mark_to_market(timestamp, current_rows)
        last_marked_equity = marked_equity
        peak_equity = max(peak_equity, marked_equity)
        if marked_equity <= day_start_equity * (1.0 - config.max_daily_loss):
            halted_days.add(day)
            close_all_positions(
                timestamp=timestamp,
                row_by_symbol=current_rows,
                reason="daily_loss_limit",
            )
            marked_equity = mark_to_market(timestamp, current_rows)
            last_marked_equity = marked_equity
        if marked_equity <= peak_equity * (1.0 - config.max_drawdown):
            drawdown_halted = True

        for symbol, bar in current_rows.items():
            signal_candidates = signal_rows.get((symbol, timestamp), [])
            if not signal_candidates:
                continue
            index = bar_indices[symbol][timestamp]
            if index + 1 >= len(bars_by_symbol[symbol]):
                continue
            next_bar = bars_by_symbol[symbol][index + 1]
            if style == "intraday" and session_date(next_bar["timestamp"]) != day:
                continue
            pending.setdefault(symbol, (signal_candidates[0], day))

    for symbol, position in list(positions.items()):
        last_bar = bars_by_symbol[symbol][-1]
        close_position(
            position,
            timestamp=last_bar["timestamp"],
            raw_exit_price=last_bar["close"],
            reason="end_of_data",
        )
        del positions[symbol]
    if equity_records:
        final_timestamp = max(row["timestamp"] for row in equity_records)
        equity_records.append(
            {
                "timestamp": final_timestamp,
                "date": session_date(final_timestamp),
                "equity": equity,
                "cash": equity,
                "open_positions": 0,
                "open_risk": 0.0,
            }
        )

    trades_frame = (
        pl.DataFrame(
            trades,
            schema={
                **TRADE_COLUMNS,
                "signal_timestamp": bars.schema["timestamp"],
                "entry_timestamp": bars.schema["timestamp"],
                "exit_timestamp": bars.schema["timestamp"],
            },
        )
        if trades
        else _empty_frame(TRADE_COLUMNS)
    )
    equity_frame = (
        pl.DataFrame(
            equity_records,
            schema={
                **EQUITY_COLUMNS,
                "timestamp": bars.schema["timestamp"],
            },
        )
        if equity_records
        else _empty_frame(EQUITY_COLUMNS)
    )
    if positions:
        raise RuntimeError("Backtest ended with open positions.")
    summary = summarize_performance(
        trades_frame,
        equity_frame,
        initial_capital=config.initial_capital,
    )
    summary.update(
        {
            "risk_per_trade": config.risk_per_trade,
            "max_open_risk": config.max_open_risk,
            "max_daily_loss": config.max_daily_loss,
            "max_drawdown_limit": config.max_drawdown,
            "max_positions": config.max_positions or (3 if style == "intraday" else 5),
            "commission_bps_per_side": config.costs.commission_bps,
            "other_fees_bps_per_side": config.costs.other_fees_bps,
            "slippage_bps_per_side": config.costs.slippage_bps,
        }
    )
    return BacktestResult(
        trades=trades_frame,
        equity_curve=equity_frame,
        summary=summary,
        yearly=summarize_by_year(
            trades_frame,
            equity_frame,
            initial_capital=config.initial_capital,
        ),
    )
