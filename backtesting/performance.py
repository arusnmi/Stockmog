from __future__ import annotations

from math import sqrt
from typing import Any

import polars as pl


def _drawdown(equity_values: list[float]) -> float:
    peak = 0.0
    max_drawdown = 0.0
    for equity in equity_values:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return max_drawdown


def summarize_performance(
    trades: pl.DataFrame,
    equity_curve: pl.DataFrame,
    *,
    initial_capital: float,
) -> dict[str, Any]:
    """Summarize closed trades and daily marked-to-market equity."""
    net_values = trades["net_pnl"].to_list() if trades.height else []
    gross_values = trades["gross_pnl"].to_list() if trades.height else []
    costs = trades["transaction_cost"].to_list() if trades.height else []
    fees = trades["fees"].to_list() if trades.height else []
    slippage_costs = (
        trades["slippage_cost"].to_list() if trades.height else []
    )
    winners = [value for value in net_values if value > 0]
    losers = [value for value in net_values if value < 0]
    total_net = sum(net_values)
    total_gross = sum(gross_values)
    total_cost = sum(costs)
    ending_equity = initial_capital + total_net
    daily_equity = (
        equity_curve.group_by("date")
        .agg(pl.col("equity").last().alias("equity"))
        .sort("date")
        if equity_curve.height
        else pl.DataFrame({"date": [], "equity": []})
    )
    daily_values = daily_equity["equity"].to_list()
    daily_returns = [
        (current - previous) / previous
        for previous, current in zip(
            [initial_capital, *daily_values[:-1]],
            daily_values,
        )
        if previous > 0
    ]
    mean_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0.0
    variance = (
        sum((value - mean_return) ** 2 for value in daily_returns)
        / (len(daily_returns) - 1)
        if len(daily_returns) > 1
        else 0.0
    )
    downside = [min(value, 0.0) for value in daily_returns]
    downside_deviation = (
        sqrt(sum(value**2 for value in downside) / len(downside))
        if downside
        else 0.0
    )
    sharpe = (
        mean_return / sqrt(variance) * sqrt(252)
        if variance > 0
        else None
    )
    sortino = (
        mean_return / downside_deviation * sqrt(252)
        if downside_deviation > 0
        else None
    )
    equity_values = (
        equity_curve["equity"].to_list()
        if equity_curve.height
        else [initial_capital]
    )
    equity_values.insert(0, initial_capital)
    max_drawdown = _drawdown(equity_values)
    losing_total = abs(sum(losers))

    return {
        "initial_capital": initial_capital,
        "ending_equity": ending_equity,
        "net_return_pct": (ending_equity / initial_capital - 1.0) * 100.0,
        "gross_pnl": total_gross,
        "transaction_costs": total_cost,
        "fees": sum(fees),
        "slippage_costs": sum(slippage_costs),
        "net_pnl": total_net,
        "trade_count": len(net_values),
        "win_rate_pct": len(winners) / len(net_values) * 100.0 if net_values else 0.0,
        "average_win": sum(winners) / len(winners) if winners else 0.0,
        "average_loss": sum(losers) / len(losers) if losers else 0.0,
        "expectancy_per_trade": total_net / len(net_values) if net_values else 0.0,
        "profit_factor": (
            sum(winners) / losing_total
            if losing_total > 0
            else (None if not winners else float("inf"))
        ),
        "max_drawdown_pct": max_drawdown * 100.0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "turnover": (
            trades["entry_notional"].sum() + trades["exit_notional"].sum()
            if trades.height
            else 0.0
        ),
    }


def summarize_by_year(
    trades: pl.DataFrame,
    equity_curve: pl.DataFrame,
    *,
    initial_capital: float,
) -> pl.DataFrame:
    """Compare marked-to-market annual returns and realized trade outcomes."""
    yearly_trades: dict[int, dict[str, float | int]] = {}
    if trades.height:
        exit_timestamp = pl.col("exit_timestamp")
        if trades.schema["exit_timestamp"].time_zone is not None:
            exit_year = exit_timestamp.dt.convert_time_zone("Asia/Kolkata").dt.year()
        else:
            exit_year = exit_timestamp.dt.year()
        for row in (
            trades.with_columns(exit_year.alias("year"))
            .group_by("year")
            .agg(
                pl.len().alias("trade_count"),
                pl.col("net_pnl").sum().alias("net_pnl"),
                pl.col("transaction_cost").sum().alias("transaction_costs"),
                (pl.col("net_pnl") > 0).mean().mul(100).alias("win_rate_pct"),
            )
            .iter_rows(named=True)
        ):
            yearly_trades[row["year"]] = row

    daily_equity = (
        equity_curve.group_by("date")
        .agg(pl.col("equity").last().alias("equity"))
        .sort("date")
        .with_columns(pl.col("date").dt.year().alias("year"))
    )
    grouped_equity: dict[int, list[float]] = {}
    for row in daily_equity.iter_rows(named=True):
        grouped_equity.setdefault(row["year"], []).append(row["equity"])

    yearly_rows = []
    starting_equity = initial_capital
    for year, values in sorted(grouped_equity.items()):
        ending_equity = values[-1]
        peak = starting_equity
        max_drawdown = 0.0
        for value in values:
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - value) / peak)
        trade_summary = yearly_trades.get(year, {})
        yearly_rows.append(
            {
                "year": year,
                "starting_equity": starting_equity,
                "ending_equity": ending_equity,
                "net_return_pct": (
                    (ending_equity / starting_equity - 1.0) * 100.0
                    if starting_equity > 0
                    else 0.0
                ),
                "max_drawdown_pct": max_drawdown * 100.0,
                "trade_count": trade_summary.get("trade_count", 0),
                "net_pnl": trade_summary.get("net_pnl", 0.0),
                "transaction_costs": trade_summary.get("transaction_costs", 0.0),
                "win_rate_pct": trade_summary.get("win_rate_pct", 0.0),
            }
        )
        starting_equity = ending_equity

    return pl.DataFrame(
        yearly_rows,
        schema={
            "year": pl.Int32,
            "starting_equity": pl.Float64,
            "ending_equity": pl.Float64,
            "net_return_pct": pl.Float64,
            "max_drawdown_pct": pl.Float64,
            "trade_count": pl.UInt32,
            "net_pnl": pl.Float64,
            "transaction_costs": pl.Float64,
            "win_rate_pct": pl.Float64,
        },
    )
