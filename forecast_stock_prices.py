"""Plot a historical, fixed-cutoff trend baseline against subsequent closes."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import requests
import yfinance as yf


SYMBOLS = ("MUTHOOTMF", "RKSWAMY", "OBEROIRLTY", "SAKAR", "BOSCHLTD")
CUTOFF = date(2026, 1, 29)
SPECIAL_SESSIONS = {
    date(2026, 2, 1): "Union Budget",
    date(2026, 11, 8): "Muhurat trading",
}
CLOCK_URL = "https://timeapi.io/api/time/current/zone"
HOLIDAYS_URL = "https://marketholidays.in/api/holidays"
ARCHIVE_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{day}_F_0000.csv.zip"
)


def _get_json(url: str, *, params: dict[str, object]) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return value


def market_clock() -> datetime:
    data = _get_json(CLOCK_URL, params={"timeZone": "Asia/Kolkata"})
    if data["timeZone"] != "Asia/Kolkata":
        raise ValueError("Clock API did not return India market time")
    return datetime.fromisoformat(data["dateTime"])


def next_friday(today: date) -> date:
    days = (4 - today.weekday()) % 7 or 7
    return today + timedelta(days=days)


def calendar_days(start: date, end: date, holidays: dict[date, str]) -> list[tuple[date, str]]:
    if end < start:
        raise ValueError("End date must be after the forecast cutoff")
    days = []
    current = start
    while current <= end:
        if current in SPECIAL_SESSIONS:
            status = "special: " + SPECIAL_SESSIONS[current]
        elif current in holidays:
            status = "holiday: " + holidays[current]
        elif current.weekday() >= 5:
            status = "weekend"
        else:
            status = "trading"
        days.append((current, status))
        current += timedelta(days=1)
    return days


def fetch_holidays(start: date, end: date) -> dict[date, str]:
    holidays = {}
    for year in range(start.year, end.year + 1):
        data = _get_json(HOLIDAYS_URL, params={"year": year})
        if data.get("year") != year or not isinstance(data.get("nse"), list) or not data["nse"]:
            raise ValueError(f"Invalid or empty NSE holiday calendar for {year}")
        for item in data["nse"]:
            day = date.fromisoformat(item["date"])
            if day.year != year:
                raise ValueError(f"Unexpected holiday date: {day}")
            holidays[day] = item["name"]
    return holidays


def make_forecasts(closes: list[float], sessions: list[date]) -> dict[date, float]:
    if len(closes) < 21 or any(not math.isfinite(x) or x <= 0 for x in closes[-21:]):
        raise ValueError("At least 21 valid historical closing prices are required")
    last = closes[-1]
    drift = math.log(last / closes[-21]) / 20
    return {day: last * math.exp(drift * index) for index, day in enumerate(sessions, 1)}


def history(symbol: str, start: date, end: date) -> dict[date, float]:
    data = yf.Ticker(symbol + ".NS").history(
        start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    if data.empty:
        raise ValueError(f"No Yahoo Finance price data for {symbol}, {start} through {end}")
    result = {}
    for stamp, value in data["Close"].items():
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"Invalid closing price for {symbol} on {stamp}")
        day = stamp.date()
        if day in result:
            raise ValueError(f"Duplicate closing price for {symbol} on {day}")
        result[day] = number
    return result


def special_closes(day: date, symbols: tuple[str, ...]) -> dict[str, float]:
    url = ARCHIVE_URL.format(day=day.strftime("%Y%m%d"))
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Unexpected NSE archive contents for {day}: {names}")
        with archive.open(names[0]) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            prices = {
                row["TckrSymb"]: float(row["ClsPric"])
                for row in reader
                if row["TckrSymb"] in symbols
                and row["SctySrs"] == "EQ"
                and row["TradDt"] == day.isoformat()
            }
    if set(prices) != set(symbols) or any(not math.isfinite(x) or x <= 0 for x in prices.values()):
        raise ValueError(f"Incomplete or invalid NSE special-session closes for {day}")
    return prices


def write_chart(path: Path, rows: list[dict], end: date, cutoff: date) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install dependencies: python -m pip install -r requirements-forecast.txt") from exc

    fig, axes = plt.subplots(len(SYMBOLS), 1, figsize=(14, 18), layout="constrained")
    for ax, symbol in zip(axes, SYMBOLS):
        selected = [row for row in rows if row["symbol"] == symbol and row["forecast"]]
        x = list(range(len(selected)))
        predicted = [float(row["forecast"]) for row in selected]
        actual = [float(row["actual_close"]) if row["actual_close"] else math.nan for row in selected]
        ax.plot(x, predicted, color="#2461b3", linewidth=1.5, label="20-session trend forecast")
        ax.plot(x, actual, color="#b83232", linewidth=1.5, label="Historical actual close")
        ax.set_title(symbol, loc="left", fontweight="bold")
        ax.set_ylabel("Close (INR)")
        step = max(1, math.ceil(len(selected) / 10))
        ticks = list(range(0, len(selected), step))
        if ticks and ticks[-1] != len(selected) - 1:
            ticks.append(len(selected) - 1)
        ax.set_xticks(ticks, [date.fromisoformat(selected[i]["date"]).strftime("%d %b")
                              for i in ticks])
        ax.set_xlabel("Trading sessions")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Fixed cutoff: {cutoff} | Forecast through {end}", fontsize=16)
    fig.text(0.5, -0.01, "Trend baseline, not the saved trading model. Holiday calendar: marketholidays.in.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(end_override: date | None, output_root: Path, *, latest: bool = False) -> Path:
    now = market_clock()
    end = end_override or next_friday(now.date())
    if end.year != 2026:
        raise ValueError(
            f"Special NSE sessions for {end.year} have not been verified. "
            "Update the script's calendar before running that year."
        )
    completed = now.date() if now.time() >= time(16, 0) else now.date() - timedelta(days=1)
    calendar_start = (completed if latest else CUTOFF) - timedelta(days=62)
    holidays = fetch_holidays(calendar_start, max(end, completed) if latest else end)
    if latest:
        completed_sessions = [
            day for day, status in calendar_days(calendar_start, completed, holidays)
            if status == "trading" or status.startswith("special:")
        ]
        if not completed_sessions:
            raise ValueError("No completed trading sessions found")
        cutoff = completed_sessions[-1]
    else:
        cutoff = CUTOFF
    if end <= cutoff:
        raise ValueError(f"End date must be after {cutoff}")
    history_start = cutoff - timedelta(days=55)
    days = calendar_days(cutoff + timedelta(days=1), end, holidays)
    history_sessions = {
        day for day, status in calendar_days(history_start, cutoff, holidays)
        if status == "trading" or status.startswith("special:")
    }
    sessions = [day for day, status in days if status == "trading" or status.startswith("special:")]
    latest_actual = max((day for day in sessions if day <= completed), default=cutoff)

    predictions = {}
    actuals = {}
    recent_sessions = sorted(history_sessions)[-21:]
    if len(recent_sessions) < 21:
        raise ValueError("The historical calendar has fewer than 21 trading sessions")
    history_special = {
        day: special_closes(day, SYMBOLS) for day in recent_sessions if day in SPECIAL_SESSIONS
    }
    for symbol in SYMBOLS:
        before = history(symbol, history_start, cutoff)
        for day, prices in history_special.items():
            before[day] = prices[symbol]
        missing_history = [day for day in recent_sessions if day not in before]
        if missing_history:
            raise ValueError(f"Missing historical closes for {symbol}: {missing_history}")
        closes = [before[day] for day in recent_sessions]
        predictions[symbol] = make_forecasts(closes, sessions)
        actuals[symbol] = (
            history(symbol, cutoff + timedelta(days=1), latest_actual)
            if latest_actual > cutoff else {}
        )
    for day in SPECIAL_SESSIONS:
        if cutoff < day <= latest_actual and day <= end:
            for symbol, value in special_closes(day, SYMBOLS).items():
                actuals[symbol][day] = value

    rows = []
    errors = []
    for symbol in SYMBOLS:
        for day, status in days:
            if status != "trading" and not status.startswith("special:"):
                continue
            forecast = predictions[symbol][day]
            actual = actuals[symbol].get(day)
            error = 100 * abs(forecast - actual) / actual if actual is not None else None
            if error is not None:
                errors.append(error)
            rows.append({
                "symbol": symbol, "date": day.isoformat(), "market_status": status,
                "forecast": f"{forecast:.2f}",
                "actual_close": f"{actual:.2f}" if actual is not None else "",
                "absolute_error_percent": f"{error:.2f}" if error is not None else "",
            })
        missing = [d for d in sessions if d <= latest_actual and d not in actuals[symbol]]
        if missing:
            print(f"Warning: {symbol} has {len(missing)} missing historical closes: "
                  + ", ".join(day.isoformat() for day in missing), flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    prefix = "forecast_latest" if latest else "forecast"
    output = output_root / f"{prefix}_{cutoff}_{end}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir()
    with (output / "daily_prices.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "symbol", "date", "market_status", "forecast", "actual_close",
            "absolute_error_percent",
        ])
        writer.writeheader()
        writer.writerows(rows)
    write_chart(output / "stock_graphs.png", rows, end, cutoff)
    metadata = {
        "cutoff": cutoff.isoformat(), "end": end.isoformat(), "clock_ist": now.isoformat(),
        "latest_actual_date": latest_actual.isoformat(), "symbols": SYMBOLS,
        "method": "20-session geometric trend, fixed at cutoff; not the saved trading model",
        "calendar_source": HOLIDAYS_URL,
        "special_sessions": {day.isoformat(): name for day, name in SPECIAL_SESSIONS.items()},
        "price_sources": ["Yahoo Finance", "NSE special-session bhavcopy"],
        "compared_stock_days": len(errors),
        "mean_absolute_percentage_error": (
            f"{math.fsum(errors) / len(errors):.2f}" if errors else None
        ),
        "limitations": [
            "Long-horizon extrapolation is unreliable and not a trading recommendation.",
            "2026 special sessions are explicitly listed; other years are not supported.",
            "Historical forecast dates are retrospective and are not an independent prospective test.",
        ],
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Calendar date in India: {now.date()} | Cutoff: {cutoff} | Forecast end: {end}")
    print(f"Saved {len(sessions)} trading-session forecasts per stock to {output}")
    if errors:
        print(f"Overall mean absolute percentage error: {math.fsum(errors) / len(errors):.2f}% "
              f"across {len(errors)} stock-days")
    else:
        print("Overall error unavailable: no actual closing prices to compare.")
    print("Warning: this is a trend baseline, not the trained trading model.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", action="store_true", help="Use the latest completed market session as cutoff")
    parser.add_argument("--end-date", type=date.fromisoformat, help="Override next Friday (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "stock_forecasts")
    args = parser.parse_args()
    run(args.end_date, args.output, latest=args.latest)


if __name__ == "__main__":
    main()
