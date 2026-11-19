"""Extend selected daily factor returns with unverified Yahoo observations."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data_pipeline.huggingface_stocks import _safe_symbol
from data_pipeline.nifty_prediction_series import _read_local, _session_day, normalize_provider_rows
from modeling.nifty_preparation import available_candles
from modeling.nifty_public_factor_research import REPOSITORY
from modeling.nifty_universe import local_symbol, parse_nifty_membership


INDEX_TICKERS = {
    "NIFTY_50:Close": "^NSEI",
    "NIFTY_100:Close": "^CNX100",
    "NIFTY_200:Close": "^CNX200",
    "NIFTY_500:Close": "^CRSLDX",
    "NIFTY_BANK:Close": "^NSEBANK",
    "NIFTY_IT:Close": "^CNXIT",
    "NIFTY_MIDCAP_50:Close": "^NSEMDCP50",
    "NIFTY_PHARMA:Close": "^CNXPHARMA",
    "SENSEX:Close": "^BSESN",
}


def _local_ticker(name: str, config: dict) -> str:
    members = parse_nifty_membership(Path(config["membership"]).read_text(encoding="utf-8-sig"))
    candidates = {symbol if symbol == "LTM" else local_symbol(symbol, members)
                  for symbol in config["symbols"]
                  if _safe_symbol(local_symbol(symbol, members)) == name}
    if len(candidates) != 1:
        raise ValueError(f"No unique Yahoo stock ticker mapping for local factor {name}")
    return candidates.pop() + ".NS"


def extend_factor(key, values, store, config, stock_root, reference, provider, now: datetime):
    source, name = key
    update = {"source": source, "name": name, "status": "unsupported",
              "yahoo_ticker": None, "rows_added": 0}
    try:
        if source == "local":
            ticker = _local_ticker(name, config)
            raw = _read_local(stock_root / "day" / _safe_symbol(name) / "1d.csv.gz")
            last_day = _session_day(raw["timestamp"].iloc[-1])
        elif source == "india:" + REPOSITORY and name in INDEX_TICKERS:
            ticker = INDEX_TICKERS[name]
            raw = store.external[key]
            last_day = raw["end"].iloc[-1]
        else:
            raise ValueError(f"No Yahoo mapping for factor {source} {name}")

        update.update(yahoo_ticker=ticker, saved_through=last_day.isoformat())
        start = last_day + timedelta(days=1)
        end = now.date() + timedelta(days=1)
        if start >= end:
            update["status"] = "current"
            return values, update
        rows = provider.fetch_history(ticker, start, end, interval="1d")
        new = normalize_provider_rows(rows, now=now)
        new = new.loc[new["timestamp"].map(_session_day) > last_day].copy()
        if new.empty:
            update["status"] = "no_new_complete_sessions"
            return values, update
        if source == "local":
            merged = pd.concat([raw.tail(1), new], ignore_index=True)
            candles = available_candles(merged, "1d", daily_source=True)
            days = candles["timestamp"].map(_session_day)
            previous = dict(zip(reference[1:], reference))
            consecutive = days.map(previous) == days.shift()
            returns = candles["close"].div(candles["close"].shift()).sub(1).where(consecutive)
            added = pd.DataFrame({"available_at": candles["available_at"],
                                  "factor_return": returns}).iloc[1:].dropna()
        else:
            previous_close = float(raw["close"].iloc[-1])
            closes = pd.concat([pd.Series([previous_close]), new["close"].reset_index(drop=True)],
                               ignore_index=True)
            dates = new["timestamp"].map(_session_day)
            added = pd.DataFrame({
                "available_at": (pd.to_datetime(dates.astype(str)).dt.tz_localize("America/New_York")
                                 + timedelta(days=2)).dt.tz_convert("UTC"),
                "factor_return": closes.div(closes.shift()).sub(1).iloc[1:].to_numpy(),
            })
        combined = pd.concat([values, added], ignore_index=True).sort_values("available_at")
        if combined["available_at"].duplicated().any():
            raise ValueError(f"Duplicate factor availability for {key}")
        update.update(status="extended", rows_added=len(added),
                      yahoo_through=_session_day(new["timestamp"].iloc[-1]).isoformat())
        return combined.reset_index(drop=True), update
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        update.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        print(f"WARNING: Factor {source} {name} was not extended: {exc}", file=sys.stderr, flush=True)
        return values, update
