"""Timestamped research/report stream for the profitable picks (M&M, MAXHEALTH, ONGC, BSE).

Sources, each kept separate and causally lagged:
* NSE corporate filings (results, analyst meets, presentations, credit ratings, actions).
* NSE quarterly results XBRL (classic and integrated filings): profit/revenue YoY growth.
* GDELT media items whose URL/title look like brokerage research (ratings, target prices).
* Optional user-supplied broker reports CSV (symbol, published_at, rating, target_price).

Only metadata and reported figures are stored; no paywalled report text is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

from data_pipeline.rl_news_stream import ALIASES, normalize_events, save_json


RESEARCH_SYMBOLS = ("M&M", "MAXHEALTH", "ONGC", "BSE")
RESEARCH_COLUMNS = ["symbol", "available_at", "source", "kind", "event_id", "score", "value"]
LAG = pd.Timedelta(minutes=15)
IST = "Asia/Kolkata"
NSE = "https://www.nseindia.com/api/"
FILING_KINDS = [
    ("analyst_meet", r"analyst|investor meet|con\. call"),
    ("presentation", r"presentation|press release|monthly business"),
    ("results", r"financial result|outcome of board"),
    ("credit_rating", r"credit rating"),
    ("corporate_action", r"acquisition|amalgamation|merger|buyback|dividend|bonus|split"),
    ("legal", r"litigation|action\(s\)|penalt|rumour|verification"),
]
POSITIVE = r"(?:^|[^a-z])(upgrade[sd]?|outperform|overweight|accumulate|strong[- ]buy|buy|bullish|raises?[- ]target|positive)(?:[^a-z]|$)"
NEGATIVE = r"(?:^|[^a-z])(downgrade[sd]?|underperform|underweight|reduce|sell|bearish|cuts?[- ]target|negative)(?:[^a-z]|$)"
RESEARCH_HINT = re.compile(
    r"target|upgrade|downgrade|brokerage|rating|outperform|underperform|overweight|underweight"
    r"|accumulate|analyst|stock-?pick|buy-call|sell-call|initiates?|coverage", re.I)
RATING_SCORES = {"strong buy": 1.0, "buy": 1.0, "outperform": 0.75, "overweight": 0.75,
                 "add": 0.5, "accumulate": 0.5, "hold": 0.0, "neutral": 0.0, "equal weight": 0.0,
                 "reduce": -0.5, "underperform": -0.75, "underweight": -0.75, "sell": -1.0}
FEATURES = ["research_filings_7d", "research_filings_30d", "research_analyst_meet_30d",
            "research_credit_score", "research_profit_yoy", "research_revenue_yoy",
            "research_days_since_earnings", "research_media_count_30d", "research_media_score_30d",
            "research_analyst_rating_90d", "research_target_upside", "research_covered"]


def direction(text: str) -> float:
    text = text.casefold()
    up, down = bool(re.search(POSITIVE, text)), bool(re.search(NEGATIVE, text))
    return 1.0 if up and not down else -1.0 if down and not up else np.nan


def normalize_research(events: pd.DataFrame) -> pd.DataFrame:
    if not set(RESEARCH_COLUMNS).issubset(events.columns):
        raise ValueError("Research events must contain " + ", ".join(RESEARCH_COLUMNS))
    result = events[RESEARCH_COLUMNS].copy()
    if result[["symbol", "available_at", "source", "kind", "event_id"]].isna().any().any():
        raise ValueError("Research identifiers and timestamps cannot be missing")
    if not all(getattr(value, "tzinfo", None) for value in result["available_at"]):
        raise ValueError("Research timestamps must be timezone-aware")
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True)
    for column in ("score", "value"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        if np.isinf(result[column]).any():
            raise ValueError(f"Infinite research {column}")
    if (result["score"].abs() > 1).any():
        raise ValueError("Research scores must lie in [-1, 1]")
    return (result.sort_values("available_at", kind="stable")
            .drop_duplicates(["symbol", "source", "event_id"], keep="first").reset_index(drop=True))


class NSEClient:
    """Public website JSON used at a polite rate; responses are cached for reproducibility."""

    def __init__(self, cache: Path, *, pause: float = 1.0):
        self.cache, self.pause = cache, pause
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements"})

    def fetch(self, url: str, params: dict | None = None, *, binary: bool = False,
              use_cache: bool = True):
        key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:32]
        path = self.cache / (key + (".bin" if binary else ".json"))
        if use_cache and path.exists():
            return path.read_bytes() if binary else json.loads(path.read_text(encoding="utf-8"))
        for attempt in range(3):
            time.sleep(self.pause * (attempt + 1))
            response = self.session.get(url, params=params, timeout=60)
            if response.status_code in (401, 403, 429, 500, 502, 503) and attempt < 2:
                continue
            response.raise_for_status()
            break
        payload = response.content if binary else response.json()
        if binary and b"xbrl" not in payload[:4000].lower():
            raise ValueError(f"Unexpected non-XBRL response from {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        if binary:
            temporary.write_bytes(payload)
        else:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(path)
        return payload


def nse_time(text: str | None) -> pd.Timestamp | None:
    if not text or not isinstance(text, str) or text.strip() in ("", "-"):
        return None
    return pd.to_datetime(text.strip(), format="%d-%b-%Y %H:%M:%S").tz_localize(IST)


def filing_events(symbol: str, rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        published = nse_time(row.get("exchdisstime")) or nse_time(row.get("an_dt"))
        if published is None:
            continue
        subject = str(row.get("desc") or "")
        text = subject + " " + str(row.get("attchmntText") or "")
        kind = next((name for name, pattern in FILING_KINDS if re.search(pattern, subject, re.I)), "filing")
        score = direction(text) if kind == "credit_rating" else np.nan
        if kind == "credit_rating" and np.isnan(score) and re.search(r"reaffirm", text, re.I):
            score = 0.0
        event_id = str(row.get("seq_id") or row.get("attchmntFile") or published.isoformat() + subject)
        records.append((symbol, published + LAG, "nse_filing", kind, event_id, score, np.nan))
    return pd.DataFrame(records, columns=RESEARCH_COLUMNS)


def xbrl_figures(content: bytes, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    """Read non-dimensional quarter profit/revenue facts; other periods and segments are ignored."""
    root = ElementTree.fromstring(content)
    local = lambda element: element.tag.rsplit("}", 1)[-1]
    contexts = set()
    for element in root.iter():
        if local(element) != "context":
            continue
        children = {local(child) for child in element.iter()}
        if children & {"segment", "scenario"}:
            continue
        dates = {local(child): (child.text or "").strip() for child in element.iter()
                 if local(child) in ("startDate", "endDate")}
        if dates.get("startDate") == start.strftime("%Y-%m-%d") and dates.get("endDate") == end.strftime("%Y-%m-%d"):
            contexts.add(element.get("id"))
    figures: dict[str, float] = {}
    for element in root.iter():
        name = local(element)
        target = {"ProfitLossForPeriod": "profit", "RevenueFromOperations": "revenue"}.get(name)
        if target and element.get("contextRef") in contexts and target not in figures:
            try:
                figures[target] = float(element.text)
            except (TypeError, ValueError):
                continue
    return figures


def earnings_events(symbol: str, client: NSEClient, *, use_cache: bool = True) -> tuple[pd.DataFrame, list]:
    classic = client.fetch(NSE + "corporates-financial-results",
                           {"index": "equities", "symbol": symbol, "period": "Quarterly",
                            "from_date": "01-01-2020", "to_date": pd.Timestamp.now(IST).strftime("%d-%m-%Y")},
                           use_cache=use_cache)
    integrated = client.fetch(NSE + "integrated-filing-results",
                              {"index": "equities", "symbol": symbol, "period_ended": "all",
                               "type": "Integrated Filing- Financials"}, use_cache=use_cache)
    filings = []
    for row in classic:
        filings.append({"end": pd.Timestamp(row["toDate"]), "time": nse_time(row["exchdisstime"]),
                        "consolidated": row["consolidated"] == "Consolidated", "xbrl": row.get("xbrl")})
    for row in integrated.get("data", []):
        if row.get("type_Sub", "Original") != "Original":
            continue
        filings.append({"end": pd.Timestamp(row["qe_Date"]), "time": nse_time(row["broadcast_Date"]),
                        "consolidated": row["consolidated"] == "Consolidated", "xbrl": row.get("xbrl")})
    warnings = []
    if integrated.get("totalCount", 0) > len(integrated.get("data", [])):
        warnings.append(f"{symbol}: integrated filing list truncated by NSE paging")
    quarters = {}
    for filing in filings:
        if filing["time"] is None or not filing["xbrl"] or not str(filing["xbrl"]).endswith(".xml"):
            continue
        current = quarters.get(filing["end"])
        rank = (not filing["consolidated"], filing["time"])
        if current is None or rank < (not current["consolidated"], current["time"]):
            quarters[filing["end"]] = filing
    figures = {}
    for end, filing in sorted(quarters.items()):
        try:
            content = client.fetch(filing["xbrl"], binary=True)
            values = xbrl_figures(content, end.to_period("Q").start_time, end)
        except (requests.RequestException, ElementTree.ParseError, ValueError) as exc:
            warnings.append(f"{symbol} {end.date()}: {type(exc).__name__}")
            continue
        figures[end] = (filing, values)
    records = []
    for end, (filing, values) in figures.items():
        prior = figures.get(end - pd.DateOffset(years=1) + pd.offsets.QuarterEnd(0))
        growth = {}
        for name in ("profit", "revenue"):
            now, before = values.get(name), prior[1].get(name) if prior else None
            growth[name] = (np.tanh((now - before) / abs(before))
                            if now is not None and before not in (None, 0) else np.nan)
        records.append((symbol, filing["time"] + LAG, "nse_results", "earnings",
                        f"{end.date()}", growth["profit"], growth["revenue"]))
    return pd.DataFrame(records, columns=RESEARCH_COLUMNS), warnings


def media_events(news: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    news = normalize_events(news)
    selected = news.loc[news["symbol"].isin(symbols) & news["url"].str.contains(RESEARCH_HINT)]
    records = []
    for row in selected.itertuples(index=False):
        score = direction(row.url.replace("-", " ").replace("_", " "))
        if np.isnan(score) and np.isfinite(row.tone):
            score = float(np.clip(row.tone / 10, -1, 1))
        records.append((row.symbol, row.available_at, "gdelt_research", "media_research",
                        row.url, score, np.nan))
    return pd.DataFrame(records, columns=RESEARCH_COLUMNS)


def report_events(path: Path, symbols: tuple[str, ...]) -> pd.DataFrame:
    """User broker reports; date-only rows become available at 09:00 IST the next day."""
    reports = pd.read_csv(path, dtype=str).fillna("")
    required = {"symbol", "published_at", "rating", "target_price"}
    if not required.issubset(reports.columns):
        raise ValueError("Reports CSV needs symbol, published_at, rating, target_price")
    records = []
    for row in reports.to_dict("records"):
        if row["symbol"] not in symbols:
            continue
        text = row["published_at"].strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            available = pd.Timestamp(text, tz=IST) + pd.Timedelta(days=1, hours=9)
        else:
            stamp = pd.Timestamp(text)
            if stamp.tzinfo is None:
                raise ValueError(f"Report time needs a timezone or date-only form: {text}")
            available = stamp + LAG
        rating = row["rating"].strip().casefold()
        if rating and rating not in RATING_SCORES:
            raise ValueError(f"Unknown rating {row['rating']!r}; use one of {sorted(RATING_SCORES)}")
        target = float(row["target_price"]) if row["target_price"].strip() else np.nan
        if np.isfinite(target) and target <= 0:
            raise ValueError("Target prices must be positive")
        identity = "|".join([row["symbol"], text, row.get("source", ""), row.get("title", ""), row["target_price"]])
        records.append((row["symbol"], available, "user_report", "analyst_report",
                        hashlib.sha256(identity.encode()).hexdigest()[:24],
                        RATING_SCORES.get(rating, np.nan), target))
    return pd.DataFrame(records, columns=RESEARCH_COLUMNS)


def collect_research(news: Path, output: Path, *, start: str = "2022-01-01", end: str | None = None,
                     symbols: tuple[str, ...] = RESEARCH_SYMBOLS, reports: Path | None = None,
                     client: NSEClient | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    client = client or NSEClient(output / "raw")
    end_day = pd.Timestamp(end or pd.Timestamp.now(IST).date())
    today = pd.Timestamp.now(IST).normalize().tz_localize(None)
    parts, failures, warnings, counts = [], [], [], {}
    for symbol in symbols:
        chunk = pd.Timestamp(start)
        while chunk <= end_day:
            stop = min(chunk + pd.Timedelta(days=89), end_day)
            try:
                rows = client.fetch(NSE + "corporate-announcements",
                                    {"index": "equities", "symbol": symbol,
                                     "from_date": chunk.strftime("%d-%m-%Y"), "to_date": stop.strftime("%d-%m-%Y")},
                                    use_cache=stop < today - pd.Timedelta(days=2))
                parts.append(filing_events(symbol, rows))
            except (requests.RequestException, ValueError) as exc:
                failures.append({"symbol": symbol, "source": "nse_filing", "from": str(chunk.date()),
                                 "error": f"{type(exc).__name__}: {exc}"})
            chunk = stop + pd.Timedelta(days=1)
        try:
            earnings, notes = earnings_events(symbol, client, use_cache=False)
            parts.append(earnings)
            warnings += notes
        except (requests.RequestException, ValueError, KeyError) as exc:
            failures.append({"symbol": symbol, "source": "nse_results", "error": f"{type(exc).__name__}: {exc}"})
        print(f"Research collected for {symbol}; failures so far {len(failures)}", flush=True)
    parts.append(media_events(pd.read_parquet(news / "events.parquet"), symbols))
    if reports is not None:
        parts.append(report_events(reports, symbols))
    events = normalize_research(pd.concat([part for part in parts if len(part)], ignore_index=True))
    events.to_parquet(output / "events.parquet", index=False)
    for (symbol, source), group in events.groupby(["symbol", "source"]):
        counts.setdefault(symbol, {})[source] = len(group)
    summary = {
        "status": "partial", "complete_research_history": False, "symbols": list(symbols),
        "start": start, "end": str(end_day.date()), "events": len(events), "counts": counts,
        "user_reports": str(reports) if reports else None, "failures": failures, "warnings": warnings,
        "lags": {"nse": "exchange dissemination time + 15 minutes", "gdelt": "GKG release + 15 minutes",
                 "user_reports": "timestamp + 15 minutes, or next day 09:00 IST for date-only rows"},
        "limitations": [
            "Paywalled broker research is not collected; only user-supplied reports add ratings/targets.",
            "NSE filing subjects are categorized by keyword; filing text is boilerplate, not sentiment.",
            "Earnings use reported profit/revenue YoY at filing time, not analyst consensus surprise.",
            "GDELT research matching uses URL keywords and can include unrelated or market-wide items.",
            "Only M&M, MAXHEALTH, ONGC and BSE are covered; other stocks receive missing research.",
        ],
    }
    save_json(output / "coverage.json", summary)
    return summary


def research_features(decisions: pd.Series, closes: pd.Series, events: pd.DataFrame,
                      symbol: str) -> pd.DataFrame:
    """Each row only sees events whose available_at is at or before its decision time."""
    output = pd.DataFrame(np.nan, index=decisions.index, columns=FEATURES)
    events = normalize_research(events)
    selected = events.loc[events["symbol"] == symbol]
    covered = symbol in set(events["symbol"])
    output["research_covered"] = float(covered)
    if not covered:
        return output
    decision_ns = pd.DatetimeIndex(pd.to_datetime(decisions, utc=True)).as_unit("ns").asi8
    day = pd.Timedelta(days=1).value

    def window(mask: pd.Series):
        part = selected.loc[mask]
        times = pd.DatetimeIndex(part["available_at"]).as_unit("ns").asi8
        return part, times, np.searchsorted(times, decision_ns, side="right")

    filings, times, right = window(selected["source"] == "nse_filing")
    for days, name in ((7, "research_filings_7d"), (30, "research_filings_30d")):
        output[name] = np.log1p(right - np.searchsorted(times, decision_ns - days * day, side="right"))
    meets, times, right = window((selected["source"] == "nse_filing") & (selected["kind"] == "analyst_meet"))
    output["research_analyst_meet_30d"] = np.log1p(right - np.searchsorted(times, decision_ns - 30 * day, side="right"))

    def latest(mask: pd.Series, column: str, max_days: int) -> tuple[np.ndarray, np.ndarray]:
        part = selected.loc[mask & selected[column].notna()]
        times = pd.DatetimeIndex(part["available_at"]).as_unit("ns").asi8
        index = np.searchsorted(times, decision_ns, side="right") - 1
        values = np.full(len(decision_ns), np.nan)
        age = np.full(len(decision_ns), np.nan)
        found = index >= 0
        age[found] = (decision_ns[found] - times[index[found]]) / day
        keep = found & (age <= max_days)
        values[keep] = part[column].to_numpy(dtype=float)[index[keep]]
        return values, np.where(keep, age, np.nan)

    output["research_credit_score"], _ = latest(selected["kind"] == "credit_rating", "score", 365)
    output["research_profit_yoy"], age = latest(selected["kind"] == "earnings", "score", 200)
    output["research_revenue_yoy"], _ = latest(selected["kind"] == "earnings", "value", 200)
    earnings_times = pd.DatetimeIndex(selected.loc[selected["kind"] == "earnings", "available_at"]).as_unit("ns").asi8
    last = np.searchsorted(earnings_times, decision_ns, side="right") - 1
    output["research_days_since_earnings"] = np.where(
        last >= 0, np.log1p((decision_ns - earnings_times[np.maximum(last, 0)]) / day), np.nan)

    def rolling_mean(mask: pd.Series, days: int) -> tuple[np.ndarray, np.ndarray]:
        part = selected.loc[mask]
        times = pd.DatetimeIndex(part["available_at"]).as_unit("ns").asi8
        score = part["score"].to_numpy(dtype=float)
        sums = np.concatenate(([0.0], np.cumsum(np.nan_to_num(score))))
        valid = np.concatenate(([0], np.cumsum(np.isfinite(score))))
        right = np.searchsorted(times, decision_ns, side="right")
        left = np.searchsorted(times, decision_ns - days * day, side="right")
        n = valid[right] - valid[left]
        mean = np.divide(sums[right] - sums[left], n, out=np.full(len(n), np.nan), where=n > 0)
        return mean, right - left

    media = selected["source"] == "gdelt_research"
    output["research_media_score_30d"], count = rolling_mean(media, 30)
    output["research_media_count_30d"] = np.log1p(count)
    output["research_analyst_rating_90d"], _ = rolling_mean(selected["kind"] == "analyst_report", 90)
    target, _ = latest(selected["kind"] == "analyst_report", "value", 180)
    close = closes.to_numpy(dtype=float)
    output["research_target_upside"] = np.clip(target / close - 1, -1, 1)
    return output


class ResearchStream:
    """Incremental poller: append new filings, results, research headlines and user reports."""

    def __init__(self, folder: Path, *, symbols: tuple[str, ...] = RESEARCH_SYMBOLS,
                 reports: Path | None = None, client: NSEClient | None = None):
        self.folder, self.symbols, self.reports = folder, symbols, reports
        self.client = client or NSEClient(folder / "raw")
        self.state_path = folder / "stream-state.json"

    def _doc_headlines(self, since: pd.Timestamp) -> pd.DataFrame:
        records = []
        hours = max(1, min(24 * 90, int((pd.Timestamp.now("UTC") - since) / pd.Timedelta(hours=1)) + 1))
        for symbol in self.symbols:
            names = " OR ".join(f'"{alias}"' for alias in ALIASES[symbol])
            for attempt in range(3):
                response = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", timeout=60, params={
                    "query": f'({names}) ("target price" OR upgrade OR downgrade OR brokerage OR rating)',
                    "mode": "artlist", "format": "json", "maxrecords": 75, "timespan": f"{hours}h"})
                if response.status_code != 429 or attempt == 2:
                    break
                time.sleep(30 * (attempt + 1))
            response.raise_for_status()
            for article in response.json().get("articles", []):
                title = article.get("title", "")
                seen = pd.to_datetime(article["seendate"], utc=True) + LAG
                target = re.search(r"target(?: price)?(?: of| to| at)?\s*(?:rs\.?|₹|inr)?\s*([\d,]{2,}(?:\.\d+)?)",
                                   title, re.I)
                records.append((symbol, seen, "gdelt_research", "media_research", article["url"],
                                direction(title), np.nan))
                if target:
                    records.append((symbol, seen, "gdelt_research", "analyst_report", article["url"] + "#target",
                                    direction(title), float(target.group(1).replace(",", ""))))
            time.sleep(5)
        return pd.DataFrame(records, columns=RESEARCH_COLUMNS)

    def poll(self) -> dict:
        state = (json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists()
                 else {"last_poll": None})
        since = pd.Timestamp(state["last_poll"]) if state["last_poll"] else pd.Timestamp.now("UTC") - pd.Timedelta(days=7)
        now = pd.Timestamp.now("UTC")
        parts, errors = [], []
        for symbol in self.symbols:
            try:
                rows = self.client.fetch(NSE + "corporate-announcements", {
                    "index": "equities", "symbol": symbol,
                    "from_date": since.tz_convert(IST).strftime("%d-%m-%Y"),
                    "to_date": now.tz_convert(IST).strftime("%d-%m-%Y")}, use_cache=False)
                parts.append(filing_events(symbol, rows))
                earnings, _ = earnings_events(symbol, self.client, use_cache=False)
                parts.append(earnings)
            except (requests.RequestException, ValueError, KeyError) as exc:
                errors.append(f"{symbol} NSE: {type(exc).__name__}: {exc}")
        try:
            parts.append(self._doc_headlines(since))
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"GDELT DOC: {type(exc).__name__}: {exc}")
        if self.reports is not None and self.reports.exists():
            parts.append(report_events(self.reports, self.symbols))
        store = self.folder / "events.parquet"
        existing = pd.read_parquet(store) if store.exists() else pd.DataFrame(columns=RESEARCH_COLUMNS)
        fresh = [part for part in parts if len(part)]
        combined = normalize_research(pd.concat([existing, *fresh], ignore_index=True)) if fresh else existing
        added = len(combined) - len(existing)
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = store.with_suffix(".tmp.parquet")
        combined.to_parquet(temporary, index=False)
        temporary.replace(store)
        state = {"last_poll": (now if not errors else since).isoformat(), "added": added, "errors": errors}
        save_json(self.state_path, state)
        return state

    def latest(self, symbol: str, close: float, at: pd.Timestamp | None = None) -> dict:
        at = at or pd.Timestamp.now("UTC")
        events = pd.read_parquet(self.folder / "events.parquet")
        row = research_features(pd.Series([at]), pd.Series([close]), events, symbol).iloc[0]
        return {name: (None if pd.isna(value) else float(value)) for name, value in row.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--news", type=Path, default=Path("data/colab/news"))
    parser.add_argument("--output", type=Path, default=Path("data/colab/research"))
    parser.add_argument("--reports", type=Path, help="Optional broker reports CSV")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--watch", action="store_true", help="Keep polling for new research")
    parser.add_argument("--interval", type=int, default=900, help="Seconds between polls (min 300)")
    args = parser.parse_args()
    if args.watch:
        stream = ResearchStream(args.output, reports=args.reports)
        while True:
            print(json.dumps({"time": pd.Timestamp.now("UTC").isoformat(), **stream.poll()}), flush=True)
            time.sleep(max(300, args.interval))
    print(json.dumps(collect_research(args.news, args.output, start=args.start, reports=args.reports), indent=2))
