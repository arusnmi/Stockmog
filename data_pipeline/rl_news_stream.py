"""Free, timestamped news metadata; never downloads publisher article bodies."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests


ALIASES = {
    "ADANIENT": ["adani enterprises"], "ADANIPORTS": ["adani ports"],
    "APOLLOHOSP": ["apollo hospitals"], "ASIANPAINT": ["asian paints"],
    "AXISBANK": ["axis bank"], "BAJAJ-AUTO": ["bajaj auto"],
    "BAJAJFINSV": ["bajaj finserv"], "BAJFINANCE": ["bajaj finance"],
    "BEL": ["bharat electronics"], "BHARTIARTL": ["bharti airtel"],
    "BPCL": ["bharat petroleum"], "BRITANNIA": ["britannia"],
    "BSE": ["bombay stock exchange", "bse limited"], "CIPLA": ["cipla"],
    "COALINDIA": ["coal india"], "DIVISLAB": ["divi's laboratories", "divis laboratories"],
    "DRREDDY": ["dr reddy", "dr. reddy"], "EICHERMOT": ["eicher motors"],
    "ETERNAL": ["zomato", "eternal limited"], "GRASIM": ["grasim"],
    "HCLTECH": ["hcl technologies", "hcltech"], "HDFC": ["housing development finance"],
    "HDFCBANK": ["hdfc bank"], "HDFCLIFE": ["hdfc life"],
    "HEROMOTOCO": ["hero motocorp"], "HINDALCO": ["hindalco"],
    "HINDUNILVR": ["hindustan unilever"], "ICICIBANK": ["icici bank"],
    "INDIGO": ["interglobe aviation", "indigo airlines"],
    "INDUSINDBK": ["indusind bank"], "INFY": ["infosys"],
    "IOC": ["indian oil"], "ITC": ["itc limited", "itc ltd"],
    "JIOFIN": ["jio financial"], "JSWSTEEL": ["jsw steel"],
    "KOTAKBANK": ["kotak mahindra bank"], "LT": ["larsen", "l&t"],
    "LTM": ["ltimindtree", "lti mindtree"], "M&M": ["mahindra & mahindra", "mahindra and mahindra"],
    "MARUTI": ["maruti suzuki"], "MAXHEALTH": ["max healthcare"],
    "NESTLEIND": ["nestle india", "nestlé india"], "NTPC": ["ntpc"],
    "ONGC": ["oil and natural gas", "ongc"], "POWERGRID": ["power grid corporation"],
    "RELIANCE": ["reliance industries"], "SBILIFE": ["sbi life"],
    "SBIN": ["state bank of india"], "SHREECEM": ["shree cement"],
    "SHRIRAMFIN": ["shriram finance", "shriram transport"],
    "SUNPHARMA": ["sun pharmaceutical", "sun pharma"],
    "TATACONSUM": ["tata consumer"], "TATASTEEL": ["tata steel"],
    "TCS": ["tata consultancy"], "TECHM": ["tech mahindra"],
    "TITAN": ["titan company"], "TMPV": ["tata motors"],
    "TRENT": ["trent limited", "trent ltd"], "ULTRACEMCO": ["ultratech"],
    "UPL": ["upl limited", "united phosphorus"], "WIPRO": ["wipro"],
}
EVENT_COLUMNS = ["symbol", "available_at", "url", "source", "tone"]


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    if not set(EVENT_COLUMNS).issubset(events.columns):
        raise ValueError("News events must contain " + ", ".join(EVENT_COLUMNS))
    result = events[EVENT_COLUMNS].copy()
    if result[["symbol", "available_at", "url", "source"]].isna().any().any():
        raise ValueError("News identifiers and timestamps cannot be missing")
    text_times = result["available_at"].astype(str)
    if not text_times.str.contains(r"(?:Z|[+-]\d\d:?\d\d)$").all():
        raise ValueError("News timestamps must include an explicit timezone")
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True)
    result["tone"] = pd.to_numeric(result["tone"], errors="raise")
    if np.isinf(result["tone"]).any():
        raise ValueError("Infinite news tone")
    return (result.sort_values("available_at", kind="stable")
            .drop_duplicates(["symbol", "url"], keep="first").reset_index(drop=True))


def parse_gkg(content: bytes, release: pd.Timestamp) -> pd.DataFrame:
    """Keep organization/URL matches; GKG tone is not a realized price outcome."""
    records = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if len(archive.infolist()) != 1 or archive.infolist()[0].file_size > 512 * 1024**2:
            raise ValueError("Unexpected or oversized GKG archive")
        with archive.open(archive.namelist()[0]) as stream:
            for raw in stream:
                row = raw.decode("utf-8", errors="replace").rstrip("\n").split("\t")
                if len(row) < 16:
                    raise ValueError("Malformed GKG 2.1 row")
                text = (row[4] + " " + row[13]).casefold()
                symbols = [symbol for symbol, aliases in ALIASES.items()
                           if any(alias in text for alias in aliases)]
                if not symbols:
                    continue
                tone = float(row[15].split(",")[0]) if row[15] else np.nan
                seen = pd.to_datetime(row[1], format="%Y%m%d%H%M%S", utc=True)
                available = max(seen, release) + pd.Timedelta(minutes=15)
                for symbol in symbols:
                    records.append((symbol, available, row[4], "gdelt_gkg", tone))
    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def collect_snapshot(release: pd.Timestamp, output: Path) -> dict:
    stamp = release.strftime("%Y%m%d%H%M%S")
    checkpoint = output / "snapshots" / f"{stamp}.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved["status"] == "ok":
            path = checkpoint.with_suffix(".parquet")
            if hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
                raise ValueError(f"Corrupt news checkpoint: {path}")
            return saved
    url = f"https://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    events = parse_gkg(response.content, release)
    target = checkpoint.with_suffix(".parquet")
    target.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(target, index=False)
    saved = {"status": "ok", "release": release.isoformat(), "url": url,
             "events": len(events), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    save_json(checkpoint, saved)
    return saved


def collect_news(run: Path, output: Path, *, workers: int = 4) -> dict:
    """Sample one historical GKG release per trading day, not the complete archive."""
    if not 1 <= workers <= 8:
        raise ValueError("News workers must be between 1 and 8")
    dates = set()
    for path in run.glob("jobs/*/1d/forecast_features.parquet"):
        times = pd.read_parquet(path, columns=["timestamp"])["timestamp"]
        dates.update(pd.to_datetime(times, utc=True).dt.tz_convert("Asia/Kolkata").dt.date)
    if not dates:
        raise ValueError("No daily stock history found")
    output.mkdir(parents=True, exist_ok=True)
    releases = [pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=9) for day in sorted(dates)]
    successes, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(collect_snapshot, release, output): release for release in releases}
        for index, future in enumerate(as_completed(futures), 1):
            release = futures[future]
            try:
                successes.append(future.result())
            except (requests.RequestException, ValueError, zipfile.BadZipFile) as exc:
                failure = {"release": release.isoformat(), "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print("NEWS FAILURE", json.dumps(failure), flush=True)
            if index % 25 == 0:
                print(f"News snapshots {index}/{len(releases)}; failures {len(failures)}", flush=True)
                save_json(output / "collection-progress.json",
                          {"completed": index, "total": len(releases), "failures": failures})
    # DOC is a recent article-search API, not a way to backfill several years.
    doc_status: dict = {"status": "not_requested"}
    doc_events = pd.DataFrame(columns=EVENT_COLUMNS)
    try:
        response = requests.get("https://api.gdeltproject.org/api/v2/doc/doc",
                                params={"query": '("NSE" OR "BSE") sourcecountry:india',
                                        "mode": "artlist", "format": "json", "maxrecords": 250,
                                        "timespan": "1month", "sort": "datedesc"}, timeout=60)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        records = []
        for article in articles:
            text = (article["title"] + " " + article["url"]).casefold()
            for symbol, aliases in ALIASES.items():
                if any(alias in text for alias in aliases):
                    seen = pd.to_datetime(article["seendate"], utc=True) + pd.Timedelta(minutes=15)
                    records.append((symbol, seen, article["url"], "gdelt_doc", np.nan))
        doc_events = pd.DataFrame(records, columns=EVENT_COLUMNS)
        doc_status = {"status": "sampled", "returned_articles": len(articles),
                      "possibly_truncated": len(articles) == 250,
                      "note": "Recent, capped search; not complete historical coverage."}
    except (requests.RequestException, ValueError, KeyError) as exc:
        doc_status = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        print("DOC NEWS UNAVAILABLE", json.dumps(doc_status), flush=True)
    parts = [pd.read_parquet(output / "snapshots" /
                             (pd.Timestamp(item["release"]).strftime("%Y%m%d%H%M%S") + ".parquet"))
             for item in successes]
    parts.append(doc_events)
    events = normalize_events(pd.concat(parts, ignore_index=True))
    events.to_parquet(output / "events.parquet", index=False)
    summary = {
        "status": "partial", "sampling": "one 15-minute GKG release per trading day at 09:00 UTC",
        "complete_news_history": False, "requested_snapshots": len(releases),
        "successful_snapshots": len(successes), "events": len(events),
        "first_release": releases[0].isoformat(), "last_release": releases[-1].isoformat(),
        "failures": failures, "doc_api": doc_status,
        "limitations": [
            "The other 95 daily GKG releases, weekends and uncollected dates are not covered.",
            "Organization/URL alias matches can miss or misattribute company news.",
            "GKG tone measures news tone, not future market impact; no article bodies are stored.",
            "GKG and DOC are two endpoints of one provider, not independent confirmations.",
            "No fabricated events or inferred zero-news coverage; missing coverage remains explicit.",
        ],
    }
    save_json(output / "coverage.json", summary)
    return summary


def news_features(decisions: pd.Series, events: pd.DataFrame, symbol: str) -> pd.DataFrame:
    events = normalize_events(events)
    selected = events.loc[events["symbol"] == symbol]
    times = pd.DatetimeIndex(selected["available_at"]).as_unit("ns").asi8
    decisions_ns = pd.DatetimeIndex(pd.to_datetime(decisions, utc=True)).as_unit("ns").asi8
    tone = selected["tone"].to_numpy(dtype=float)
    sums = np.concatenate(([0.0], np.cumsum(np.nan_to_num(tone))))
    counts = np.concatenate(([0], np.cumsum(np.isfinite(tone))))
    output = pd.DataFrame(index=decisions.index)
    right = np.searchsorted(times, decisions_ns, side="right")
    for hours in (24, 168):
        left = np.searchsorted(times, decisions_ns - pd.Timedelta(hours=hours).value, side="right")
        n = right - left
        valid = counts[right] - counts[left]
        total = sums[right] - sums[left]
        output[f"news_count_{hours}h"] = np.log1p(n)
        output[f"news_tone_{hours}h"] = np.divide(
            total, valid * 100, out=np.zeros(len(n)), where=valid > 0)
        output[f"news_tone_missing_{hours}h"] = (valid == 0).astype(float)
    output["news_history_incomplete"] = 1.0
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("data/models/nifty-timeframe"))
    parser.add_argument("--output", type=Path, default=Path("data/colab/news"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(collect_news(args.run, args.output, workers=args.workers), indent=2))
