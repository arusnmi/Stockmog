"""Export trusted local stock/timeframe history into a portable Colab archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pandas as pd

from data_pipeline.huggingface_stocks import INTRADAY_TIMEFRAMES, _safe_symbol
from data_pipeline.rl_news_stream import normalize_events, save_json
from data_pipeline.rl_research_stream import normalize_research
from modeling.nifty_timeframe_data import read_stock
from modeling.nifty_universe import local_symbol, parse_nifty_membership


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def export_job(run: Path, entry: dict, config: dict, members: tuple,
               destination: Path) -> dict:
    symbol, timeframe = entry["symbol"], entry["timeframe"]
    relative = Path("jobs") / _safe_symbol(symbol) / timeframe
    source, target = run / relative, destination / relative
    meta = json.loads((source / "prepared.json").read_text(encoding="utf-8"))
    if sha256(source / "prepared.json") != entry["prepared_sha256"]:
        raise ValueError(f"Prepared metadata changed: {source}")
    for name in ("examples", "forecast_features"):
        if sha256(source / (name + ".parquet")) != meta[name + "_sha256"]:
            raise ValueError(f"Prepared data changed: {source / name}")
    summary = json.loads((source / "model" / "training-summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary["provenance"]["holdout_start"] != meta["holdout_start"]:
        raise ValueError(f"Unverified predictor cutoff: {source}")
    if summary["feature_names"] != meta["feature_names"]:
        raise ValueError(f"Predictor feature order mismatch: {source}")
    model_names = ["price-lstm.pt", "price-tree.ubj", "training-summary.json"]
    for name, key in (("price-lstm.pt", "lstm_sha256"), ("price-tree.ubj", "tree_sha256")):
        if sha256(source / "model" / name) != summary[key]:
            raise ValueError(f"Corrupt predictor: {source / name}")
    features = pd.read_parquet(source / "forecast_features.parquet")
    candles = read_stock(Path(config["stocks"]), local_symbol(symbol, members), timeframe)
    if candles is None:
        raise ValueError(f"Missing stock candles for {symbol} {timeframe}")
    candles = candles.loc[~candles["is_partial"]]
    frame = features.merge(candles[["timestamp", "open", "high", "low", "volume"]],
                           on="timestamp", validate="one_to_one", how="left")
    if frame[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError(f"Features have no matching complete candle: {source}")
    examples = pd.read_parquet(source / "examples.parquet",
                               columns=["timestamp", "target_timestamp", "label_available_at",
                                        "target_log_return"])
    frame = frame.merge(examples, on="timestamp", validate="one_to_one", how="left")
    contiguous = frame["timestamp"].eq(frame["target_timestamp"].shift())
    if timeframe in INTRADAY_TIMEFRAMES:
        days = frame["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
        contiguous &= days.eq(days.shift())
    starts = np.where(contiguous, 0, np.arange(len(frame)))
    frame["sequence_start"] = np.maximum.accumulate(starts)
    for column in meta["feature_names"]:
        frame[column] = frame[column].astype(np.float32)
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / "history.parquet", compression="zstd", index=False)
    portable_meta = {key: meta[key] for key in
                     ("symbol", "timeframe", "sequence_length", "feature_names", "selection",
                      "validation_start", "holdout_start")}
    portable_meta["predictor_frozen_at"] = (
        pd.Timestamp(meta["holdout_start"], tz="Asia/Kolkata").tz_convert("UTC").isoformat())
    save_json(target / "prepared.json", portable_meta)
    for name in model_names:
        (target / "model").mkdir(exist_ok=True)
        shutil.copyfile(source / "model" / name, target / "model" / name)
    eligible_days = frame.loc[frame["available_at"] >= pd.Timestamp(
        portable_meta["predictor_frozen_at"]), "available_at"].dt.tz_convert("Asia/Kolkata").dt.date.nunique()
    return {"symbol": symbol, "timeframe": timeframe, "folder": relative.as_posix(),
            "rows": len(frame), "first_available": frame["available_at"].min().isoformat(),
            "last_available": frame["available_at"].max().isoformat(),
            "predictor_frozen_at": portable_meta["predictor_frozen_at"],
            "post_freeze_sessions": int(eligible_days), "rl_eligible": eligible_days >= 15,
            "rl_ineligible_reason": (None if eligible_days >= 15
                                     else "Fewer than 15 post-predictor decision sessions")}


def build_archive(run: Path, news: Path, output: Path, *, repo: Path,
                  symbols: list[str] | None = None, research: Path | None = None) -> Path:
    if output.exists():
        raise FileExistsError(f"Use a new archive directory: {output}")
    events = normalize_events(pd.read_parquet(news / "events.parquet"))
    coverage = json.loads((news / "coverage.json").read_text(encoding="utf-8"))
    preparation = json.loads((run / "preparation-summary.json").read_text(encoding="utf-8"))
    if preparation["status"] != "complete":
        raise ValueError("Source preparation is incomplete")
    config = json.loads((run / "identity.json").read_text(encoding="utf-8"))["config"]
    members = parse_nifty_membership(Path(config["membership"]).read_text(encoding="utf-8-sig"))
    entries = [job for job in preparation["jobs"] if symbols is None or job["symbol"] in symbols]
    if not entries or (symbols and set(symbols) != {job["symbol"] for job in entries}):
        raise ValueError("Requested symbols are absent from prepared data")
    output.mkdir(parents=True)
    save_json(output / "export-state.json", {"status": "incomplete", "expected_jobs": len(entries)})
    jobs = []
    for index, entry in enumerate(entries, 1):
        jobs.append(export_job(run, entry, config, members, output))
        if index % 10 == 0:
            print(f"Exported {index}/{len(entries)} jobs", flush=True)
    events.to_parquet(output / "news.parquet", index=False)
    save_json(output / "news-coverage.json", coverage)
    if research is not None:
        copy_research(research, output)
    return package_archive(output, repo=repo, jobs=jobs,
                           skipped=preparation.get("skipped", []), coverage=coverage)


def package_archive(output: Path, *, repo: Path, jobs: list[dict],
                    skipped: list[dict], coverage: dict,
                    archive_path: Path | None = None) -> Path:
    """Package generated history and current source; can refresh this task's archive."""
    if not (output / "export-state.json").is_file():
        raise ValueError("Only an existing generated export can be packaged")
    for directory in ("data_pipeline", "modeling", "backtesting", "features", "strategies", "config"):
        for source in (repo / directory).rglob("*.py"):
            if "__pycache__" not in source.parts:
                target = output / "source" / source.relative_to(repo)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for name in ("requirements.txt", "requirements-training.txt", "requirements-rl.txt"):
        shutil.copyfile(repo / name, output / "source" / name)
    catalog = Path("docs") / "strategy-catalog.csv"
    (output / "source" / catalog.parent).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / catalog, output / "source" / catalog)
    files = {path.relative_to(output).as_posix(): sha256(path)
             for path in sorted(output.rglob("*"))
             if path.is_file() and path.name not in {"export-state.json", "manifest.json"}}
    manifest = {
        "version": 1, "status": "complete", "research_only": True, "jobs": jobs, "files": files,
        "news_complete": coverage.get("complete_news_history", False),
        "news_status": coverage["status"], "source_skipped_jobs": skipped,
        **research_manifest(output),
        "rl_eligible_jobs": sum(job.get("rl_eligible", True) for job in jobs),
        "notes": [
            "Full available prepared history; not a claim of complete exchange or news coverage.",
            "Original predictor holdout is reassigned to chronological RL train/validation/test.",
            "Previously examined report periods are not a pristine prospective test.",
            "Only frozen-predictor-era rows may train PPO; earlier history supplies context only.",
            "Forward target returns are labels only and never included in observations.",
            "Selected factors preserve existing release-lag assumptions, not verified vintage data.",
            "Historical research-universe membership can introduce survivorship bias.",
            "The archive contains trusted model weights and local research data; keep it private.",
        ],
    }
    save_json(output / "manifest.json", manifest)
    save_json(output / "export-state.json", {"status": "complete", "jobs": len(jobs)})
    archive_path = archive_path or output.with_suffix(".tar")
    if archive_path.suffix != ".tar":
        raise ValueError(f"Archive path must end in .tar: {archive_path}")
    with tarfile.open(archive_path.with_suffix(".tar.tmp"), "w") as archive:
        archive.add(output, arcname="stockmog-ppo")
    archive_path.with_suffix(".tar.tmp").replace(archive_path)
    save_json(archive_path.with_suffix(".sha256.json"),
              {"file": archive_path.name, "sha256": sha256(archive_path),
               "bytes": archive_path.stat().st_size, "jobs": len(jobs),
               "news_complete": coverage.get("complete_news_history", False),
               "research_included": (output / "research.parquet").exists()})
    print(f"Archive ready: {archive_path} ({archive_path.stat().st_size / 2**30:.2f} GiB)", flush=True)
    return archive_path


def copy_research(research: Path, output: Path) -> None:
    events = normalize_research(pd.read_parquet(research / "events.parquet"))
    coverage = json.loads((research / "coverage.json").read_text(encoding="utf-8"))
    events.to_parquet(output / "research.parquet", index=False)
    save_json(output / "research-coverage.json", coverage)


def research_manifest(output: Path) -> dict:
    path = output / "research-coverage.json"
    if not path.exists():
        return {}
    coverage = json.loads(path.read_text(encoding="utf-8"))
    return {"research_complete": coverage.get("complete_research_history", False),
            "research_status": coverage["status"], "research_symbols": coverage["symbols"]}


def attach_research(output: Path, research: Path, *, repo: Path,
                    archive_path: Path | None = None) -> Path:
    """Add or refresh the research stream in an existing export without re-exporting jobs."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name in ("news.parquet", *[job["folder"] + "/history.parquet" for job in manifest["jobs"]]):
        if sha256(output / name) != manifest["files"][name]:
            raise ValueError(f"Existing export changed: {name}")
    copy_research(research, output)
    coverage = json.loads((output / "news-coverage.json").read_text(encoding="utf-8"))
    return package_archive(output, repo=repo, jobs=manifest["jobs"],
                           skipped=manifest["source_skipped_jobs"], coverage=coverage,
                           archive_path=archive_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("data/models/nifty-timeframe"))
    parser.add_argument("--news", type=Path, default=Path("data/colab/news"))
    parser.add_argument("--output", type=Path, default=Path("data/colab/ppo-dataset"))
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--research", type=Path, help="Research stream folder (data/colab/research)")
    parser.add_argument("--attach-research", action="store_true",
                        help="Add --research to the existing --output export and repackage")
    parser.add_argument("--archive", type=Path,
                        help="Destination .tar path when repackaging with --attach-research")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.attach_research:
        if args.research is None:
            parser.error("--attach-research needs --research")
        attach_research(args.output, args.research, repo=repo, archive_path=args.archive)
    else:
        if args.archive is not None:
            parser.error("--archive requires --attach-research")
        build_archive(args.run, args.news, args.output, repo=repo, symbols=args.symbols,
                      research=args.research)
