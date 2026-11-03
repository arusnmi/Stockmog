from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
DB_DIR = ROOT_DIR / "database"
DB_PATH = DB_DIR / "trading.db"


def ensure_project_structure() -> None:
    for directory in (DATA_DIR, RAW_DIR, PROCESSED_DIR, FEATURES_DIR, DB_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for child in ("intraday", "daily", "indices"):
        (RAW_DIR / child).mkdir(parents=True, exist_ok=True)

    for child in ("intraday", "daily"):
        (PROCESSED_DIR / child).mkdir(parents=True, exist_ok=True)

    for child in ("intraday", "swing"):
        (FEATURES_DIR / child).mkdir(parents=True, exist_ok=True)
