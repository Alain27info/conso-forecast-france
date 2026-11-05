"""Download and load the RTE eCO2mix national consumption dataset (open data).

Source : https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/
Licence: Licence Ouverte v2.0 (Etalab), attribution to RTE required.

Usage (from the project root):
    python -m src.data
"""
from pathlib import Path

import httpx
import pandas as pd

DATASET_ID = "eco2mix-national-cons-def"
EXPORT_URL = (
    f"https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{DATASET_ID}/exports/csv"
)
# Only the columns we need: timestamp, target, RTE's own forecasts (benchmark only), data status.
COLUMNS = ["date_heure", "consommation", "prevision_j1", "prevision_j", "nature"]
RAW_PATH = Path("data/eco2mix_national_cons_def.csv")


def download_raw(path: Path = RAW_PATH) -> Path:
    """Download the dataset as a CSV file (semicolon-separated) into `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    params = {"delimiter": ";", "select": ",".join(COLUMNS), "order_by": "date_heure"}
    with httpx.stream("GET", EXPORT_URL, params=params, timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return path


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV (timestamps parsed as UTC, as published by RTE)."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    return df


def to_half_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Regular 30-minute UTC grid; duplicated timestamps are averaged, gaps stay NaN.

    The raw file has one row per 15 minutes, but `consommation` is only filled on
    :00 and :30. Duplicates (4 per spring clock change) are averaged because their
    values are sometimes different. Missing slots (1 hour each autumn) are NOT filled,
    since interpolating would use a future value.
    """
    df = raw.dropna(subset=["consommation"]).set_index("date_heure").sort_index()
    df = df.groupby(level=0).agg(
        {"consommation": "mean", "prevision_j1": "mean", "prevision_j": "mean", "nature": "first"}
    )
    grid = pd.date_range(df.index.min(), df.index.max(), freq="30min", tz="UTC", name="date_heure")
    return df.reindex(grid)


def to_paris_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a UTC-indexed frame to Europe/Paris local time (for calendar features)."""
    return df.tz_convert("Europe/Paris")


if __name__ == "__main__":
    out = download_raw()
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
