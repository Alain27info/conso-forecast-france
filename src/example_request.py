"""Write a ready-to-send /predict request built from the real data (the 16 last days, to forecast the next day).

Usage (from the project root):
    python -m src.example_request            # writes data/example_request.json
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @data/example_request.json
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from src.config import TZ
from src.data import load_raw, to_half_hourly

OUTPUT = Path("data/example_request.json")
HISTORY_DAYS = 16


def build_request(series: pd.Series, day: dt.date, history_days: int = HISTORY_DAYS) -> dict:
    """The /predict request body for Paris local day `day`: the `history_days` days before it, nothing from `day`."""
    end = pd.Timestamp(day).tz_localize(TZ)
    history = series[(series.index >= end - pd.Timedelta(days=history_days)) & (series.index < end)]
    return {
        "date": day.isoformat(),
        "history": [{"timestamp": t.isoformat(), "consumption_mw": None if pd.isna(v) else round(float(v), 1)}
                    for t, v in history.items()],
    }


def main() -> None:
    """Build the request for the first day after the end of the data."""
    series = to_half_hourly(load_raw())["consommation"]
    last = series.dropna().index[-1].tz_convert(TZ).tz_localize(None)
    payload = build_request(series, (last.normalize() + pd.Timedelta(days=1)).date())
    OUTPUT.write_text(json.dumps(payload))
    print(f"wrote {OUTPUT}: date {payload['date']}, {len(payload['history'])} observations")


if __name__ == "__main__":
    main()
