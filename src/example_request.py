"""Write a ready-to-send /predict request built from the real data (the 16 last days, to forecast the next day).

Usage (from the project root):
    python -m src.example_request            # writes data/example_request.json
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @data/example_request.json
"""
import json
from pathlib import Path

import pandas as pd

from src.config import TZ
from src.data import load_raw, to_half_hourly

OUTPUT = Path("data/example_request.json")
HISTORY_DAYS = 16


def main() -> None:
    """Build the request for the first day after the end of the data."""
    series = to_half_hourly(load_raw())["consommation"]
    last = series.dropna().index[-1].tz_convert(TZ).tz_localize(None)
    day = last.normalize() + pd.Timedelta(days=1)
    start = (day - pd.Timedelta(days=HISTORY_DAYS)).tz_localize(TZ)
    history = series[(series.index >= start) & (series.index < day.tz_localize(TZ))]
    payload = {
        "date": day.date().isoformat(),
        "history": [{"timestamp": t.isoformat(), "consumption_mw": None if pd.isna(v) else round(float(v), 1)}
                    for t, v in history.items()],
    }
    OUTPUT.write_text(json.dumps(payload))
    print(f"wrote {OUTPUT}: date {payload['date']}, {len(payload['history'])} observations")


if __name__ == "__main__":
    main()
