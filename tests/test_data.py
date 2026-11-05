import numpy as np
import pandas as pd

from src.data import to_half_hourly, to_paris_time


def _raw(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a raw-like frame from (UTC timestamp, consommation) pairs."""
    df = pd.DataFrame(rows, columns=["date_heure", "consommation"])
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df["prevision_j1"] = 0.0
    df["prevision_j"] = 0.0
    df["nature"] = "Données définitives"
    return df


def test_quarter_hour_rows_without_consumption_are_dropped():
    raw = _raw([("2024-01-01 00:00", 100.0), ("2024-01-01 00:15", np.nan), ("2024-01-01 00:30", 200.0)])
    out = to_half_hourly(raw)
    assert list(out.index.minute) == [0, 30]


def test_duplicated_timestamps_are_averaged():
    raw = _raw([("2024-03-31 01:00", 100.0), ("2024-03-31 01:00", 120.0), ("2024-03-31 01:30", 200.0)])
    out = to_half_hourly(raw)
    assert out["consommation"].iloc[0] == 110.0


def test_missing_slot_stays_nan_and_is_not_interpolated():
    raw = _raw([("2024-10-27 00:00", 100.0), ("2024-10-27 01:00", 300.0)])
    out = to_half_hourly(raw)
    assert len(out) == 3                                  # 00:00, 00:30, 01:00 on a regular grid
    assert np.isnan(out["consommation"].iloc[1])          # no value invented from the future


def test_paris_time_is_utc_plus_two_in_summer_and_plus_one_in_winter():
    idx = pd.DatetimeIndex(["2024-07-01 10:00", "2024-01-01 10:00"], tz="UTC")
    local = to_paris_time(pd.DataFrame({"x": [1, 2]}, index=idx))
    assert list(local.index.hour) == [12, 11]
