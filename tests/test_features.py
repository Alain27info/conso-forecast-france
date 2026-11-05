from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import SEED, TZ
from src.features import (
    build_features,
    calendar_features,
    easter_sunday,
    french_public_holidays,
    lagged_value,
)


def _series() -> pd.Series:
    """Random-looking but reproducible 30-minute UTC series covering the whole of 2024."""
    idx = pd.date_range("2024-01-01", "2024-12-31 23:30", freq="30min", tz="UTC")
    values = np.random.default_rng(SEED).normal(50_000, 5_000, len(idx))
    return pd.Series(values, index=idx)


def _day_targets(y: pd.Series, local_day: str) -> pd.DatetimeIndex:
    """Slots of one Paris local day (23 h on spring clock change, 25 h on autumn clock change)."""
    start = pd.Timestamp(local_day, tz=TZ)
    end = (pd.Timestamp(local_day) + pd.Timedelta(days=1)).tz_localize(TZ)    # next local midnight
    return y.index[(y.index >= start) & (y.index < end)]


# ------------------------------------------------------------ calendar
@pytest.mark.parametrize("year, expected", [(2023, date(2023, 4, 9)), (2024, date(2024, 3, 31)),
                                            (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5))])
def test_easter_sunday_known_dates(year, expected):
    assert easter_sunday(year) == expected


def test_french_holidays_2024():
    h = french_public_holidays(2024)
    assert len(h) == 11
    assert {date(2024, 4, 1), date(2024, 5, 9), date(2024, 5, 20)} <= h    # Easter Monday, Ascension, Whit Monday
    assert {date(2024, 1, 1), date(2024, 7, 14), date(2024, 12, 25)} <= h
    assert date(2024, 5, 2) not in h


def test_calendar_flags_around_14_july_2024():
    idx = pd.DatetimeIndex(["2024-07-13 10:00", "2024-07-14 10:00", "2024-07-15 10:00"], tz="UTC")
    cal = calendar_features(idx)
    assert list(cal["is_holiday"]) == [0, 1, 0]
    assert list(cal["is_day_before_holiday"]) == [1, 0, 0]
    assert list(cal["is_day_after_holiday"]) == [0, 0, 1]
    assert cal["day_of_week"].iloc[1] == 6                                   # 14 July 2024 was a Sunday


def test_slot_is_paris_local_time_summer_and_winter():
    idx = pd.DatetimeIndex(["2024-07-01 10:00", "2024-01-01 11:30"], tz="UTC")   # both = 12:00 / 12:30 in Paris
    assert list(calendar_features(idx)["slot"]) == [24, 25]


def test_cyclic_encoding_makes_23h30_close_to_midnight():
    idx = pd.DatetimeIndex(["2024-01-10 22:30", "2024-01-10 23:00"], tz="UTC")   # 23:30 and 00:00 Paris
    c = calendar_features(idx)
    gap = np.hypot(c["slot_sin"].iloc[0] - c["slot_sin"].iloc[1], c["slot_cos"].iloc[0] - c["slot_cos"].iloc[1])
    assert gap < 0.15                                                         # 47 vs 0 as raw numbers: far apart


# ---------------------------------------------------------------- lags
def test_lagged_value_matches_the_source_slot():
    y = _series()
    target = pd.DatetimeIndex(["2024-05-15 10:00"], tz="UTC")
    assert lagged_value(y, target, 7).iloc[0] == y["2024-05-08 10:00"]


def test_mean_of_previous_day_uses_the_complete_previous_local_day():
    y = _series()
    target = pd.DatetimeIndex(["2024-05-15 10:00"], tz="UTC")
    feats = build_features(y, target)
    previous_day = y["2024-05-13 22:00":"2024-05-14 21:30"]                  # local 14 May, 00:00 to 23:30
    assert feats["mean_prev_day"].iloc[0] == pytest.approx(previous_day.mean())


# ------------------------------------------------------ anti-leakage
@pytest.mark.parametrize("local_day", ["2024-02-14", "2024-03-31", "2024-05-01", "2024-10-27", "2024-12-25"])
def test_features_do_not_change_if_the_future_changes(local_day):
    """Anti-leakage: features of day D are identical whether history stops at D's start, or the
    future is garbage, or the future is the real data."""
    y = _series()
    target = _day_targets(y, local_day)
    day_start = pd.Timestamp(local_day, tz=TZ)

    truncated = build_features(y[y.index < day_start], target)
    full = build_features(y, target)
    garbage = y.copy()
    garbage[garbage.index >= day_start] = 1e9
    corrupted = build_features(garbage, target)

    pd.testing.assert_frame_equal(truncated, full)
    pd.testing.assert_frame_equal(truncated, corrupted)


def test_the_leak_check_has_teeth_a_30_minute_lag_would_be_caught():
    """A deliberately leaky feature (value 30 min earlier) must FAIL the check used above."""
    y = _series()
    day_start = pd.Timestamp("2024-05-14", tz=TZ)
    target = _day_targets(y, "2024-05-14")

    def leaky(history: pd.Series) -> pd.Series:
        return history.reindex(target - pd.Timedelta(minutes=30)).set_axis(target)

    truncated = leaky(y[y.index < day_start])           # history ends at midnight: only the first slot is known
    full = leaky(y)
    assert not truncated.equals(full)
