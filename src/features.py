"""Features for the day-ahead forecast. Every feature is known at D-1 23:30 for a target on day D.

Two families:
  * calendar features: known in advance (we know today that tomorrow is a holiday);
  * lag features: consumption observed on *earlier local days* (at least 1 day before).

Leakage rule: `build_features(history, target_index)` only reads `history` values that
belong to a local day strictly before the target day. No lag shorter than one day exists.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import TEST_END, TEST_START, TZ

LAG_DAYS = (1, 2, 7, 14)
SLOTS_PER_DAY = 48


# ---------------------------------------------------------------- calendar
def easter_sunday(year: int) -> date:
    """Easter Sunday of `year` (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def french_public_holidays(year: int) -> set[date]:
    """Metropolitan France public holidays (without Alsace-Moselle local holidays)."""
    easter = easter_sunday(year)
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    moving = [easter + timedelta(days=n) for n in (1, 39, 50)]   # Easter Monday, Ascension, Whit Monday
    return {date(year, m, d) for m, d in fixed} | set(moving)


def calendar_features(target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar variables in Paris local time (known well before the forecast is made)."""
    local = target_index.tz_convert(TZ)
    day = pd.DatetimeIndex(local.tz_localize(None).normalize())          # local date, naive midnight
    years = range(day.year.min() - 1, day.year.max() + 2)
    holidays = pd.DatetimeIndex([pd.Timestamp(h) for y in years for h in french_public_holidays(y)])
    slot = local.hour * 2 + local.minute // 30
    doy = local.day_of_year
    return pd.DataFrame(
        {
            "slot": slot,                                                # 0..47, half-hour of the local day
            "day_of_week": local.day_of_week,                            # 0 = Monday
            "month": local.month,
            "is_holiday": day.isin(holidays).astype(int),
            "is_day_before_holiday": (day + pd.Timedelta(days=1)).isin(holidays).astype(int),
            "is_day_after_holiday": (day - pd.Timedelta(days=1)).isin(holidays).astype(int),
            "slot_sin": np.sin(2 * np.pi * slot / SLOTS_PER_DAY),        # cyclic encoding: 23:30 is close to 00:00
            "slot_cos": np.cos(2 * np.pi * slot / SLOTS_PER_DAY),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=target_index,
    )


# ------------------------------------------------------------------- lags
def lagged_value(history: pd.Series, target_index: pd.DatetimeIndex, days: int) -> pd.Series:
    """Value observed `days` calendar days earlier at the same Paris wall-clock time.

    Matching wall-clock time (not a fixed 24 h) keeps the daily shape aligned across
    clock changes. Source times that do not exist or are ambiguous become NaN.
    """
    wall_clock = target_index.tz_convert(TZ).tz_localize(None) - pd.Timedelta(days=days)
    source = wall_clock.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    return pd.Series(history.reindex(source).to_numpy(), index=target_index)


def mean_of_previous_days(history: pd.Series, target_index: pd.DatetimeIndex, n_days: int) -> pd.Series:
    """Mean consumption over the `n_days` complete local days ending the day before the target day."""
    local_day = history.index.tz_convert(TZ).tz_localize(None).normalize()
    daily = history.groupby(local_day).mean().asfreq("D")                  # one value per local day
    rolling = daily.rolling(n_days, min_periods=n_days).mean()             # window ends on each day (inclusive)
    target_day = target_index.tz_convert(TZ).tz_localize(None).normalize()
    return pd.Series(rolling.reindex(target_day - pd.Timedelta(days=1)).to_numpy(), index=target_index)


# --------------------------------------------------------------- assembly
def build_features(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """All features for `target_index`, using only history from local days before each target day."""
    feats = calendar_features(target_index)
    for days in LAG_DAYS:
        feats[f"lag_{days}d"] = lagged_value(history, target_index, days)
    feats["mean_prev_day"] = mean_of_previous_days(history, target_index, 1)
    feats["mean_prev_7d"] = mean_of_previous_days(history, target_index, 7)
    return feats


def make_dataset(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """Features X and target y for the slots in [start, end); rows without a measured target are dropped."""
    target = series[(series.index >= start) & (series.index < end)].dropna()
    return build_features(series, target.index), target


def train_test_datasets(series: pd.Series):
    """(X_train, y_train, X_test, y_test), split at the project's TEST_START / TEST_END."""
    X_train, y_train = make_dataset(series, series.index[0], TEST_START)
    X_test, y_test = make_dataset(series, TEST_START, TEST_END)
    return X_train, y_train, X_test, y_test
