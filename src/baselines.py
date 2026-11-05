"""Naive baselines. A forecaster is a function (history, target_index) -> predictions.

`history` only contains data strictly before the target day (the evaluation code
cuts it), so a forecaster cannot see the future even by mistake.
"""
import pandas as pd

from src.config import TZ


def _same_local_time_lag(history: pd.Series, target_index: pd.DatetimeIndex, days: int) -> pd.Series:
    """Value observed `days` calendar days earlier at the same Paris wall-clock time.

    Matching wall-clock time (not a fixed 24 h) keeps the daily shape aligned across
    clock changes. Source times that do not exist or are ambiguous become NaN.
    """
    wall_clock = target_index.tz_convert(TZ).tz_localize(None) - pd.Timedelta(days=days)
    source = wall_clock.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    return pd.Series(history.reindex(source).to_numpy(), index=target_index)


def forecast_day_before(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Baseline 1: same time the day before."""
    return _same_local_time_lag(history, target_index, days=1)


def forecast_week_before(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Baseline 2: same time, same weekday, one week before."""
    return _same_local_time_lag(history, target_index, days=7)
