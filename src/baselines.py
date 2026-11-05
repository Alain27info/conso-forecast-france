"""Naive baselines. A forecaster is a function (history, target_index) -> predictions.

`history` only contains data strictly before the target day (the evaluation code
cuts it), so a forecaster cannot see the future even by mistake.
"""
import pandas as pd

from src.features import lagged_value


def forecast_day_before(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Baseline 1: same time the day before."""
    return lagged_value(history, target_index, days=1)


def forecast_week_before(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Baseline 2: same time, same weekday, one week before."""
    return lagged_value(history, target_index, days=7)
