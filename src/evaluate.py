"""Generic evaluation: same test period and same metrics for every model.

Rule (decision logged in docs/LOGBOOK.md): the forecast for local day D may use
everything observed up to D-1 23:30 (Paris time), and nothing later.
"""
from typing import Callable

import pandas as pd

from src.config import TEST_END, TEST_START, TZ
from src.metrics import mae, mape, rmse

# A forecaster maps (history before the target day, target timestamps) to predictions.
Forecaster = Callable[[pd.Series, pd.DatetimeIndex], pd.Series]


def backtest(
    forecaster: Forecaster, series: pd.Series, start: pd.Timestamp = TEST_START, end: pd.Timestamp = TEST_END
) -> pd.Series:
    """Forecast each Paris local day in [start, end) using only data before that day."""
    days = pd.date_range(start.tz_localize(None), end.tz_localize(None), freq="D", inclusive="left").tz_localize(TZ)
    parts = []
    for day_start, day_end in zip(days, days[1:].append(pd.DatetimeIndex([end]))):
        lo, hi = series.index.searchsorted([day_start, day_end])
        if hi > lo:
            parts.append(forecaster(series.iloc[:lo], series.index[lo:hi]))   # history is cut at day start
    return pd.concat(parts)


def compare(y_true: pd.Series, predictions: dict[str, pd.Series]) -> pd.DataFrame:
    """Score every model on the same rows (where truth and all predictions exist)."""
    frame = pd.concat([y_true.rename("truth"), *[p.rename(k) for k, p in predictions.items()]], axis=1).dropna()
    rows = {
        name: {
            "MAE (MW)": mae(frame["truth"], frame[name]),
            "RMSE (MW)": rmse(frame["truth"], frame[name]),
            "MAPE (%)": mape(frame["truth"], frame[name]),
            "n": len(frame),
        }
        for name in predictions
    }
    return pd.DataFrame(rows).T


def mae_by_weekday(y_true: pd.Series, predictions: dict[str, pd.Series]) -> pd.DataFrame:
    """MAE per Paris weekday (0 = Monday) for each model, on the same rows."""
    frame = pd.concat([y_true.rename("truth"), *[p.rename(k) for k, p in predictions.items()]], axis=1).dropna()
    weekday = frame.index.tz_convert(TZ).day_of_week
    return pd.DataFrame(
        {name: (frame["truth"] - frame[name]).abs().groupby(weekday).mean() for name in predictions}
    )
