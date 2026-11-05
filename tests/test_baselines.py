import numpy as np
import pandas as pd

from src.baselines import forecast_day_before, forecast_week_before
from src.config import TZ
from src.evaluate import backtest, compare


def _series() -> pd.Series:
    """Synthetic 30-minute UTC series whose value is its own position (easy to check)."""
    idx = pd.date_range("2024-03-01", "2024-04-20", freq="30min", tz="UTC")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


def _day(date: str) -> pd.Timestamp:
    return pd.Timestamp(date, tz=TZ)


def test_day_before_returns_value_of_previous_day_same_time():
    y = _series()
    target = pd.DatetimeIndex(["2024-03-10 12:00"], tz="UTC")             # no clock change nearby
    pred = forecast_day_before(y[y.index < "2024-03-10"], target)
    assert pred.iloc[0] == y["2024-03-09 12:00"]


def test_week_before_returns_value_of_previous_week_same_time():
    y = _series()
    target = pd.DatetimeIndex(["2024-03-12 08:30"], tz="UTC")
    pred = forecast_week_before(y[y.index < "2024-03-12"], target)
    assert pred.iloc[0] == y["2024-03-05 08:30"]


def test_day_before_follows_paris_wall_clock_across_spring_clock_change():
    y = _series()
    # 2024-03-31: Paris goes from UTC+1 to UTC+2. Noon in Paris is 10:00 UTC today, 11:00 UTC yesterday.
    target = pd.DatetimeIndex(["2024-03-31 10:00"], tz="UTC")
    pred = forecast_day_before(y[y.index < "2024-03-30 22:00"], target)    # history ends the evening before
    assert pred.iloc[0] == y["2024-03-30 11:00"]


def test_baselines_need_only_data_from_before_the_target_day():
    y = _series()
    target = y.index[(y.index >= "2024-03-20") & (y.index < "2024-03-21")]
    history = y[y.index < "2024-03-20"]                                    # nothing from the target day
    for forecaster in (forecast_day_before, forecast_week_before):
        assert forecaster(history, target).notna().all()


def test_backtest_does_not_use_future_data():
    """Anti-leakage: corrupting everything from day D onward must not change day D's forecast."""
    y = _series()
    day, nxt = _day("2024-03-15"), _day("2024-03-16")
    for forecaster in (forecast_day_before, forecast_week_before):
        clean = backtest(forecaster, y, start=day, end=nxt)
        corrupted = y.copy()
        corrupted[corrupted.index >= day] = 1e9                           # garbage from day D onward
        assert backtest(forecaster, corrupted, start=day, end=nxt).equals(clean)


def test_compare_scores_all_models_on_the_same_rows():
    truth = pd.Series([100.0, 200.0, 300.0])
    a = pd.Series([110.0, 190.0, np.nan])                                  # model A has a gap
    b = pd.Series([100.0, 200.0, 300.0])
    table = compare(truth, {"a": a, "b": b})
    assert table.loc["a", "n"] == table.loc["b", "n"] == 2
    assert table.loc["b", "MAE (MW)"] == 0.0
