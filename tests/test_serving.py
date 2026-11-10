import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.config import TZ
from src.gru import gru_forecaster
from src.serving import InvalidHistory, load_artifact

DAY = dt.date(2024, 5, 10)


def _history_before(y: pd.Series, day: dt.date) -> pd.Series:
    return y[y.index < pd.Timestamp(day).tz_localize(TZ)]


def test_loaded_model_gives_the_same_forecast_as_the_evaluation_code(tiny_model):
    """Training/serving consistency: the API path and the backtest path give identical numbers."""
    path, y, members, scaler = tiny_model
    history = _history_before(y, DAY)
    served = load_artifact(path).predict(history, DAY)
    targets = served.index
    expected = pd.concat([gru_forecaster(m["model"], scaler)(history, targets) for m in members], axis=1).mean(axis=1)
    np.testing.assert_allclose(served.to_numpy(), expected.to_numpy(), rtol=1e-5)
    assert len(served) == 48 and np.isfinite(served).all()


def test_clock_change_days_have_46_and_50_forecast_points(tiny_model):
    path, y, *_ = tiny_model
    predictor = load_artifact(path)
    for day, expected in ((dt.date(2024, 3, 31), 46), (dt.date(2024, 10, 27), 50), (DAY, 48)):
        forecast = predictor.predict(_history_before(y, day), day)
        assert len(forecast) == expected and np.isfinite(forecast).all()


def test_history_containing_the_target_day_is_refused(tiny_model):
    """The information rule is enforced on the input: no data from day D or later."""
    path, y, *_ = tiny_model
    day_start = pd.Timestamp(DAY).tz_localize(TZ)
    with pytest.raises(InvalidHistory, match="day to forecast"):
        load_artifact(path).predict(y[y.index <= day_start], DAY)


def test_history_that_is_too_short_or_too_incomplete_is_refused(tiny_model):
    path, y, *_ = tiny_model
    predictor = load_artifact(path)
    history = _history_before(y, DAY)
    with pytest.raises(InvalidHistory, match="present"):
        predictor.predict(history.iloc[-3 * 48:], DAY)                     # 3 days instead of 14
    holes = history.copy()
    recent = holes.index[-14 * 48:]
    holes.loc[recent[np.random.default_rng(0).random(len(recent)) < 0.2]] = np.nan      # about 20 % missing
    with pytest.raises(InvalidHistory, match="present"):
        predictor.predict(holes, DAY)


def test_a_few_missing_points_are_accepted(tiny_model):
    path, y, *_ = tiny_model
    history = _history_before(y, DAY)
    history.iloc[-100:-97] = np.nan
    assert np.isfinite(load_artifact(path).predict(history, DAY)).all()


def test_naive_unaligned_and_invalid_values_are_refused(tiny_model):
    path, y, *_ = tiny_model
    predictor = load_artifact(path)
    history = _history_before(y, DAY)
    naive = history.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(InvalidHistory, match="timezone"):
        predictor.predict(naive, DAY)
    shifted = history.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=7)
    with pytest.raises(InvalidHistory, match=":00 or :30"):
        predictor.predict(shifted, DAY)
    negative = history.copy()
    negative.iloc[-5] = -10.0
    with pytest.raises(InvalidHistory, match="between 0 and"):
        predictor.predict(negative, DAY)
    with pytest.raises(InvalidHistory, match="empty"):
        predictor.predict(history.iloc[:0], DAY)


def test_missing_model_file_gives_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train_final"):
        load_artifact(tmp_path / "absent.pt")
