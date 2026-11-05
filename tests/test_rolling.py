import numpy as np
import pandas as pd

from src.config import SEED, TEST_END, TEST_START, TZ
from src.features import make_dataset
from src.rolling import Fold, make_folds, training_data
from src.train_gbm import fit_final, gbm_forecaster


def _series() -> pd.Series:
    """Reproducible two-year series (2023-2024) with a daily shape."""
    idx = pd.date_range("2023-01-01", "2024-12-31 23:30", freq="30min", tz="UTC")
    hour = idx.tz_convert(TZ).hour + idx.tz_convert(TZ).minute / 60
    noise = np.random.default_rng(SEED).normal(0, 500, len(idx))
    return pd.Series(50_000 + 8_000 * np.sin(2 * np.pi * hour / 24) + noise, index=idx)


def _small_fold() -> Fold:
    start = pd.Timestamp("2024-09-01", tz=TZ)
    return Fold("small", start - pd.DateOffset(years=1), start, pd.Timestamp("2024-09-08", tz=TZ))


def test_folds_are_successive_and_do_not_overlap():
    folds = make_folds()
    assert len(folds) == 5
    for previous, current in zip(folds, folds[1:]):
        assert previous.test_end == current.test_start           # contiguous test years, no gap, no overlap


def test_last_fold_is_the_project_test_period():
    last = make_folds()[-1]
    assert (last.test_start, last.test_end) == (TEST_START, TEST_END)


def test_tuning_period_ends_exactly_where_the_test_period_starts():
    for fold in make_folds():
        assert fold.val_start < fold.test_start < fold.test_end
        assert fold.test_start - fold.val_start >= pd.Timedelta(days=365)


def test_training_data_stops_before_the_test_period():
    series = _series()
    X_all, y_all = make_dataset(series, series.index[0], pd.Timestamp("2025-01-01", tz=TZ))
    fold = _small_fold()
    for residual in (False, True):
        X, y = training_data(X_all, y_all, fold, residual)
        assert y.index.max() < fold.test_start
        assert X.index.equals(y.index)


def test_training_data_does_not_depend_on_the_future():
    """Garbage after the test start must not change a single training row or feature."""
    series = _series()
    fold = _small_fold()
    end = pd.Timestamp("2025-01-01", tz=TZ)
    garbage = series.copy()
    garbage[garbage.index >= fold.test_start] = 1e9
    X_clean, y_clean = training_data(*make_dataset(series, series.index[0], end), fold)
    X_dirty, y_dirty = training_data(*make_dataset(garbage, garbage.index[0], end), fold)
    pd.testing.assert_frame_equal(X_clean, X_dirty)
    pd.testing.assert_series_equal(y_clean, y_dirty)


def test_residual_target_is_y_minus_the_day_before():
    series = _series()
    X_all, y_all = make_dataset(series, series.index[0], pd.Timestamp("2025-01-01", tz=TZ))
    fold = _small_fold()
    X, y_plain = training_data(X_all, y_all, fold)
    X_res, y_res = training_data(X_all, y_all, fold, residual=True)
    expected = (y_plain - X["lag_1d"]).dropna()
    pd.testing.assert_series_equal(y_res, expected)
    assert X_res.index.equals(expected.index)


def test_residual_forecaster_adds_the_day_before_back():
    series = _series()
    fold = _small_fold()
    X_all, y_all = make_dataset(series, series.index[0], pd.Timestamp("2025-01-01", tz=TZ))
    X, y = training_data(X_all, y_all, fold, residual=True)
    columns = list(X.columns)
    model = fit_final(X, y, columns, "squared_error", 10)
    day = pd.Timestamp("2024-11-12", tz=TZ)
    targets = series.index[(series.index >= day) & (series.index < day + pd.Timedelta(days=1))]
    history = series[series.index < day]
    plain = gbm_forecaster(model, columns, residual=False)(history, targets)
    shifted = gbm_forecaster(model, columns, residual=True)(history, targets)
    lag = X_all.loc[targets, "lag_1d"]
    pd.testing.assert_series_equal(shifted - plain, lag, check_names=False)
