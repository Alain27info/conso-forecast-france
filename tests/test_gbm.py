import numpy as np
import pandas as pd

from src.config import SEED, TEST_START, TZ
from src.features import make_dataset
from src.train_gbm import VAL_START, fit_final, gbm_forecaster, new_model, select_iterations


def _series() -> pd.Series:
    """Reproducible 2024 series with a daily shape, so that a tiny model can learn something."""
    idx = pd.date_range("2024-01-01", "2024-12-31 23:30", freq="30min", tz="UTC")
    local_hour = idx.tz_convert(TZ).hour + idx.tz_convert(TZ).minute / 60
    noise = np.random.default_rng(SEED).normal(0, 500, len(idx))
    return pd.Series(50_000 + 8_000 * np.sin(2 * np.pi * local_hour / 24) + noise, index=idx)


def test_validation_period_is_the_year_before_the_test_period():
    assert VAL_START == TEST_START - pd.DateOffset(years=1)
    assert VAL_START < TEST_START


def test_scikit_learn_random_early_stopping_is_off():
    assert new_model("squared_error", 10).early_stopping is False


def test_select_iterations_returns_a_sensible_tree_count():
    y = _series()
    split = pd.Timestamp("2024-09-01", tz=TZ)
    X, target = make_dataset(y, y.index[0], pd.Timestamp("2024-12-31", tz=TZ))
    is_val = target.index >= split
    n_trees, val_mae = select_iterations(X[~is_val], target[~is_val], X[is_val], target[is_val], "squared_error", max_iter=30)
    assert 1 <= n_trees <= 30
    assert val_mae > 0


def test_forecaster_prediction_does_not_depend_on_the_future():
    """Anti-leakage at model level: garbage after the start of day D must not change day D's forecast."""
    y = _series()
    cutoff = pd.Timestamp("2024-09-01", tz=TZ)
    X, target = make_dataset(y, y.index[0], cutoff)
    columns = list(X.columns)
    model = fit_final(X, target, columns, "squared_error", 20)
    forecast = gbm_forecaster(model, columns)

    day = pd.Timestamp("2024-11-12", tz=TZ)
    targets = y.index[(y.index >= day) & (y.index < day + pd.Timedelta(days=1))]
    clean = forecast(y[y.index < day], targets)
    garbage = y.copy()
    garbage[garbage.index >= day] = 1e9
    corrupted = forecast(garbage, targets)

    assert clean.index.equals(targets)
    pd.testing.assert_series_equal(clean, corrupted)
