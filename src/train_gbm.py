"""Gradient boosting model (scikit-learn HistGradientBoostingRegressor) for the day-ahead forecast.

Tuning rule: anything that is chosen (loss, number of trees) is chosen on a TEMPORAL validation
period that precedes the test period. The test period is never used for tuning.
Scikit-learn's own early stopping is switched off: it samples its validation set at random,
which on a time series puts immediate neighbours in both sets (leakage).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import SEED, TEST_START
from src.features import build_features
from src.metrics import mae

VAL_START = TEST_START - pd.DateOffset(years=1)      # validation = the 12 months before the test period
LOSSES = ("squared_error", "absolute_error")


def new_model(loss: str, max_iter: int) -> HistGradientBoostingRegressor:
    """A boosting model with fixed, moderate settings; only `loss` and `max_iter` get tuned."""
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.1, max_iter=max_iter, max_leaf_nodes=31, min_samples_leaf=100,
        early_stopping=False, random_state=SEED,
    )


def select_iterations(X_tr, y_tr, X_val, y_val, loss: str, max_iter: int = 600) -> tuple[int, float]:
    """Fit on the training part and return (best number of trees, validation MAE at that point)."""
    model = new_model(loss, max_iter).fit(X_tr, y_tr)
    curve = [mae(y_val, pred) for pred in model.staged_predict(X_val)]    # MAE after 1, 2, ... trees
    best = int(np.argmin(curve))
    return best + 1, float(curve[best])


def tune(X_train, y_train, columns: list[str], val_start: pd.Timestamp = VAL_START) -> tuple[str, int, float]:
    """Choose (loss, number of trees) on the temporal validation split (rows from `val_start` on)."""
    is_val = y_train.index >= val_start
    X_tr, y_tr = X_train.loc[~is_val, columns], y_train[~is_val]
    X_val, y_val = X_train.loc[is_val, columns], y_train[is_val]
    results = [(loss, *select_iterations(X_tr, y_tr, X_val, y_val, loss)) for loss in LOSSES]
    return min(results, key=lambda r: r[2])


def fit_final(X_train, y_train, columns: list[str], loss: str, n_trees: int) -> HistGradientBoostingRegressor:
    """Refit on the whole training period with the tuned settings."""
    return new_model(loss, n_trees).fit(X_train[columns], y_train)


def gbm_forecaster(model: HistGradientBoostingRegressor, columns: list[str], residual: bool = False):
    """Wrap a fitted model as a forecaster (history, target_index) -> predictions, as in src/evaluate.py.

    If `residual` is True the model predicts y - lag_1d, so the day-before value is added back.
    """

    def forecast(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        features = build_features(history, target_index)
        prediction = model.predict(features[columns])
        if residual:
            prediction = prediction + features["lag_1d"].to_numpy()
        return pd.Series(prediction, index=target_index)

    return forecast
