"""Rolling-origin validation: the same evaluation repeated on several successive test years.

For each fold the model sees only data BEFORE the fold's test period (expanding window), and
its settings (loss, number of trees) are tuned on the 12 months just before the test period.
Nothing is ever chosen using a fold's test period.
"""
from typing import NamedTuple

import pandas as pd

from src.config import TZ
from src.evaluate import backtest
from src.train_gbm import fit_final, gbm_forecaster, tune

FIRST_TEST_YEAR, LAST_TEST_YEAR = 2021, 2025        # test years start on 1 September of each year


class Fold(NamedTuple):
    name: str
    val_start: pd.Timestamp      # tuning period = [val_start, test_start)
    test_start: pd.Timestamp     # test period = [test_start, test_end)
    test_end: pd.Timestamp


def make_folds(first_year: int = FIRST_TEST_YEAR, last_year: int = LAST_TEST_YEAR) -> list[Fold]:
    """One fold per test year: 1 Sept of `year` to 1 Sept of `year + 1`, tuned on the year before."""
    folds = []
    for year in range(first_year, last_year + 1):
        start = pd.Timestamp(f"{year}-09-01", tz=TZ)
        end = pd.Timestamp(f"{year + 1}-09-01", tz=TZ)
        folds.append(Fold(f"{year}-{str(year + 1)[2:]}", start - pd.DateOffset(years=1), start, end))
    return folds


def training_data(X_all: pd.DataFrame, y_all: pd.Series, fold: Fold, residual: bool = False):
    """Rows strictly before the fold's test period; optionally with the target y - lag_1d."""
    before = y_all.index < fold.test_start
    X, y = X_all[before], y_all[before]
    if residual:
        known = X["lag_1d"].notna()
        X, y = X[known], y[known] - X.loc[known, "lag_1d"]
    return X, y


def forecast_fold(series: pd.Series, X_all: pd.DataFrame, y_all: pd.Series, fold: Fold, residual: bool = False):
    """Tune and fit on the past of `fold`, then backtest on its test period. Returns (predictions, info)."""
    X, y = training_data(X_all, y_all, fold, residual)
    columns = list(X.columns)
    loss, n_trees, val_mae = tune(X, y, columns, fold.val_start)
    model = fit_final(X, y, columns, loss, n_trees)
    predictions = backtest(gbm_forecaster(model, columns, residual), series, fold.test_start, fold.test_end)
    return predictions, {"loss": loss, "trees": n_trees, "validation MAE": round(val_mae)}
