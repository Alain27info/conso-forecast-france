"""Train the gradient boosting model, compare it with the baselines, and inspect feature importance.

Usage (from the project root):
    python -m src.run_gbm
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.baselines import forecast_day_before, forecast_week_before
from src.config import SEED, TEST_END, TEST_START
from src.data import load_raw, to_half_hourly
from src.evaluate import backtest, compare, mae_by_weekday
from src.features import train_test_datasets
from src.metrics import mae
from src.train_gbm import VAL_START, fit_final, gbm_forecaster, new_model, tune

MODEL_PATH = Path("models/gbm.joblib")


def lag_columns(columns) -> list[str]:
    """Columns built from past consumption (lags and means); the others are calendar variables."""
    return [c for c in columns if c.startswith(("lag_", "mean_prev"))]


def grouped_permutation_loss(model, X, y, group: list[str]) -> float:
    """MAE increase when a whole group of columns is shuffled together (one shared permutation)."""
    rng = np.random.default_rng(SEED)
    shuffled = X.copy()
    order = rng.permutation(len(X))
    shuffled[group] = X[group].to_numpy()[order]
    return mae(y, model.predict(shuffled)) - mae(y, model.predict(X))


def main() -> None:
    """Run the whole comparison and print the tables."""
    series = to_half_hourly(load_raw())["consommation"]
    X_train, y_train, X_test, y_test = train_test_datasets(series)
    all_cols = list(X_train.columns)
    cal_cols = [c for c in all_cols if c not in lag_columns(all_cols)]
    pd.set_option("display.width", 130)

    print(f"Validation period (tuning): {VAL_START.date()} to {(TEST_START - pd.Timedelta(days=1)).date()}")
    models, predictions = {}, {}
    for name, cols in {"gbm_all_features": all_cols, "gbm_calendar_only": cal_cols}.items():
        loss, n_trees, val_mae = tune(X_train, y_train, cols)
        print(f"{name}: loss={loss}, trees={n_trees}, validation MAE={val_mae:.0f} MW")
        models[name] = fit_final(X_train, y_train, cols, loss, n_trees)
        predictions[name] = backtest(gbm_forecaster(models[name], cols), series)
    joblib.dump({"model": models["gbm_all_features"], "columns": all_cols}, MODEL_PATH)

    # Sanity: the day-by-day backtest (history cut every day) must equal the batch prediction.
    batch = pd.Series(models["gbm_all_features"].predict(X_test[all_cols]), index=X_test.index)
    gap = (predictions["gbm_all_features"].reindex(X_test.index) - batch).abs().max()
    print(f"Max gap between backtest and batch predictions: {gap:.6f} MW")

    predictions["baseline_day_before"] = backtest(forecast_day_before, series)
    predictions["baseline_week_before"] = backtest(forecast_week_before, series)
    test = series[(series.index >= TEST_START) & (series.index < TEST_END)]
    predictions["RTE_J-1 (external reference)"] = to_half_hourly(load_raw())["prevision_j1"].reindex(test.index)

    print("\nScores on the test period (same rows for every model):")
    print(compare(test, predictions).round(2).to_string())
    print("\nMAE (MW) by weekday (0 = Monday):")
    print(mae_by_weekday(test, predictions).round(0).to_string())

    # Importance: computed on the validation period with a model that has not seen it.
    is_val = y_train.index >= VAL_START
    probe = new_model(*tune(X_train, y_train, all_cols)[:2]).fit(X_train.loc[~is_val, all_cols], y_train[~is_val])
    Xv, yv = X_train.loc[is_val, all_cols], y_train[is_val]
    perm = permutation_importance(probe, Xv, yv, scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED)
    imp = pd.Series(perm.importances_mean, index=all_cols).sort_values(ascending=False)
    print("\nPermutation importance on the validation period (MAE increase in MW when the column is shuffled):")
    print(imp.round(0).to_string())
    print("\nGroups shuffled together (MAE increase in MW):")
    print(f"  all lag/mean columns together : {grouped_permutation_loss(probe, Xv, yv, lag_columns(all_cols)):.0f}")
    print(f"  all calendar columns together : {grouped_permutation_loss(probe, Xv, yv, cal_cols):.0f}")


if __name__ == "__main__":
    main()
