"""Diagnostic: does the GRU win because of its architecture or because it sees more information?

The GRU reads the whole previous day (all 48 slots, including the last hours before midnight), while the
gradient boosting of the earlier steps only reads lags at the SAME time of day. Here the gradient boosting
receives the same extra information: the 48 values of the previous local day. Same folds, same rules.

Needs data/rolling_predictions.csv and data/gru_predictions.csv (run_rolling and run_gru first).

Usage (from the project root):
    python -m src.run_gbm_profile             # fit and forecast every fold (folds run in parallel)
    python -m src.run_gbm_profile --cached    # reuse data/gbm_profile_predictions.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TEST_END, TZ
from src.data import load_raw, to_half_hourly
from src.evaluate import backtest, compare, mae_by
from src.features import build_features, make_dataset
from src.gru import SLOTS, DayData, make_day_data
from src.rolling import Fold, make_folds, training_data
from src.run_gru import PREDICTIONS as GRU_PREDICTIONS
from src.run_gru import SEEDS
from src.run_rolling import CACHE as ROLLING_CACHE
from src.run_rolling import load_cached
from src.train_gbm import fit_final, tune

PREDICTIONS = Path("data/gbm_profile_predictions.csv")
PROFILE_COLUMNS = [f"prev_day_slot_{k}" for k in range(SLOTS)]


def profile_matrix(data: DayData, index: pd.DatetimeIndex) -> pd.DataFrame:
    """For each timestamp, the 48 values of the previous local day (NaN if there is no previous day)."""
    day = index.tz_convert(TZ).tz_localize(None).normalize()
    position = data.dates.get_indexer(day) - 1
    values = np.where((position >= 0)[:, None], data.filled[np.maximum(position, 0)], np.nan)
    return pd.DataFrame(values, index=index, columns=PROFILE_COLUMNS)


def profile_forecaster(model, columns: list[str], residual: bool):
    """Forecaster using the usual features plus the previous day's profile, read from `history` only."""

    def forecast(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        day = target_index.tz_convert(TZ)[0].tz_localize(None).normalize()
        recent = make_day_data(history.iloc[-5 * 50:])
        previous = pd.DataFrame(recent.filled, index=recent.dates).loc[day - pd.Timedelta(days=1)].to_numpy()
        profile = pd.DataFrame(np.tile(previous, (len(target_index), 1)), index=target_index, columns=PROFILE_COLUMNS)
        features = pd.concat([build_features(history, target_index), profile], axis=1)
        prediction = model.predict(features[columns])
        if residual:
            prediction = prediction + features["lag_1d"].to_numpy()
        return pd.Series(prediction, index=target_index)

    return forecast


def profile_fold(fold: Fold) -> pd.DataFrame:
    """Tune, fit and forecast one fold, for the plain target and for the change from the day before."""
    series = to_half_hourly(load_raw())["consommation"]
    X_all, y_all = make_dataset(series, series.index[0], TEST_END)
    X_all = pd.concat([X_all, profile_matrix(make_day_data(series), X_all.index)], axis=1)
    out = {}
    for name, residual in {"gbm_profile": False, "gbm_profile_residual": True}.items():
        X, y = training_data(X_all, y_all, fold, residual)
        columns = list(X.columns)
        loss, n_trees, _ = tune(X, y, columns, fold.val_start)
        model = fit_final(X, y, columns, loss, n_trees)
        out[name] = backtest(profile_forecaster(model, columns, residual), series, fold.test_start, fold.test_end)
    return pd.DataFrame(out).assign(fold=fold.name)


def main() -> None:
    """Fit (or reload) the diagnostic models and compare them with the GRU and the earlier models."""
    pd.set_option("display.width", 160)
    for needed in (ROLLING_CACHE, GRU_PREDICTIONS):
        if not needed.exists():
            sys.exit(f"{needed} is missing: run run_rolling and run_gru first.")
    if "--cached" in sys.argv:
        diag = pd.read_csv(PREDICTIONS, index_col=0, parse_dates=True).tz_convert("UTC")
    else:
        folds = make_folds()
        with ProcessPoolExecutor(max_workers=len(folds)) as pool:
            diag = pd.concat(list(pool.map(profile_fold, folds)))
        diag.to_csv(PREDICTIONS)
    gru = pd.read_csv(GRU_PREDICTIONS, index_col=0, parse_dates=True).tz_convert("UTC").drop(columns=["fold"])
    frame = load_cached().join(gru, how="inner").join(diag.drop(columns=["fold"]), how="inner")
    for variant in ("gru", "gru_residual"):
        frame[f"{variant}_ensemble"] = frame[[f"{variant}_s{s}" for s in SEEDS]].mean(axis=1)
    models = ["baseline_day_before", "gbm", "gbm_residual", "gbm_profile", "gbm_profile_residual",
              "gru_ensemble", "gru_residual_ensemble", "RTE_J-1"]
    for metric in ("MAE (MW)", "RMSE (MW)"):
        rows = {fold: compare(part["truth"], {m: part[m] for m in models})[metric] for fold, part in frame.groupby("fold")}
        table = pd.DataFrame(rows).T
        print(f"\n{metric} by fold (same rows for every model, GRU = mean prediction of 3 seeds):")
        print(pd.concat([table, table.mean().to_frame("mean").T, table.std().to_frame("std").T]).round(1).T.to_string())
    clean = frame.dropna(subset=models)
    preds = {m: clean[m] for m in ("gbm", "gbm_profile", "gbm_profile_residual", "gru_ensemble", "gru_residual_ensemble")}
    print("\nMAE (MW) by Paris hour, all folds pooled:")
    print(mae_by(clean["truth"], preds, lambda local: local.hour).round(0).to_string())
    print(f"\nrows compared: {len(clean)} of {len(frame)}")


if __name__ == "__main__":
    main()
