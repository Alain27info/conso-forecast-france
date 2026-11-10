"""SARIMA on the same five rolling-origin folds as the other models, then the common comparison table.

Needs data/rolling_predictions.csv (from `python -m src.run_rolling`) for the other models.

Usage (from the project root):
    python -m src.run_sarima             # fit and forecast every fold (about 10 min, folds run in parallel)
    python -m src.run_sarima --cached    # reuse data/sarima_predictions.csv
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.data import load_raw, to_half_hourly
from src.evaluate import backtest, mae_by
from src.rolling import Fold, make_folds
from src.run_rolling import CACHE as ROLLING_CACHE
from src.run_rolling import load_cached, metric_table
from src.sarima import SPECS, SarimaForecaster, choose_spec

PREDICTIONS = Path("data/sarima_predictions.csv")
SPEC_LOG = Path("data/sarima_specs.csv")
COMPARED = ["baseline_day_before", "gbm", "gbm_residual", "sarima", "RTE_J-1"]


def sarima_fold(fold: Fold) -> tuple[pd.Series, dict]:
    """Choose the specification on the 28 days before the fold, then forecast the fold's test year."""
    start = time.time()
    series = to_half_hourly(load_raw())["consommation"]
    name, scores = choose_spec(series, fold.test_start)
    predictions = backtest(SarimaForecaster(SPECS[name]), series, fold.test_start, fold.test_end)
    info = {"fold": fold.name, "spec": name, **{f"val_mae_{k}": round(v) for k, v in scores.items()},
            "seconds": round(time.time() - start)}
    return predictions.rename("sarima"), info


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every fold in parallel; returns (predictions with a `fold` column, one row of info per fold)."""
    folds = make_folds()
    with ProcessPoolExecutor(max_workers=len(folds)) as pool:
        results = list(pool.map(sarima_fold, folds))
    predictions = pd.concat([p.to_frame().assign(fold=f.name) for (p, _), f in zip(results, folds)])
    info = pd.DataFrame([i for _, i in results])
    predictions.to_csv(PREDICTIONS)
    info.to_csv(SPEC_LOG, index=False)
    return predictions, info


def main() -> None:
    """Fit (or reload) SARIMA, merge with the other models' predictions, and print the comparison."""
    pd.set_option("display.width", 150)
    if not ROLLING_CACHE.exists():
        sys.exit("Run `python -m src.run_rolling` first: it saves the other models' predictions.")
    if "--cached" in sys.argv:
        sarima = pd.read_csv(PREDICTIONS, index_col=0, parse_dates=True).tz_convert("UTC")
        info = pd.read_csv(SPEC_LOG)
    else:
        sarima, info = compute()
    print("Chosen specification per fold (validation MAE in MW, 28 days before the test year):")
    print(info.to_string(index=False))

    frame = load_cached()
    frame["sarima"] = sarima["sarima"].reindex(frame.index)
    for metric in ("MAE (MW)", "RMSE (MW)", "MAPE (%)"):
        table = metric_table(frame, metric, COMPARED)
        summary = pd.concat([table, table.mean().to_frame("mean").T, table.std().to_frame("std").T])
        print(f"\n{metric} by fold (same rows for every model):")
        print(summary.round(2).to_string())

    clean = frame.dropna(subset=COMPARED)
    preds = {m: clean[m] for m in ("baseline_day_before", "gbm", "sarima")}
    bias = {name: (part["sarima"] - part["truth"]).mean() for name, part in clean.groupby("fold")}
    print("\nSARIMA mean signed error by fold (MW):", {k: round(v) for k, v in bias.items()})
    print("\nMAE (MW) by weekday, all folds pooled (0 = Monday):")
    print(mae_by(clean["truth"], preds, lambda local: local.day_of_week).round(0).to_string())
    print("\nMAE (MW) by Paris hour, all folds pooled:")
    print(mae_by(clean["truth"], preds, lambda local: local.hour).round(0).to_string())
    print("\nMAE (MW) by season, all folds pooled (winter = Dec-Feb):")
    seasons = {0: "winter", 1: "spring", 2: "summer", 3: "autumn"}
    print(mae_by(clean["truth"], preds, lambda local: pd.Index(local.month % 12 // 3).map(seasons)).round(0).to_string())
    print(f"\nrows missing for sarima in the test years: {int(frame['sarima'].isna().sum())} of {len(frame)}")


if __name__ == "__main__":
    main()
