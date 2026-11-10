"""Rolling-origin validation of the gradient boosting model against the baselines, plus error analysis.

Usage (from the project root):
    python -m src.run_rolling             # recompute everything (slow: tuning + backtest for each fold)
    python -m src.run_rolling --cached    # reuse data/rolling_predictions.csv for the analysis tables
"""
import sys
import time
from pathlib import Path

import pandas as pd

from src.baselines import forecast_day_before, forecast_week_before
from src.config import TEST_END, TZ
from src.data import load_raw, to_half_hourly
from src.evaluate import backtest, compare, mae_by
from src.features import make_dataset
from src.rolling import forecast_fold, make_folds

CACHE = Path("data/rolling_predictions.csv")
MODELS = ["gbm", "gbm_residual", "baseline_day_before", "baseline_week_before", "RTE_J-1"]


def compute_predictions() -> pd.DataFrame:
    """Predictions of every model on every fold's test period (one row per half-hour, with a `fold` column)."""
    data = to_half_hourly(load_raw())
    series = data["consommation"]
    X_all, y_all = make_dataset(series, series.index[0], TEST_END)    # features only read the past, so one pass is enough
    parts = []
    for fold in make_folds():
        start = time.time()
        preds = {}
        preds["gbm"], info = forecast_fold(series, X_all, y_all, fold)
        preds["gbm_residual"], info_res = forecast_fold(series, X_all, y_all, fold, residual=True)
        preds["baseline_day_before"] = backtest(forecast_day_before, series, fold.test_start, fold.test_end)
        preds["baseline_week_before"] = backtest(forecast_week_before, series, fold.test_start, fold.test_end)
        test = series[(series.index >= fold.test_start) & (series.index < fold.test_end)]
        preds["RTE_J-1"] = data["prevision_j1"].reindex(test.index)
        frame = pd.concat([test.rename("truth"), *[p.rename(k) for k, p in preds.items()]], axis=1)
        frame["fold"] = fold.name
        parts.append(frame)
        print(f"{fold.name}: gbm {info} | residual {info_res} | {time.time() - start:.0f} s", flush=True)
    result = pd.concat(parts)
    CACHE.parent.mkdir(exist_ok=True)
    result.to_csv(CACHE)
    return result


def load_cached() -> pd.DataFrame:
    """Read back the predictions saved by a previous full run."""
    return pd.read_csv(CACHE, index_col=0, parse_dates=True).tz_convert("UTC")


def metric_table(frame: pd.DataFrame, metric: str, models: list[str] = MODELS) -> pd.DataFrame:
    """One row per fold, one column per model; same rows for all models inside a fold."""
    rows = {}
    for name, part in frame.groupby("fold"):
        scores = compare(part["truth"], {m: part[m] for m in models})
        rows[name] = scores[metric]
    return pd.DataFrame(rows).T


def print_fold_tables(frame: pd.DataFrame) -> None:
    """Per-fold MAE / RMSE / MAPE, with mean and standard deviation across folds, and the gain over the day-before."""
    for metric in ("MAE (MW)", "RMSE (MW)", "MAPE (%)"):
        table = metric_table(frame, metric)
        summary = pd.concat([table, table.mean().to_frame("mean").T, table.std().to_frame("std").T])
        print(f"\n{metric} by fold:")
        print(summary.round(2).to_string())
    mae_table = metric_table(frame, "MAE (MW)")
    gain = 100 * (1 - mae_table["gbm"] / mae_table["baseline_day_before"])
    print("\nMAE gain of gbm over the day-before baseline (%), by fold:")
    print(gain.round(1).to_string())
    print(f"folds won by gbm over day-before: {(gain > 0).sum()} / {len(gain)}")
    rows = {}
    for name, part in frame.dropna().groupby("fold"):
        rows[name] = {m: (part[m] - part["truth"]).mean() for m in ("gbm", "gbm_residual", "baseline_day_before")}
    print("\nMean signed error (prediction - truth, MW) by fold; negative = under-forecast on average:")
    print(pd.DataFrame(rows).T.round(0).to_string())


def print_error_analysis(frame: pd.DataFrame) -> None:
    """Where do the models fail? By hour, season, weekday, and the worst days (all folds pooled, same rows)."""
    truth = frame["truth"]
    preds = {m: frame[m] for m in ("gbm", "gbm_residual", "baseline_day_before")}
    print("\nMAE (MW) by Paris hour, all folds pooled:")
    print(mae_by(truth, preds, lambda local: local.hour).round(0).to_string())
    seasons = {0: "winter", 1: "spring", 2: "summer", 3: "autumn"}
    print("\nMAE (MW) by season (winter = Dec-Feb):")
    print(mae_by(truth, preds, lambda local: pd.Index((local.month % 12 // 3)).map(seasons)).round(0).to_string())
    print("\nMAE (MW) by weekday, all folds pooled (0 = Monday):")
    print(mae_by(truth, preds, lambda local: local.day_of_week).round(0).to_string())
    print("\nWednesday + Thursday only, MAE (MW) by fold:")
    rows = {}
    for name, part in frame.groupby("fold"):
        local = part.index.tz_convert(TZ)
        sel = part[local.day_of_week.isin([2, 3])].dropna()
        rows[name] = {m: (sel["truth"] - sel[m]).abs().mean() for m in preds}
    print(pd.DataFrame(rows).T.round(0).to_string())
    clean = frame.dropna()
    daily = (clean["truth"] - clean["gbm"]).abs().groupby(clean.index.tz_convert(TZ).date).mean()
    print("\nWorst 10 days for gbm (daily MAE, MW):")
    print(daily.sort_values(ascending=False).head(10).round(0).to_string())


def main() -> None:
    """Run (or reload) the rolling-origin evaluation and print all tables."""
    pd.set_option("display.width", 140)
    frame = load_cached() if "--cached" in sys.argv else compute_predictions()
    print_fold_tables(frame)
    print_error_analysis(frame)


if __name__ == "__main__":
    main()
