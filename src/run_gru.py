"""GRU on the same five rolling-origin folds, with 3 seeds and two variants, then the comparison tables.

Needs data/rolling_predictions.csv (from `python -m src.run_rolling`) for the other models.

Usage (from the project root):
    python -m src.run_gru             # train and forecast every fold (folds run in parallel)
    python -m src.run_gru --cached    # reuse data/gru_predictions.csv
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from src.config import SEED
from src.data import load_raw, to_half_hourly
from src.evaluate import backtest, compare, mae_by
from src.gru import fit_fold_model, gru_forecaster, make_day_data
from src.rolling import Fold, make_folds
from src.run_rolling import CACHE as ROLLING_CACHE
from src.run_rolling import load_cached

PREDICTIONS = Path("data/gru_predictions.csv")
INFO = Path("data/gru_info.csv")
SEEDS = (SEED, SEED + 1, SEED + 2)
VARIANTS = {"gru": False, "gru_residual": True}          # residual = learn the change from the day before
REFERENCE = ["baseline_day_before", "gbm", "gbm_residual", "RTE_J-1"]


def gru_fold(fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train every (variant, seed) on the past of the fold and forecast its test year."""
    torch.set_num_threads(2)
    series = to_half_hourly(load_raw())["consommation"]
    data = make_day_data(series)
    predictions, rows = {}, []
    for variant, residual in VARIANTS.items():
        for seed in SEEDS:
            start = time.time()
            model, scaler, epochs, val_mae = fit_fold_model(data, fold.val_start, fold.test_start, residual, seed)
            predictions[f"{variant}_s{seed}"] = backtest(gru_forecaster(model, scaler), series, fold.test_start, fold.test_end)
            rows.append({"fold": fold.name, "variant": variant, "seed": seed, "epochs": epochs,
                         "validation_MAE": round(val_mae), "seconds": round(time.time() - start)})
    return pd.DataFrame(predictions), pd.DataFrame(rows)


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every fold in parallel; returns (predictions with a `fold` column, training info)."""
    folds = make_folds()
    with ProcessPoolExecutor(max_workers=len(folds)) as pool:
        results = list(pool.map(gru_fold, folds))
    predictions = pd.concat([p.assign(fold=f.name) for (p, _), f in zip(results, folds)])
    info = pd.concat([i for _, i in results], ignore_index=True)
    predictions.to_csv(PREDICTIONS)
    info.to_csv(INFO, index=False)
    return predictions, info


def seed_columns(frame: pd.DataFrame, variant: str) -> list[str]:
    return [f"{variant}_s{s}" for s in SEEDS]


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Per fold and metric: reference models, then GRU mean / min / max over seeds, and the 3-seed ensemble."""
    rows = {}
    for fold, part in frame.groupby("fold"):
        models = {m: part[m] for m in REFERENCE}
        for variant in VARIANTS:
            cols = seed_columns(part, variant)
            models.update({c: part[c] for c in cols})
            models[f"{variant}_ensemble"] = part[cols].mean(axis=1)
        scores = compare(part["truth"], models)
        row = {}
        for metric in ("MAE (MW)", "RMSE (MW)", "MAPE (%)"):
            for m in REFERENCE:
                row[(metric, m)] = scores.loc[m, metric]
            for variant in VARIANTS:
                per_seed = scores.loc[seed_columns(part, variant), metric]
                row[(metric, f"{variant} mean")] = per_seed.mean()
                row[(metric, f"{variant} min")] = per_seed.min()
                row[(metric, f"{variant} max")] = per_seed.max()
                row[(metric, f"{variant} ensemble")] = scores.loc[f"{variant}_ensemble", metric]
        rows[fold] = row
    return pd.DataFrame(rows).T


def main() -> None:
    """Train (or reload) the GRUs, merge with the reference predictions and print the tables."""
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 30)
    if not ROLLING_CACHE.exists():
        sys.exit("Run `python -m src.run_rolling` first: it saves the other models' predictions.")
    if "--cached" in sys.argv:
        gru = pd.read_csv(PREDICTIONS, index_col=0, parse_dates=True).tz_convert("UTC")
        info = pd.read_csv(INFO)
    else:
        gru, info = compute()
    print("Epochs chosen on the validation year and validation MAE (MW), by fold / variant / seed:")
    print(info.pivot_table(index="fold", columns=["variant", "seed"], values=["epochs", "validation_MAE"]).to_string())
    print(f"training time per (variant, seed): mean {info['seconds'].mean():.0f} s")

    frame = load_cached().drop(columns=["fold"]).join(gru.drop(columns=["fold"]), how="inner")
    frame["fold"] = gru["fold"].reindex(frame.index)
    summary = summarise(frame)
    for metric in ("MAE (MW)", "RMSE (MW)", "MAPE (%)"):
        table = summary[metric]
        table = pd.concat([table, table.mean().to_frame("mean").T, table.std().to_frame("std").T])
        print(f"\n{metric} by fold (same rows for every model, GRU over 3 seeds):")
        print(table.round(2).T.to_string())

    clean = frame.dropna()
    ensembles = {v: clean[seed_columns(clean, v)].mean(axis=1) for v in VARIANTS}
    preds = {"baseline_day_before": clean["baseline_day_before"], "gbm": clean["gbm"],
             "gbm_residual": clean["gbm_residual"], "gru_ensemble": ensembles["gru"],
             "gru_residual_ensemble": ensembles["gru_residual"]}
    for name, group_of in {
        "weekday (0 = Monday)": lambda local: local.day_of_week,
        "Paris hour": lambda local: local.hour,
    }.items():
        print(f"\nMAE (MW) by {name}, all folds pooled:")
        print(mae_by(clean["truth"], preds, group_of).round(0).to_string())
    seasons = {0: "winter", 1: "spring", 2: "summer", 3: "autumn"}
    print("\nMAE (MW) by season, all folds pooled (winter = Dec-Feb):")
    print(mae_by(clean["truth"], preds, lambda local: pd.Index(local.month % 12 // 3).map(seasons)).round(0).to_string())
    bias = {v: {f: round(float((clean.loc[clean['fold'] == f, seed_columns(clean, v)].mean(axis=1)
                                - clean.loc[clean['fold'] == f, 'truth']).mean())) for f in sorted(clean['fold'].unique())}
            for v in VARIANTS}
    print("\nMean signed error of the 3-seed ensemble by fold (MW):", bias)
    print(f"\nrows compared: {len(clean)} of {len(frame)}")


if __name__ == "__main__":
    main()
