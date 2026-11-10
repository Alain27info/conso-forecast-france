"""Build api/static/results.json (AGGREGATED metrics only, no raw data) for the dashboard.

It reads the predictions cached by the earlier scripts, so run those first:
    python -m src.run_rolling && python -m src.run_sarima && python -m src.run_gru && python -m src.run_gbm_profile
Then (from the project root):
    python -m src.make_dashboard_data
All models are scored on exactly the same rows.
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from src.config import TZ
from src.evaluate import compare, mae_by
from src.run_gbm_profile import PREDICTIONS as PROFILE_PREDICTIONS
from src.run_gru import PREDICTIONS as GRU_PREDICTIONS
from src.run_gru import SEEDS
from src.run_rolling import CACHE as ROLLING_PREDICTIONS
from src.run_sarima import PREDICTIONS as SARIMA_PREDICTIONS

OUTPUT = Path("api/static/results.json")

# id -> (label shown on the page, role). The role is only used to style the page.
MODELS = {
    "baseline_day_before": ("Baseline « veille »", "baseline"),
    "baseline_week_before": ("Baseline « semaine précédente »", "baseline"),
    "gbm": ("Gradient boosting", "candidate"),
    "gbm_residual": ("Gradient boosting « écart à la veille »", "candidate"),
    "sarima": ("SARIMA", "candidate"),
    "gru_ensemble": ("GRU (direct)", "candidate"),
    "gru_residual_ensemble": ("GRU « écart à la veille » (modèle final)", "final"),
    "RTE_J-1": ("RTE J-1 (référence externe)", "reference"),
    "gbm_profile": ("Gradient boosting + profil de la veille", "diagnostic"),
    "gbm_profile_residual": ("Gradient boosting « écart » + profil de la veille", "diagnostic"),
}
FINAL_MODEL = "gru_residual_ensemble"
WEEKDAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
SEASONS = ["Hiver (déc.–févr.)", "Printemps", "Été", "Automne"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).tz_convert("UTC")


def assemble() -> pd.DataFrame:
    """One frame with the truth, every model's prediction and the fold, from the cached predictions."""
    frame = _read(ROLLING_PREDICTIONS)
    frame["sarima"] = _read(SARIMA_PREDICTIONS)["sarima"]
    gru = _read(GRU_PREDICTIONS)
    frame["gru_ensemble"] = gru[[f"gru_s{s}" for s in SEEDS]].mean(axis=1)
    frame["gru_residual_ensemble"] = gru[[f"gru_residual_s{s}" for s in SEEDS]].mean(axis=1)
    profile = _read(PROFILE_PREDICTIONS)
    frame["gbm_profile"], frame["gbm_profile_residual"] = profile["gbm_profile"], profile["gbm_profile_residual"]
    return frame


def build_results(frame: pd.DataFrame) -> dict:
    """Aggregate metrics (by fold, mean over folds, by hour / weekday / season), same rows for every model."""
    clean = frame[["truth", "fold", *MODELS]].dropna()
    folds = sorted(clean["fold"].unique())
    by_fold = {metric: {m: [] for m in MODELS} for metric in ("MAE (MW)", "RMSE (MW)", "MAPE (%)")}
    for fold in folds:
        part = clean[clean["fold"] == fold]
        scores = compare(part["truth"], {m: part[m] for m in MODELS})
        for metric, series in by_fold.items():
            for m in MODELS:
                series[m].append(round(float(scores.loc[m, metric]), 2))
    mean = {m: {"mae": round(sum(by_fold["MAE (MW)"][m]) / len(folds), 1),
                "mae_std": round(float(pd.Series(by_fold["MAE (MW)"][m]).std()), 1),
                "rmse": round(sum(by_fold["RMSE (MW)"][m]) / len(folds), 1),
                "mape": round(sum(by_fold["MAPE (%)"][m]) / len(folds), 2)} for m in MODELS}
    preds = {m: clean[m] for m in MODELS}
    groupings = {
        "by_hour": (list(range(24)), lambda local: local.hour),
        "by_weekday": (WEEKDAYS, lambda local: local.day_of_week),
        "by_season": (SEASONS, lambda local: pd.Index(local.month % 12 // 3)),
    }
    grouped = {}
    for name, (labels, key) in groupings.items():
        table = mae_by(clean["truth"], preds, key).sort_index()
        grouped[name] = {"labels": labels, "values": {m: [round(float(v), 1) for v in table[m]] for m in MODELS}}
    return {
        "meta": {"generated_on": dt.date.today().isoformat(), "rows": int(len(clean)), "folds": folds,
                 "period": "1er septembre 2021 – 31 août 2026 (5 années de test)"},
        "models": [{"id": m, "label": label, "role": role} for m, (label, role) in MODELS.items()],
        "by_fold": {"folds": folds, **{metric: values for metric, values in by_fold.items()}},
        "mean": mean,
        "days": ranked_days(clean, FINAL_MODEL),
        **grouped,
    }


def ranked_days(clean: pd.DataFrame, model: str, n_worst: int = 3) -> dict:
    """Days ranked by the model's daily MAE: the typical day (median) and the worst days, from the backtest."""
    local_day = clean.index.tz_convert(TZ).date
    daily = (clean[model] - clean["truth"]).abs().groupby(local_day).mean().sort_values(ascending=False)
    entry = lambda position: {"date": daily.index[position].isoformat(), "mae": round(float(daily.iloc[position]), 1),
                              "rank": position + 1}
    return {"count": int(len(daily)), "typical": entry(len(daily) // 2), "worst": [entry(i) for i in range(n_worst)]}


def main() -> None:
    """Write the dashboard data file."""
    results = build_results(assemble())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    final = results["mean"][FINAL_MODEL]["mae"]
    print(f"wrote {OUTPUT}: {results['meta']['rows']} rows scored per model, final model MAE {final} MW")


if __name__ == "__main__":
    main()
