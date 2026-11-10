"""Train the final model on everything available and save it (weights + scaler) for the API.

The number of epochs of each ensemble member is chosen on the 12 months before the deployment day
(temporal validation), then the member is refitted on all days before the deployment day.

Usage (from the project root):
    python -m src.train_final        # ~4 min, writes models/gru_final.pt (not versioned)
"""
from pathlib import Path

import pandas as pd

from src.config import SEED, TZ
from src.data import load_raw, to_half_hourly
from src.gru import fit_fold_model, make_day_data
from src.serving import save_artifact

MODEL_PATH = Path("models/gru_final.pt")
SEEDS = (SEED, SEED + 1, SEED + 2)


def main() -> None:
    """Fit one residual GRU per seed (validation on the last 12 months) and save the ensemble."""
    series = to_half_hourly(load_raw())["consommation"]
    data = make_day_data(series)
    last = series.dropna().index[-1].tz_convert(TZ).tz_localize(None)
    deploy_day = last.normalize() + pd.Timedelta(days=1)               # first day with no observation
    deploy_start = deploy_day.tz_localize(TZ)
    val_start = (deploy_day - pd.DateOffset(years=1)).tz_localize(TZ)
    members, scaler = [], None
    for seed in SEEDS:
        model, scaler, epochs, val_mae = fit_fold_model(data, val_start, deploy_start, True, seed)
        members.append({"seed": seed, "epochs": epochs, "validation_mae": val_mae, "model": model})
        print(f"seed {seed}: {epochs} epochs, validation MAE {val_mae:.0f} MW", flush=True)
    version = f"gru-residual-{len(SEEDS)}seeds-trained-until-{last.date()}"
    save_artifact(MODEL_PATH, members, scaler, version, str(last.date()), residual=True)
    print(f"saved {MODEL_PATH} ({version})")


if __name__ == "__main__":
    main()
