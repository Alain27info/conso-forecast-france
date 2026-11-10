import numpy as np
import pandas as pd
import pytest

from src.config import SEED, TZ
from src.gru import fit_scaler, local_day_matrix, make_day_data, positions_between, samples, train
from src.serving import save_artifact


def synthetic_series(days: int = 300) -> pd.Series:
    """Reproducible half-hourly consumption from 2024-02-01 (UTC) with a daily shape and a weekend dip."""
    idx = pd.date_range("2024-02-01", periods=days * 48, freq="30min", tz="UTC")
    local = idx.tz_convert(TZ)
    hour = local.hour + local.minute / 60
    weekend = np.where(local.day_of_week >= 5, -4_000, 0)
    noise = np.random.default_rng(SEED).normal(0, 300, len(idx))
    return pd.Series(50_000 + 8_000 * np.sin(2 * np.pi * hour / 24) + weekend + noise, index=idx)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """A 2-member ensemble trained for one epoch and saved with the real `save_artifact`: (path, series, members, scaler)."""
    y = synthetic_series()
    data = make_day_data(y)
    cutoff = pd.Timestamp("2024-04-15")
    scaler = fit_scaler(local_day_matrix(y), cutoff)
    train_samples = samples(data, scaler, positions_between(data, data.dates[0], cutoff))
    members = []
    for seed in (SEED, SEED + 1):
        model, _ = train(train_samples, epochs=1, seed=seed, residual=True)
        members.append({"seed": seed, "epochs": 1, "validation_mae": 0.0, "model": model})
    path = tmp_path_factory.mktemp("model") / "tiny.pt"
    save_artifact(path, members, scaler, "tiny-test", "2024-04-14", residual=True)
    return path, y, members, scaler
