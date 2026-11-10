"""Model file and prediction service, shared by the API and the scripts.

The forecast of local day D uses the SAME code as the evaluation (`gru_forecaster`), so the API cannot
drift from what was measured in the backtest. The model file stores the weights AND the scaler together.

Input rule (information rule of the project): the history may only contain data from before day D.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import TZ
from src.gru import HIDDEN, WINDOW_DAYS, DayAheadGRU, Scaler, gru_forecaster

STEP = pd.Timedelta(minutes=30)
MIN_COVERAGE = 0.95          # share of non-missing points required in the 14 days before the target day
MAX_MW = 200_000             # sanity bound on a national consumption value
FORMAT_VERSION = 1


class InvalidHistory(ValueError):
    """The history sent by the caller cannot be used to forecast the requested day."""


def save_artifact(path: Path, members: list[dict], scaler: Scaler, version: str, trained_until: str,
                  residual: bool, hidden: int = HIDDEN) -> None:
    """Write the weights of every ensemble member and the scaler in one file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": FORMAT_VERSION, "version": version, "trained_until": trained_until,
        "residual": residual, "hidden": hidden, "window_days": WINDOW_DAYS,
        "scaler": {"mean": scaler.mean, "std": scaler.std},
        "members": [{"seed": m["seed"], "epochs": m["epochs"], "validation_mae": m["validation_mae"],
                     "state_dict": m["model"].state_dict()} for m in members],
    }, path)


class Predictor:
    """An ensemble of trained GRUs and the scaler they were trained with."""

    def __init__(self, models: list[DayAheadGRU], scaler: Scaler, version: str, trained_until: str):
        self.models, self.scaler, self.version, self.trained_until = models, scaler, version, trained_until

    def predict(self, history: pd.Series, day: dt.date) -> pd.Series:
        """Half-hourly forecast (UTC index) of local day `day`, from a history that ends before that day."""
        day_start = pd.Timestamp(day).tz_localize(TZ)
        next_start = pd.Timestamp(day + dt.timedelta(days=1)).tz_localize(TZ)
        history = check_history(history, day_start)
        targets = pd.date_range(day_start.tz_convert("UTC"), next_start.tz_convert("UTC"), freq=STEP, inclusive="left")
        forecasts = [gru_forecaster(model, self.scaler)(history, targets) for model in self.models]
        return pd.concat(forecasts, axis=1).mean(axis=1)


def load_artifact(path: Path) -> Predictor:
    """Load the model file written by `save_artifact`; weights only, no arbitrary code is executed."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Create it with `python -m src.train_final`.")
    content = torch.load(path, map_location="cpu", weights_only=True)
    if content["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported model file format {content['format_version']}")
    models = []
    for member in content["members"]:
        model = DayAheadGRU(content["residual"], content["hidden"])
        model.load_state_dict(member["state_dict"])
        models.append(model.eval())
    return Predictor(models, Scaler(**content["scaler"]), content["version"], content["trained_until"])


def check_history(history: pd.Series, day_start: pd.Timestamp) -> pd.Series:
    """Validate the history and put it on a regular 30-minute grid ending just before `day_start`."""
    if len(history) == 0:
        raise InvalidHistory("The history is empty.")
    index = history.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise InvalidHistory("Timestamps must include a timezone.")
    history = history.sort_index()
    history.index = history.index.tz_convert("UTC")
    if history.index.has_duplicates:
        raise InvalidHistory("Duplicate timestamps in the history.")
    idx = history.index
    if not ((idx.minute % 30 == 0) & (idx.second == 0) & (idx.microsecond == 0)).all():
        raise InvalidHistory("Timestamps must be on :00 or :30.")
    if (idx >= day_start).any():
        raise InvalidHistory("The history contains data from the day to forecast or later: it must end before it.")
    values = history.to_numpy(dtype=float)
    if (np.isinf(values) | (values <= 0) | (values > MAX_MW)).any():
        raise InvalidHistory(f"Consumption values must be between 0 and {MAX_MW} MW.")
    last = day_start.tz_convert("UTC") - STEP
    grid = pd.date_range(idx.min(), last, freq=STEP)
    history = history.reindex(grid)
    window_start = (day_start.tz_localize(None) - pd.Timedelta(days=WINDOW_DAYS)).tz_localize(TZ)
    coverage = history.reindex(pd.date_range(window_start.tz_convert("UTC"), last, freq=STEP)).notna().mean()
    if coverage < MIN_COVERAGE:
        raise InvalidHistory(
            f"Only {coverage:.0%} of the {WINDOW_DAYS} days before the target day are present "
            f"(at least {MIN_COVERAGE:.0%} required)."
        )
    return history
