"""GRU day-ahead forecaster (PyTorch), plugged into the generic evaluation.

Representation: one row per Paris local day with 48 columns = the local half-hour slots. The model
reads the last WINDOW_DAYS days (one GRU step per day: 48 consumption values + calendar of that day)
and outputs the 48 values of the target day at once (direct strategy, no error accumulation).

Leakage rules:
  * the normalisation (mean, std) is fitted on training days only, then applied to everything;
  * the input window of day D only contains days D-WINDOW_DAYS .. D-1;
  * early stopping uses a validation period that comes AFTER the training days (never a random split).
"""
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import TZ
from src.features import calendar_features

WINDOW_DAYS = 14
SLOTS = 48
N_DAY_FEATURES = 12            # weekday one-hot (7) + 3 holiday flags + 2 day-of-year terms
MAX_EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
HIDDEN = 64


# ------------------------------------------------------------------ data
def local_day_matrix(series: pd.Series) -> pd.DataFrame:
    """Rows = Paris local dates, columns = local half-hour slot 0..47 (23 h days leave NaN, 25 h days are averaged)."""
    local = series.index.tz_convert(TZ)
    day = local.tz_localize(None).normalize()
    slot = local.hour * 2 + local.minute // 30
    matrix = series.groupby([day, slot]).mean().unstack().reindex(columns=range(SLOTS))
    return matrix.reindex(pd.date_range(matrix.index.min(), matrix.index.max(), freq="D"))


def day_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """Calendar vector of each local date (known in advance): weekday one-hot, holiday flags, day of year."""
    noon = (dates + pd.Timedelta(hours=12)).tz_localize(TZ)
    c = calendar_features(noon)
    onehot = np.eye(7)[c["day_of_week"].to_numpy()]
    others = c[["is_holiday", "is_day_before_holiday", "is_day_after_holiday", "doy_sin", "doy_cos"]].to_numpy()
    return np.column_stack([onehot, others]).astype(np.float32)


class Scaler(NamedTuple):
    mean: float
    std: float


def fit_scaler(matrix: pd.DataFrame, before: pd.Timestamp) -> Scaler:
    """Mean and std of the consumption of the days strictly before `before` (a naive local date)."""
    values = matrix[matrix.index < before].to_numpy()
    return Scaler(float(np.nanmean(values)), float(np.nanstd(values)))


class DayData(NamedTuple):
    dates: pd.DatetimeIndex
    raw: np.ndarray        # (n_days, 48), NaN where a slot is missing
    filled: np.ndarray     # same, NaN filled inside the day by interpolation (inputs only)
    feats: np.ndarray      # (n_days, N_DAY_FEATURES)


def make_day_data(series: pd.Series) -> DayData:
    """Day matrix, its filled version and the calendar features, for every local day of `series`."""
    matrix = local_day_matrix(series)
    filled = matrix.interpolate(axis=1, limit_direction="both")   # uses only slots of the same (past) day
    return DayData(matrix.index, matrix.to_numpy(), filled.to_numpy(), day_features(matrix.index))


def samples(data: DayData, scaler: Scaler, positions: np.ndarray):
    """(inputs, target-day features, normalised targets) for the target days at `positions`."""
    norm = lambda a: (a - scaler.mean) / scaler.std
    seq = np.stack([
        np.concatenate([np.nan_to_num(norm(data.filled[p - WINDOW_DAYS:p])), data.feats[p - WINDOW_DAYS:p]], axis=1)
        for p in positions
    ]).astype(np.float32)
    return (torch.from_numpy(seq), torch.from_numpy(data.feats[positions]),
            torch.from_numpy(norm(data.raw[positions]).astype(np.float32)))


# ----------------------------------------------------------------- model
class DayAheadGRU(nn.Module):
    """GRU over the past days; the hidden state and the target-day calendar feed a small head giving 48 values."""

    def __init__(self, residual: bool, hidden: int = HIDDEN):
        super().__init__()
        self.residual = residual
        self.gru = nn.GRU(SLOTS + N_DAY_FEATURES, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + N_DAY_FEATURES, 128), nn.ReLU(), nn.Linear(128, SLOTS))

    def forward(self, seq: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(seq)
        out = self.head(torch.cat([hidden[-1], target_feats], dim=1))
        return out + seq[:, -1, :SLOTS] if self.residual else out      # residual: learn the change from the day before


def masked_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error ignoring the NaN targets."""
    mask = ~torch.isnan(target)
    errors = (prediction - torch.nan_to_num(target)).abs() * mask
    return errors.sum() / mask.sum().clamp(min=1)


def train(train_samples, epochs: int, seed: int, residual: bool, val_samples=None, std: float = 1.0):
    """Train with Adam; returns (model, validation MAE in MW after each epoch, empty if no validation)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = DayAheadGRU(residual)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    seq, feats, y = train_samples
    curve = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(seq), generator=generator)
        for i in range(0, len(order), BATCH_SIZE):
            batch = order[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            masked_l1(model(seq[batch], feats[batch]), y[batch]).backward()
            optimizer.step()
        if val_samples is not None:
            model.eval()
            with torch.no_grad():
                curve.append(float(masked_l1(model(val_samples[0], val_samples[1]), val_samples[2])) * std)
    return model, curve


def positions_between(data: DayData, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Positions of the target days with start <= date < end (naive local dates) that have a full input window."""
    return np.flatnonzero((data.dates >= start) & (data.dates < end) & (np.arange(len(data.dates)) >= WINDOW_DAYS))


def fit_fold_model(data: DayData, val_start: pd.Timestamp, test_start: pd.Timestamp, residual: bool, seed: int):
    """Choose the number of epochs on the validation days, then refit on all days before the test period.

    Returns (model, scaler, best number of epochs, validation MAE in MW).
    """
    val_day, test_day = val_start.tz_localize(None), test_start.tz_localize(None)
    scaler = fit_scaler(pd.DataFrame(data.raw, index=data.dates), val_day)       # training days only
    train_pos = positions_between(data, data.dates[0], val_day)
    val_pos = positions_between(data, val_day, test_day)
    _, curve = train(samples(data, scaler, train_pos), MAX_EPOCHS, seed, residual,
                     samples(data, scaler, val_pos), scaler.std)
    best_epochs = int(np.argmin(curve)) + 1
    scaler = fit_scaler(pd.DataFrame(data.raw, index=data.dates), test_day)      # all days before the test
    all_pos = positions_between(data, data.dates[0], test_day)
    model, _ = train(samples(data, scaler, all_pos), best_epochs, seed, residual)
    return model, scaler, best_epochs, float(min(curve))


# ------------------------------------------------------------- forecaster
def gru_forecaster(model: DayAheadGRU, scaler: Scaler):
    """Wrap a trained model as a forecaster (history, target_index) -> predictions, as in src/evaluate.py."""

    def forecast(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        local = target_index.tz_convert(TZ)
        day = local[0].tz_localize(None).normalize()
        recent = make_day_data(history.iloc[-(WINDOW_DAYS + 3) * 50:])            # only the last days of the history
        window = pd.date_range(day - pd.Timedelta(days=WINDOW_DAYS), day - pd.Timedelta(days=1), freq="D")
        rows = pd.DataFrame(recent.filled, index=recent.dates).reindex(window).to_numpy()
        seq = np.concatenate([np.nan_to_num((rows - scaler.mean) / scaler.std), day_features(window)], axis=1)
        seq = seq[None].astype(np.float32)
        target_feats = day_features(pd.DatetimeIndex([day]))
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(seq), torch.from_numpy(target_feats))[0].numpy()
        slot = local.hour * 2 + local.minute // 30
        return pd.Series(out[slot] * scaler.std + scaler.mean, index=target_index)

    return forecast
