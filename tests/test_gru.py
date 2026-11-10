import numpy as np
import pandas as pd
import torch

from src.config import SEED, TZ
from src.evaluate import backtest
from src.gru import (
    SLOTS, WINDOW_DAYS, DayAheadGRU, fit_scaler, gru_forecaster, local_day_matrix, make_day_data, masked_l1,
    positions_between, samples, train,
)


def _series(days: int = 80) -> pd.Series:
    """Reproducible half-hourly series (from 2024-02-01) with a daily shape, a weekend dip and noise."""
    idx = pd.date_range("2024-02-01", periods=days * 48, freq="30min", tz="UTC")
    local = idx.tz_convert(TZ)
    hour = local.hour + local.minute / 60
    weekend = np.where(local.day_of_week >= 5, -4_000, 0)
    noise = np.random.default_rng(SEED).normal(0, 300, len(idx))
    return pd.Series(50_000 + 8_000 * np.sin(2 * np.pi * hour / 24) + weekend + noise, index=idx)


def test_day_matrix_has_48_slots_and_handles_the_clock_changes():
    idx = pd.date_range("2024-03-29", "2024-04-02", freq="30min", tz="UTC", inclusive="left")
    series = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    matrix = local_day_matrix(series)
    assert matrix.shape[1] == SLOTS
    spring_day = pd.Timestamp("2024-03-31")                       # 23-hour day: 02:00-03:00 does not exist
    assert matrix.loc[spring_day, [4, 5]].isna().all()
    assert matrix.loc[spring_day].notna().sum() == 46
    idx = pd.date_range("2024-10-26", "2024-10-30", freq="30min", tz="UTC", inclusive="left")
    series = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    autumn_day = pd.Timestamp("2024-10-27")                       # 25-hour day: 02:00-03:00 happens twice
    local = series.index.tz_convert(TZ)
    twice = series[(local.date == autumn_day.date()) & (local.hour == 2) & (local.minute == 0)]
    assert len(twice) == 2
    assert local_day_matrix(series).loc[autumn_day, 4] == twice.mean()


def test_scaler_uses_only_days_before_the_cutoff():
    y = _series()
    matrix = local_day_matrix(y)
    cutoff = pd.Timestamp("2024-03-15")
    corrupted = matrix.copy()
    corrupted[corrupted.index >= cutoff] = 1e9
    assert fit_scaler(matrix, cutoff) == fit_scaler(corrupted, cutoff)


def test_input_window_only_contains_the_previous_days():
    """Inputs of day D are identical if every day from D onwards is replaced by garbage."""
    y = _series()
    day = pd.Timestamp("2024-03-20")
    garbage = y.copy()
    garbage[garbage.index.tz_convert(TZ).tz_localize(None) >= day] = 1e9
    clean, dirty = make_day_data(y), make_day_data(garbage)
    scaler = fit_scaler(local_day_matrix(y), pd.Timestamp("2024-03-10"))
    position = int(np.flatnonzero(clean.dates == day)[0])
    clean_seq, clean_feats, _ = samples(clean, scaler, np.array([position]))
    dirty_seq, dirty_feats, _ = samples(dirty, scaler, np.array([position]))
    assert clean_seq.shape == (1, WINDOW_DAYS, SLOTS + 12)
    assert torch.equal(clean_seq, dirty_seq)
    assert torch.equal(clean_feats, dirty_feats)


def test_training_positions_stop_before_the_cutoff():
    data = make_day_data(_series())
    cutoff = pd.Timestamp("2024-03-15")
    positions = positions_between(data, data.dates[0], cutoff)
    assert data.dates[positions].max() < cutoff
    assert positions.min() >= WINDOW_DAYS


def test_masked_loss_ignores_missing_targets():
    prediction = torch.tensor([[1.0, 5.0]])
    target = torch.tensor([[2.0, float("nan")]])
    assert float(masked_l1(prediction, target)) == 1.0


def _tiny_training(seed: int, residual: bool = False):
    y = _series()
    data = make_day_data(y)
    scaler = fit_scaler(local_day_matrix(y), pd.Timestamp("2024-04-01"))
    train_samples = samples(data, scaler, positions_between(data, data.dates[0], pd.Timestamp("2024-04-01")))
    model, _ = train(train_samples, epochs=2, seed=seed, residual=residual)
    return y, model, scaler


def test_same_seed_gives_identical_predictions_and_different_seeds_do_not():
    day = pd.Timestamp("2024-04-10", tz=TZ)
    outputs = []
    for seed in (SEED, SEED, SEED + 1):
        y, model, scaler = _tiny_training(seed)
        targets = y.index[(y.index >= day) & (y.index < day + pd.Timedelta(days=1))]
        outputs.append(gru_forecaster(model, scaler)(y[y.index < day], targets))
    pd.testing.assert_series_equal(outputs[0], outputs[1])
    assert not np.allclose(outputs[0].to_numpy(), outputs[2].to_numpy())


def test_forecaster_matches_the_batch_path_and_ignores_the_future():
    y, model, scaler = _tiny_training(SEED)
    day = pd.Timestamp("2024-04-10", tz=TZ)
    targets = y.index[(y.index >= day) & (y.index < day + pd.Timedelta(days=1))]
    clean = gru_forecaster(model, scaler)(y[y.index < day], targets)
    garbage = y.copy()
    garbage[garbage.index >= day] = 1e9
    dirty = gru_forecaster(model, scaler)(garbage[garbage.index < day], targets)
    pd.testing.assert_series_equal(clean, dirty)
    # same numbers as the batch path used for training
    data = make_day_data(y)
    position = int(np.flatnonzero(data.dates == pd.Timestamp("2024-04-10"))[0])
    seq, feats, _ = samples(data, scaler, np.array([position]))
    model.eval()
    with torch.no_grad():
        batch = model(seq, feats)[0].numpy() * scaler.std + scaler.mean
    local = targets.tz_convert(TZ)
    np.testing.assert_allclose(clean.to_numpy(), batch[local.hour * 2 + local.minute // 30], rtol=1e-4)


def test_forecaster_works_in_the_generic_backtest_on_a_25_hour_day():
    y = pd.Series(np.random.default_rng(SEED).normal(50_000, 500, 120 * 48),
                  index=pd.date_range("2024-07-01", periods=120 * 48, freq="30min", tz="UTC"))
    data = make_day_data(y)
    scaler = fit_scaler(local_day_matrix(y), pd.Timestamp("2024-10-20"))
    model, _ = train(samples(data, scaler, positions_between(data, data.dates[0], pd.Timestamp("2024-10-20"))),
                     epochs=1, seed=SEED, residual=True)
    start = pd.Timestamp("2024-10-26", tz=TZ)
    predictions = backtest(gru_forecaster(model, scaler), y, start, pd.Timestamp("2024-10-29", tz=TZ))
    assert len(predictions) == 48 + 50 + 48                      # Oct 27 has 25 hours = 50 half-hours
    assert np.isfinite(predictions).all()
    assert isinstance(DayAheadGRU(False), torch.nn.Module)
