import numpy as np
import pandas as pd

from src.config import SEED, TZ
from src.gru import SLOTS, make_day_data
from src.run_gbm_profile import PROFILE_COLUMNS, profile_forecaster, profile_matrix


def _series(days: int = 60) -> pd.Series:
    """Half-hourly series where every local day has a distinct level, so a wrong day is easy to spot."""
    idx = pd.date_range("2024-02-01", periods=days * 48, freq="30min", tz="UTC")
    local = idx.tz_convert(TZ)
    day_number = (local.tz_localize(None).normalize() - pd.Timestamp("2024-02-01")).days
    noise = np.random.default_rng(SEED).normal(0, 1, len(idx))
    return pd.Series(40_000 + 100 * day_number + noise, index=idx)


class _RecordingModel:
    """Stand-in for a fitted model: remembers the features it receives and predicts zero."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.features = features
        return np.zeros(len(features))


def test_profile_of_day_d_is_the_previous_local_day():
    y = _series()
    day = pd.Timestamp("2024-03-20", tz=TZ)
    targets = y.index[(y.index >= day) & (y.index < day + pd.Timedelta(days=1))]
    profile = profile_matrix(make_day_data(y), targets)
    previous = y[(y.index >= day - pd.Timedelta(days=1)) & (y.index < day)]
    assert profile.shape == (len(targets), SLOTS)
    np.testing.assert_allclose(profile.iloc[0].to_numpy(), previous.to_numpy(), atol=1e-6)
    assert (profile.nunique() == 1).all()                        # same 48 values for every slot of day D


def test_forecast_time_profile_equals_training_time_profile_and_ignores_the_future():
    y = _series()
    day = pd.Timestamp("2024-03-20", tz=TZ)
    targets = y.index[(y.index >= day) & (y.index < day + pd.Timedelta(days=1))]
    expected = profile_matrix(make_day_data(y), targets)
    garbage = y.copy()
    garbage[garbage.index >= day] = 1e9
    model = _RecordingModel()
    columns = ["lag_1d"] + PROFILE_COLUMNS
    profile_forecaster(model, columns, residual=False)(garbage[garbage.index < day], targets)
    np.testing.assert_allclose(model.features[PROFILE_COLUMNS].to_numpy(), expected.to_numpy(), atol=1e-6)
