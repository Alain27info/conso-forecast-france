import numpy as np
import pandas as pd

from src.config import SEED, TZ
from src.evaluate import backtest
from src.sarima import DAY, SPECS, STEP, SarimaForecaster, calendar_exog, differenced_exog


def _series() -> pd.Series:
    """Reproducible 40-day half-hourly series with a daily shape and a weekend dip."""
    idx = pd.date_range("2024-03-01", periods=40 * DAY, freq="30min", tz="UTC")
    local = idx.tz_convert(TZ)
    hour = local.hour + local.minute / 60
    weekend = np.where(local.day_of_week >= 5, -4_000, 0)
    noise = np.random.default_rng(SEED).normal(0, 300, len(idx))
    return pd.Series(50_000 + 8_000 * np.sin(2 * np.pi * hour / 24) + weekend + noise, index=idx)


class _ZeroForecast:
    """Stand-in for a fitted model whose forecast of the differenced series is always 0."""

    def forecast(self, steps, exog=None):
        return np.zeros(steps)


def _stub_forecaster(history: pd.Series) -> SarimaForecaster:
    forecaster = SarimaForecaster(SPECS["arma11_calendar"])
    forecaster._result, forecaster._last_seen = _ZeroForecast(), history.index[-1]
    return forecaster


def test_differenced_exog_is_x_minus_x_a_day_earlier_and_has_no_gaps():
    idx = pd.date_range("2024-03-20", periods=2 * DAY, freq="30min", tz="UTC")
    expected = calendar_exog(idx).to_numpy() - calendar_exog(idx - DAY * STEP).to_numpy()
    result = differenced_exog(idx)
    np.testing.assert_allclose(result.to_numpy(), expected)
    assert not result.isna().any().any()


def test_zero_difference_forecast_gives_the_day_before():
    """Integration step: if z_hat = 0, y_hat is exactly y a day earlier."""
    y = _series()
    day = pd.Timestamp("2024-04-02", tz=TZ)
    history = y[y.index < day]
    targets = y.index[(y.index >= day) & (y.index < day + pd.Timedelta(days=1))]
    prediction = _stub_forecaster(history)(history, targets)
    expected = history.reindex(targets - DAY * STEP)
    np.testing.assert_allclose(prediction.to_numpy(), expected.to_numpy())


def test_integration_beyond_one_day_reuses_its_own_forecast():
    """On a 25-hour day (50 steps) the last two steps use the forecast of the first two."""
    y = _series()
    history = y[y.index < pd.Timestamp("2024-04-02", tz=TZ)]
    targets = pd.date_range(history.index[-1] + STEP, periods=DAY + 2, freq="30min", tz="UTC")
    prediction = _stub_forecaster(history)(history, targets)
    np.testing.assert_allclose(prediction.iloc[DAY:].to_numpy(), prediction.iloc[:2].to_numpy())


def test_fitted_forecaster_returns_sensible_levels_on_the_requested_index():
    y = _series()
    day = pd.Timestamp("2024-04-02", tz=TZ)
    history = y[y.index < day]
    targets = y.index[(y.index >= day) & (y.index < day + pd.Timedelta(days=1))]
    prediction = SarimaForecaster(SPECS["arma11_no_calendar"], train_days=10)(history, targets)
    assert prediction.index.equals(targets)
    assert np.isfinite(prediction).all()
    assert abs(prediction.mean() / y[targets].mean() - 1) < 0.2


def test_backtest_forecasts_do_not_depend_on_data_from_day_d_onwards():
    """Two series identical before day D, garbage from D on: forecasts of the days before D must match."""
    y = _series()
    day = pd.Timestamp("2024-04-06", tz=TZ)
    garbage = y.copy()
    garbage[garbage.index >= day] = 1e9
    spec = SPECS["arma11_no_calendar"]
    clean_pred = backtest(SarimaForecaster(spec, train_days=10), y, day - pd.Timedelta(days=2), day)
    dirty_pred = backtest(SarimaForecaster(spec, train_days=10), garbage, day - pd.Timedelta(days=2), day)
    pd.testing.assert_series_equal(clean_pred, dirty_pred)
