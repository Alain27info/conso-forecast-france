"""SARIMA forecaster (daily seasonality), plugged into the generic evaluation.

Model: z_t = y_t - y_{t-48} (seasonal difference, D = 1, period = 48 half-hours = 24 h) follows
  z_t = beta . (x_t - x_{t-48}) + ARMA(p, q) x seasonal MA(1) at lag 48,
where x are calendar variables (weekday, holidays). The difference is done by hand so that the
integration back to levels is explicit: y_hat_t = z_hat_t + y_{t-48}.

Compute cap: statsmodels' state-space form with period 48 is slow, so the parameters are estimated
once on the last TRAIN_DAYS days before the first forecast, then the Kalman filter is only
*extended* with each new day (no re-estimation). Only data given in `history` is ever read.
"""
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.features import calendar_features
from src.metrics import mae

STEP = pd.Timedelta(minutes=30)
DAY = 48                      # half-hours in 24 h (UTC grid)
TRAIN_DAYS = 56               # window used to estimate the parameters
VAL_DAYS = 28                 # validation window used to choose among the candidate specifications


class Spec(NamedTuple):
    order: tuple[int, int, int]            # (p, 0, q) on the seasonally differenced series
    seasonal_order: tuple[int, int, int, int]
    use_exog: bool


SPECS = {
    "arma11_calendar": Spec((1, 0, 1), (0, 0, 1, DAY), True),
    "arma21_calendar": Spec((2, 0, 1), (0, 0, 1, DAY), True),
    "arma11_no_calendar": Spec((1, 0, 1), (0, 0, 1, DAY), False),
}


def calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Weekday dummies (Tuesday..Sunday) and holiday flags: all known in advance."""
    c = calendar_features(index)
    dow = pd.get_dummies(c["day_of_week"]).reindex(columns=range(1, 7), fill_value=0).astype(float)
    dow.columns = [f"dow{d}" for d in dow.columns]
    flags = c[["is_holiday", "is_day_before_holiday", "is_day_after_holiday"]].astype(float)
    return pd.concat([dow, flags], axis=1)


def differenced_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """x_t - x_{t-24h}: the same difference as the one applied to the target."""
    return calendar_exog(index) - calendar_exog(index - DAY * STEP).to_numpy()


def _regular(obj):
    """Give a regular half-hourly index an explicit frequency (statsmodels needs it to extend a model)."""
    obj = obj.copy()
    obj.index = pd.DatetimeIndex(obj.index, freq=STEP)
    return obj


class SarimaForecaster:
    """Callable (history, target_index) -> predictions. Stateful: remembers the filter between days."""

    def __init__(self, spec: Spec, train_days: int = TRAIN_DAYS):
        self.spec, self.train_days = spec, train_days
        self._result = None
        self._last_seen = None

    def _z_and_x(self, history: pd.Series, index: pd.DatetimeIndex):
        """Seasonally differenced target and differenced calendar variables for `index`."""
        z = (history - history.shift(DAY)).loc[index]
        x = differenced_exog(index) if self.spec.use_exog else None
        return _regular(z), (None if x is None else _regular(x))

    def _fit(self, history: pd.Series) -> None:
        """Estimate the parameters on the last `train_days` days of `history`."""
        index = history.index[-self.train_days * DAY:]
        z, x = self._z_and_x(history, index)
        model = SARIMAX(z, exog=x, order=self.spec.order, seasonal_order=self.spec.seasonal_order, trend="n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = model.fit(disp=False, maxiter=50)
        self._last_seen = history.index[-1]

    def _advance(self, history: pd.Series) -> None:
        """Feed the observations that arrived since the last call to the Kalman filter (no re-estimation)."""
        new_index = history.index[history.index > self._last_seen]
        z, x = self._z_and_x(history, new_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = self._result.extend(z, exog=x)
        self._last_seen = history.index[-1]

    def __call__(self, history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        if self._result is None or history.index[-1] < self._last_seen:
            self._fit(history)
        elif history.index[-1] > self._last_seen:
            self._advance(history)
        assert target_index[0] == self._last_seen + STEP, "targets must start right after the history"
        exog = _regular(differenced_exog(target_index)) if self.spec.use_exog else None
        z_hat = np.asarray(self._result.forecast(len(target_index), exog=exog))
        base = history.reindex(target_index - DAY * STEP).to_numpy(copy=True)   # y_{t-48}: observed for the first 48 steps
        y_hat = np.empty(len(target_index))
        for k in range(len(target_index)):
            if k >= DAY:                                                  # beyond one day: use our own forecast
                base[k] = y_hat[k - DAY]
            y_hat[k] = z_hat[k] + base[k]
        return pd.Series(y_hat, index=target_index)


def choose_spec(series: pd.Series, test_start: pd.Timestamp) -> tuple[str, dict[str, float]]:
    """Pick the specification with the lowest MAE over the VAL_DAYS days before `test_start`."""
    from src.evaluate import backtest        # local import: evaluate does not depend on this module

    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    scores = {}
    for name, spec in SPECS.items():
        pred = backtest(SarimaForecaster(spec), series, val_start, test_start)
        frame = pd.concat([series.rename("truth"), pred.rename("pred")], axis=1, join="inner").dropna()
        scores[name] = mae(frame["truth"], frame["pred"])
    return min(scores, key=scores.get), scores
