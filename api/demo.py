"""Routes used only by the demo web page: they read the LOCAL dataset (data/, never versioned).

/predict itself stays stateless. These routes only (1) build a realistic /predict request for a given day
and (2) return the curves needed to compare the forecast: actual consumption, RTE's J-1 forecast, and the
"same time yesterday" baseline. Without the local dataset they answer 503 and the page says so.
"""
import datetime as dt

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.baselines import forecast_day_before
from src.config import TZ
from src.data import load_raw, to_half_hourly
from src.example_request import HISTORY_DAYS, build_request
from src.serving import day_targets

NO_DATA = "Local dataset not found. Download it with `python -m src.data` to enable the demo page."


class DemoData:
    """The local half-hourly dataset, loaded on first use (or injected, for tests)."""

    def __init__(self, frame: pd.DataFrame | None = None):
        self._frame = frame

    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            try:
                self._frame = to_half_hourly(load_raw())
            except FileNotFoundError as error:
                raise HTTPException(status_code=503, detail=NO_DATA) from error
        return self._frame


def _points(values: pd.Series) -> list[dict]:
    return [{"timestamp": t.isoformat(), "consumption_mw": None if pd.isna(v) else round(float(v), 1)}
            for t, v in values.items()]


def _local_date(timestamp: pd.Timestamp) -> dt.date:
    return timestamp.tz_convert(TZ).date()


def _last_forecastable_day(observed: pd.Series) -> dt.date:
    """The day after the last COMPLETE local day of observations (the last day itself if it is only partly observed)."""
    last = observed.index[-1].tz_convert(TZ)
    complete = (last.hour, last.minute) == (23, 30)
    return last.date() + dt.timedelta(days=1 if complete else 0)


def build_router() -> APIRouter:
    router = APIRouter(prefix="/demo", tags=["demo"])

    @router.get("/range")
    def date_range(request: Request) -> dict:
        """Days the demo page can forecast: from 16 days after the first observation to the day after the last one."""
        series = request.app.state.demo.frame()["consommation"].dropna()
        return {
            "first_date": (_local_date(series.index[0]) + dt.timedelta(days=HISTORY_DAYS)).isoformat(),
            "last_date": _last_forecastable_day(series).isoformat(),
            "last_observation_date": _local_date(series.index[-1]).isoformat(),
            "trained_until": request.app.state.predictor.trained_until,
        }

    @router.get("/day")
    def day(date: dt.date, request: Request) -> dict:
        """A /predict request for `date` plus the curves to compare with the forecast (null where unknown)."""
        frame = request.app.state.demo.frame()
        series = frame["consommation"]
        observed = series.dropna()
        first = _local_date(observed.index[0]) + dt.timedelta(days=HISTORY_DAYS)
        last = _last_forecastable_day(observed)
        if not first <= date <= last:
            raise HTTPException(status_code=422, detail=f"date must be between {first} and {last}")
        targets = day_targets(date)
        history = series[series.index < targets[0]]
        trained_until = dt.date.fromisoformat(request.app.state.predictor.trained_until)
        return {
            "date": date.isoformat(),
            "in_sample": date <= trained_until,
            "request": build_request(series, date),
            "actual": _points(series.reindex(targets)),
            "rte_j1": _points(frame["prevision_j1"].reindex(targets)),
            "day_before": _points(forecast_day_before(history, targets)),
        }

    return router
