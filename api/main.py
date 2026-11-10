"""FastAPI service: GET /health and POST /predict.

The model is loaded ONCE at startup (lifespan), not at every request. All the forecasting logic lives in
src/serving.py, which uses the same code as the evaluation: this file only handles HTTP and validation.

Run:  uvicorn api.main:app        (model file: $MODEL_PATH, default models/gru_final.pt)
"""
import datetime as dt
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AwareDatetime, BaseModel, Field

from api.demo import DemoData, build_router
from src.serving import MAX_MW, InvalidHistory, load_artifact

logger = logging.getLogger("api")
STATIC_DIR = Path(__file__).parent / "static"
MAX_OBSERVATIONS = 5000        # about 100 days of half-hourly data: protects the service from huge payloads

Megawatts = Annotated[float, Field(gt=0, le=MAX_MW)]


class Observation(BaseModel):
    timestamp: AwareDatetime                  # must carry a timezone (an offset or Z)
    consumption_mw: Megawatts | None = None   # null = missing value


class PredictRequest(BaseModel):
    date: dt.date = Field(description="Paris local day to forecast")
    history: list[Observation] = Field(
        max_length=MAX_OBSERVATIONS,
        description="Half-hourly national consumption, ending before the day to forecast (at least 14 days)",
    )


class Point(BaseModel):
    timestamp: AwareDatetime
    consumption_mw: float


class PredictResponse(BaseModel):
    date: dt.date
    model_version: str
    timezone: str = "UTC"
    predictions: list[Point]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    trained_until: str


def create_app(model_path: Path | None = None, demo_data: DemoData | None = None) -> FastAPI:
    """Build the application; the model file is read when the server starts."""
    path = model_path or Path(os.environ.get("MODEL_PATH", "models/gru_final.pt"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = load_artifact(path)
        logger.info("model loaded: %s", app.state.predictor.version)
        yield

    app = FastAPI(title="conso-forecast", version="1.0", lifespan=lifespan)
    app.state.demo = demo_data or DemoData()
    app.include_router(build_router())
    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        predictor = request.app.state.predictor
        return HealthResponse(status="ok", model_version=predictor.version, trained_until=predictor.trained_until)

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        predictor = request.app.state.predictor
        index = pd.DatetimeIndex([o.timestamp.astimezone(dt.timezone.utc) for o in body.history])
        history = pd.Series([np.nan if o.consumption_mw is None else o.consumption_mw for o in body.history],
                            index=index, dtype=float)
        try:
            forecast = predictor.predict(history, body.date)
        except InvalidHistory as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        logger.info("predict date=%s history=%d model=%s", body.date, len(body.history), predictor.version)
        points = [Point(timestamp=t.to_pydatetime(), consumption_mw=round(float(v), 1)) for t, v in forecast.items()]
        return PredictResponse(date=body.date, model_version=predictor.version, predictions=points)

    return app


app = create_app()
