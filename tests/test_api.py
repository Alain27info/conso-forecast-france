import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import create_app
from src.config import TZ
from src.serving import load_artifact

DAY = dt.date(2024, 5, 10)


def _payload(y: pd.Series, day: dt.date = DAY, days: int = 16) -> dict:
    """Request body: `days` days of history ending the day before `day`."""
    end = pd.Timestamp(day).tz_localize(TZ)
    history = y[(y.index < end) & (y.index >= end - pd.Timedelta(days=days))]
    return {"date": day.isoformat(),
            "history": [{"timestamp": t.isoformat(), "consumption_mw": round(float(v), 1)} for t, v in history.items()]}


@pytest.fixture()
def client(tiny_model):
    path, y, *_ = tiny_model
    with TestClient(create_app(path)) as test_client:
        yield test_client, y


def test_health_reports_the_loaded_model(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "tiny-test", "trained_until": "2024-04-14"}


def test_predict_returns_48_increasing_half_hours_starting_at_local_midnight(client):
    test_client, y = client
    response = test_client.post("/predict", json=_payload(y))
    assert response.status_code == 200
    body = response.json()
    stamps = pd.DatetimeIndex([p["timestamp"] for p in body["predictions"]])
    assert len(stamps) == 48 and stamps.is_monotonic_increasing
    assert stamps[0] == pd.Timestamp(DAY).tz_localize(TZ).tz_convert("UTC")
    assert all(np.isfinite(p["consumption_mw"]) for p in body["predictions"])
    assert body["model_version"] == "tiny-test" and body["timezone"] == "UTC"


def test_api_numbers_match_the_predictor(client, tiny_model):
    test_client, y = client
    path, *_ = tiny_model
    payload = _payload(y)
    served = [p["consumption_mw"] for p in test_client.post("/predict", json=payload).json()["predictions"]]
    history = pd.Series({pd.Timestamp(o["timestamp"]): o["consumption_mw"] for o in payload["history"]})
    expected = load_artifact(path).predict(history, DAY)
    np.testing.assert_allclose(served, expected.to_numpy(), atol=0.06)       # the API rounds to 0.1 MW


def test_null_values_are_accepted_as_missing(client):
    test_client, y = client
    payload = _payload(y)
    for i in (-50, -49, -48):
        payload["history"][i]["consumption_mw"] = None
    assert test_client.post("/predict", json=payload).status_code == 200


def test_data_from_the_target_day_is_refused_with_422(client):
    test_client, y = client
    payload = _payload(y)
    payload["history"].append({"timestamp": pd.Timestamp(DAY).tz_localize(TZ).isoformat(), "consumption_mw": 50000.0})
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "day to forecast" in response.json()["detail"]


def test_bad_requests_are_refused_with_422(client):
    test_client, y = client
    good = _payload(y)
    naive = {**good, "history": [{**o, "timestamp": o["timestamp"][:19]} for o in good["history"]]}
    assert test_client.post("/predict", json=naive).status_code == 422                       # no timezone
    assert test_client.post("/predict", json={**good, "history": good["history"][-100:]}).status_code == 422   # too short
    assert test_client.post("/predict", json={**good, "history": []}).status_code == 422
    assert test_client.post("/predict", json={**good, "date": "not-a-date"}).status_code == 422
    negative = {**good, "history": [{**good["history"][0], "consumption_mw": -5.0}, *good["history"][1:]]}
    assert test_client.post("/predict", json=negative).status_code == 422
    assert test_client.post("/predict", json={"date": DAY.isoformat()}).status_code == 422   # history missing


def test_model_is_loaded_once_for_many_requests(tiny_model, monkeypatch):
    path, y, *_ = tiny_model
    calls = []

    def counting_loader(p):
        calls.append(p)
        return load_artifact(p)

    monkeypatch.setattr(api_main, "load_artifact", counting_loader)
    with TestClient(create_app(path)) as test_client:
        for _ in range(3):
            assert test_client.post("/predict", json=_payload(y)).status_code == 200
    assert len(calls) == 1


def test_server_refuses_to_start_without_a_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "absent.pt")):
            pass
