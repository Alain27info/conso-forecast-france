import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.demo as demo_module
from api.demo import DemoData
from api.main import STATIC_DIR, create_app
from src.config import TZ


@pytest.fixture()
def client(tiny_model):
    """App with the tiny test model and a synthetic dataset (RTE forecast = 1 % above the truth)."""
    path, y, *_ = tiny_model
    y = y[y.index < pd.Timestamp("2024-11-20", tz=TZ)]                 # the data ends at 23:30 of 2024-11-19, like the real data
    frame = pd.DataFrame({"consommation": y, "prevision_j1": y * 1.01})
    with TestClient(create_app(path, demo_data=DemoData(frame))) as test_client:
        yield test_client, y


def test_home_page_and_static_files_are_served(client):
    test_client, _ = client
    page = test_client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/static/app.js" in page.text and "conso-forecast" in page.text
    for name in ("style.css", "charts.js", "app.js"):
        assert test_client.get(f"/static/{name}").status_code == 200


def test_results_file_used_by_the_dashboard_is_present_and_consistent():
    results = json.loads((STATIC_DIR / "results.json").read_text(encoding="utf-8"))
    assert results["by_fold"]["folds"] == results["meta"]["folds"] and len(results["meta"]["folds"]) == 5
    for model in results["models"]:
        assert len(results["by_fold"]["MAE (MW)"][model["id"]]) == 5
        assert len(results["by_hour"]["values"][model["id"]]) == 24
        assert len(results["by_weekday"]["values"][model["id"]]) == 7
        assert len(results["by_season"]["values"][model["id"]]) == 4
    assert results["mean"]["gru_residual_ensemble"]["mae"] < results["mean"]["baseline_day_before"]["mae"]
    assert len(results["days"]["worst"]) == 3 and results["days"]["count"] > 1000


def test_range_gives_the_days_the_page_can_forecast(client):
    test_client, y = client
    body = test_client.get("/demo/range").json()
    assert body["last_observation_date"] == "2024-11-19"
    assert body["last_date"] == "2024-11-20"                            # the day after the last complete day
    assert body["first_date"] < body["last_date"] and body["trained_until"] == "2024-04-14"


def test_last_day_is_the_observed_day_itself_when_it_is_only_partly_observed(tiny_model):
    path, y, *_ = tiny_model
    partial = y[y.index <= pd.Timestamp("2024-11-20 10:00", tz=TZ)]     # the data stops in the middle of 2024-11-20
    frame = pd.DataFrame({"consommation": partial, "prevision_j1": partial})
    with TestClient(create_app(path, demo_data=DemoData(frame))) as test_client:
        assert test_client.get("/demo/range").json()["last_date"] == "2024-11-20"
        assert test_client.get("/demo/day", params={"date": "2024-11-21"}).status_code == 422


def test_day_returns_a_request_that_predict_accepts(client):
    test_client, _ = client
    body = test_client.get("/demo/day", params={"date": "2024-06-12"}).json()
    assert len(body["request"]["history"]) == 16 * 48
    assert body["in_sample"] is False                                  # after the model's training end
    assert sum(p["consumption_mw"] is not None for p in body["actual"]) == 48
    assert len(body["rte_j1"]) == len(body["day_before"]) == 48
    response = test_client.post("/predict", json=body["request"])
    assert response.status_code == 200 and len(response.json()["predictions"]) == 48


def test_demo_request_never_contains_data_from_the_target_day(client):
    """The information rule holds for the requests the page builds."""
    test_client, _ = client
    body = test_client.get("/demo/day", params={"date": "2024-06-12"}).json()
    day_start = pd.Timestamp("2024-06-12", tz=TZ)
    assert max(pd.Timestamp(o["timestamp"]) for o in body["request"]["history"]) < day_start


def test_day_inside_the_training_period_is_flagged_in_sample(client):
    test_client, _ = client
    assert test_client.get("/demo/day", params={"date": "2024-04-10"}).json()["in_sample"] is True


def test_day_after_the_last_observation_has_no_actual_curve(client):
    test_client, y = client
    body = test_client.get("/demo/day", params={"date": "2024-11-20"}).json()
    assert all(p["consumption_mw"] is None for p in body["actual"])
    assert all(p["consumption_mw"] is not None for p in body["day_before"])


def test_clock_change_days_have_46_and_50_points(client):
    test_client, _ = client
    assert len(test_client.get("/demo/day", params={"date": "2024-03-31"}).json()["actual"]) == 46
    assert len(test_client.get("/demo/day", params={"date": "2024-10-27"}).json()["actual"]) == 50


def test_dates_outside_the_available_range_are_refused(client):
    test_client, _ = client
    assert test_client.get("/demo/day", params={"date": "2030-01-01"}).status_code == 422
    assert test_client.get("/demo/day", params={"date": "2024-02-02"}).status_code == 422     # not enough history yet
    assert test_client.get("/demo/day", params={"date": "not-a-date"}).status_code == 422


def test_without_the_local_dataset_demo_routes_say_so_but_the_api_still_works(tiny_model, monkeypatch):
    path, y, *_ = tiny_model

    def missing(*args, **kwargs):
        raise FileNotFoundError("no data")

    monkeypatch.setattr(demo_module, "load_raw", missing)
    with TestClient(create_app(path, demo_data=DemoData())) as test_client:
        response = test_client.get("/demo/range")
        assert response.status_code == 503 and "python -m src.data" in response.json()["detail"]
        assert test_client.get("/health").status_code == 200
