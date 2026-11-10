import numpy as np
import pandas as pd

from src.make_dashboard_data import MODELS, build_results


def _frame() -> pd.DataFrame:
    """A year of hourly data in two folds, every model with a different constant error."""
    index = pd.date_range("2024-01-01", periods=366 * 24, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    truth = pd.Series(50_000 + rng.normal(0, 1_000, len(index)), index=index)
    frame = pd.DataFrame({"truth": truth})
    for position, model in enumerate(MODELS):
        frame[model] = truth + 100 * (position + 1)                      # model k is off by exactly 100 (k + 1) MW
    frame["fold"] = np.where(np.arange(len(index)) < len(index) // 2, "A", "B")
    return frame


def test_results_have_the_expected_shape_and_numbers():
    results = build_results(_frame())
    assert results["by_fold"]["folds"] == ["A", "B"]
    assert [m["id"] for m in results["models"]] == list(MODELS)
    for position, model in enumerate(MODELS):
        assert results["mean"][model]["mae"] == 100 * (position + 1)     # constant error: MAE = the error
        assert results["by_fold"]["MAE (MW)"][model] == [100 * (position + 1)] * 2
        assert len(results["by_hour"]["values"][model]) == 24
        assert len(results["by_weekday"]["values"][model]) == 7
        assert len(results["by_season"]["values"][model]) == 4


def test_days_are_ranked_by_the_daily_error_of_the_final_model():
    frame = _frame()
    day = pd.Timestamp("2024-05-05", tz="Europe/Paris")
    in_day = (frame.index >= day) & (frame.index < day + pd.Timedelta(days=1))
    frame.loc[in_day, "gru_residual_ensemble"] += 5_000                  # one very bad day for the final model
    days = build_results(frame)["days"]
    assert days["worst"][0]["date"] == "2024-05-05" and days["worst"][0]["rank"] == 1
    assert [w["mae"] for w in days["worst"]] == sorted((w["mae"] for w in days["worst"]), reverse=True)
    assert days["count"] >= 365 and 0 < days["typical"]["rank"] < days["count"]
    assert days["typical"]["mae"] < days["worst"][0]["mae"]


def test_every_model_is_scored_on_the_same_rows():
    frame = _frame()
    frame.iloc[10, frame.columns.get_loc("sarima")] = np.nan             # one missing prediction for one model
    results = build_results(frame)
    assert results["meta"]["rows"] == len(frame) - 1                     # the row is dropped for all the models
