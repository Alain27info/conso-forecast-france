"""Score the two baselines (and RTE's own J-1 forecast, as an external reference).

Usage (from the project root):
    python -m src.run_baselines
"""
import pandas as pd

from src.baselines import forecast_day_before, forecast_week_before
from src.config import TEST_END, TEST_START
from src.data import load_raw, to_half_hourly
from src.evaluate import backtest, compare, mae_by_weekday


def main() -> None:
    """Load the data, run both baselines on the test period and print the tables."""
    df = to_half_hourly(load_raw())
    y = df["consommation"]
    test = y[(y.index >= TEST_START) & (y.index < TEST_END)]

    predictions = {
        "baseline_day_before": backtest(forecast_day_before, y),
        "baseline_week_before": backtest(forecast_week_before, y),
        "RTE_J-1 (external reference)": df["prevision_j1"].reindex(test.index),
    }
    pd.set_option("display.width", 120)
    print(f"Test period: {TEST_START.date()} to {(TEST_END - pd.Timedelta(days=1)).date()} (Paris days)")
    print(f"Test slots: {len(test)}, of which with a measured value: {int(test.notna().sum())}")
    print("\nScores (same rows for every model):")
    print(compare(test, predictions).round(2).to_string())
    print("\nMAE (MW) by weekday (0 = Monday):")
    print(mae_by_weekday(test, predictions).round(0).to_string())


if __name__ == "__main__":
    main()
