"""Build the train/test feature tables on the real data and print a summary.

Usage (from the project root):
    python -m src.run_features
"""
import pandas as pd

from src.data import load_raw, to_half_hourly
from src.features import train_test_datasets


def main() -> None:
    """Print the shape, missing values and correlation with the target of each feature."""
    series = to_half_hourly(load_raw())["consommation"]
    X_train, y_train, X_test, y_test = train_test_datasets(series)
    pd.set_option("display.width", 120)
    print(f"Train: {X_train.shape[0]} rows, {X_train.index[0]} to {X_train.index[-1]}")
    print(f"Test : {X_test.shape[0]} rows, {X_test.index[0]} to {X_test.index[-1]}")
    print(f"Features ({X_train.shape[1]}): {list(X_train.columns)}")
    print("\nMissing values per feature (train / test):")
    print(pd.DataFrame({"train": X_train.isna().sum(), "test": X_test.isna().sum()}).query("train > 0 or test > 0").to_string())
    print("\nPearson correlation with the target (train only):")
    print(X_train.corrwith(y_train).sort_values(key=abs, ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
