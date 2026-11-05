"""Forecast error metrics. Inputs are array-likes of the same length, in MW."""
import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float arrays and check that their shapes match."""
    t, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    return t, p


def mae(y_true, y_pred) -> float:
    """Mean absolute error: average size of the error, in MW."""
    t, p = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in MW: punishes large errors more than MAE."""
    t, p = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in %: undefined if a true value is 0."""
    t, p = _as_arrays(y_true, y_pred)
    if np.any(t == 0):
        raise ValueError("MAPE is undefined when a true value is 0")
    return float(np.mean(np.abs((t - p) / t)) * 100)
