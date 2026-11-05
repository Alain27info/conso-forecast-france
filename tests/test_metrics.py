import numpy as np
import pytest

from src.metrics import mae, mape, rmse


def test_mae_hand_computed():
    assert mae([100, 200], [110, 180]) == 15.0          # (10 + 20) / 2


def test_rmse_hand_computed():
    assert rmse([100, 200], [110, 180]) == pytest.approx(np.sqrt(250))   # sqrt((100 + 400) / 2)


def test_rmse_punishes_one_big_error_more_than_mae():
    spread = ([0, 0, 0, 0], [10, 10, 10, 10])           # four errors of 10
    one_big = ([0, 0, 0, 0], [0, 0, 0, 40])             # one error of 40, same total absolute error
    assert mae(*one_big) == mae(*spread) == 10
    assert rmse(*one_big) > rmse(*spread)


def test_mape_hand_computed():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)   # (10% + 10%) / 2


def test_mape_is_capped_at_100_percent_when_under_forecasting_but_not_when_over_forecasting():
    assert mape([100], [0]) == 100.0                    # worst possible under-forecast (predict 0)
    assert mape([100], [300]) == 200.0                  # over-forecasts can exceed 100%


def test_mape_rejects_zero_truth():
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])
