import numpy as np
import pandas as pd

from clearlane.modeling import _calibrate_blend, predict_models


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


def test_predict_models_applies_saved_blend_weights() -> None:
    frame = pd.DataFrame(
        {
            "feature": [0.0, 1.0],
            "count_lag_84": [3.0, 5.0],
            "impact_lag_84": [4.0, 8.0],
            "count_roll_mean_84": [1.0, 1.0],
            "impact_roll_mean_84": [2.0, 2.0],
        }
    )
    hotspot = ConstantModel(0.0)
    hotspot._clearlane_blend_weights = {"model": 0.0, "count_lag_84": 1.0}
    impact = ConstantModel(0.0)
    impact._clearlane_blend_weights = {"model": 0.0, "impact_lag_84": 1.0}
    impact._clearlane_prediction_cap = 100.0

    predictions = predict_models(
        {"hotspot": hotspot, "impact": impact},
        frame,
        ["feature"],
    )

    assert predictions["predicted_violation_count"].tolist() == [3.0, 5.0]
    assert predictions["predicted_impact_units"].tolist() == [4.0, 8.0]


def test_calibrate_blend_can_select_recent_history_signal() -> None:
    validation = pd.DataFrame(
        {
            "target_window_start": pd.to_datetime(
                ["2024-03-25 08:00:00+05:30"] * 3
                + ["2024-03-25 10:00:00+05:30"] * 3
            ),
            "target_violation_count": [9.0, 1.0, 0.0, 7.0, 2.0, 0.0],
            "count_lag_84": [9.0, 1.0, 0.0, 7.0, 2.0, 0.0],
            "count_roll_mean_84": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    config = {
        "model": {
            "calibration_grid_step": 0.5,
            "minimum_model_blend_weight": 0.0,
            "calibration_ranking_weight": 0.15,
        },
        "deployment": {"top_k_metrics": [1]},
    }

    weights, metrics = _calibrate_blend(
        validation,
        raw_prediction=np.zeros(len(validation)),
        kind="count",
        config=config,
    )

    assert weights["count_lag_84"] == 1.0
    assert metrics["mae"] == 0.0
