from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error


def _regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = actual.astype(float).to_numpy()
    predicted_values = np.clip(predicted.astype(float).to_numpy(), 1e-9, None)
    actual_rank = pd.Series(actual_values).rank(method="average")
    predicted_rank = pd.Series(predicted_values).rank(method="average")
    rank_correlation = actual_rank.corr(predicted_rank)
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(mean_squared_error(actual_values, predicted_values) ** 0.5),
        "poisson_deviance": float(mean_poisson_deviance(actual_values, predicted_values)),
        "rank_correlation": float(rank_correlation if pd.notna(rank_correlation) else 0.0),
    }


def _ranking_metrics_by_window(
    frame: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    top_k: int,
) -> dict[str, float]:
    precision_values: list[float] = []
    capture_values: list[float] = []
    weighted_actual_captured = 0.0
    weighted_actual_total = 0.0

    for _, group in frame.groupby("target_window_start", sort=False):
        k = min(top_k, len(group))
        if k == 0:
            continue
        predicted_top = set(group.nlargest(k, predicted_column)["zone_id"])
        actual_top = set(group.nlargest(k, actual_column)["zone_id"])
        precision_values.append(len(predicted_top & actual_top) / k)

        actual_total = float(group[actual_column].sum())
        captured = float(group.loc[group["zone_id"].isin(predicted_top), actual_column].sum())
        capture_values.append(captured / actual_total if actual_total > 0 else 0.0)
        weighted_actual_captured += captured
        weighted_actual_total += actual_total

    return {
        f"precision_at_{top_k}": float(np.mean(precision_values) if precision_values else 0.0),
        f"mean_capture_at_{top_k}": float(np.mean(capture_values) if capture_values else 0.0),
        f"weighted_capture_at_{top_k}": float(
            weighted_actual_captured / weighted_actual_total if weighted_actual_total else 0.0
        ),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "evaluation_start": predictions["target_window_start"].min().isoformat(),
        "evaluation_end": predictions["target_window_start"].max().isoformat(),
        "rows": int(len(predictions)),
        "windows": int(predictions["target_window_start"].nunique()),
        "zones": int(predictions["zone_id"].nunique()),
    }

    systems = {
        "clearlane": {
            "count": "predicted_violation_count",
            "impact": "predicted_impact_units",
        },
        "same_window_last_week": {
            "count": "count_lag_84",
            "impact": "impact_lag_84",
        },
        "seven_day_average": {
            "count": "count_roll_mean_84",
            "impact": "impact_roll_mean_84",
        },
    }
    for system_name, columns in systems.items():
        system_metrics: dict[str, Any] = {
            "count": _regression_metrics(
                predictions["target_violation_count"], predictions[columns["count"]]
            ),
            "impact": _regression_metrics(
                predictions["target_impact_units"], predictions[columns["impact"]]
            ),
        }
        for k in config["deployment"]["top_k_metrics"]:
            system_metrics["count"].update(
                _ranking_metrics_by_window(
                    predictions,
                    "target_violation_count",
                    columns["count"],
                    int(k),
                )
            )
            system_metrics["impact"].update(
                _ranking_metrics_by_window(
                    predictions,
                    "target_impact_units",
                    columns["impact"],
                    int(k),
                )
            )
        metrics[system_name] = system_metrics
    return metrics
