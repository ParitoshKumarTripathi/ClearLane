from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


class WeightedModelEnsemble(RegressorMixin, BaseEstimator):
    def __init__(self, models: list[Any], weights: list[float]) -> None:
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        self.models = models
        self.weights = [float(weight) / total for weight in weights]

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> "WeightedModelEnsemble":
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        predictions = [
            weight * np.asarray(model.predict(x), dtype=float)
            for model, weight in zip(self.models, self.weights, strict=True)
        ]
        return np.sum(predictions, axis=0)


def _resolved_model_config(config: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    base = {
        key: value
        for key, value in config["model"].items()
        if not isinstance(value, dict)
    }
    if model_name:
        overrides = config["model"].get(model_name, {})
        if isinstance(overrides, dict):
            base.update(overrides)
    return base


def _config_with_model_overrides(
    config: dict[str, Any],
    model_name: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    patched = deepcopy(config)
    existing = patched["model"].get(model_name, {})
    if not isinstance(existing, dict):
        existing = {}
    patched["model"][model_name] = {**existing, **overrides}
    return patched


def build_model(
    config: dict[str, Any],
    random_state: int,
    loss: str = "poisson",
    model_name: str | None = None,
) -> HistGradientBoostingRegressor:
    model_config = _resolved_model_config(config, model_name)
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=float(model_config["learning_rate"]),
        max_iter=int(model_config["max_iter"]),
        max_leaf_nodes=int(model_config["max_leaf_nodes"]),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        l2_regularization=float(model_config["l2_regularization"]),
        early_stopping=bool(model_config["early_stopping"]),
        validation_fraction=float(model_config["validation_fraction"]),
        n_iter_no_change=12,
        random_state=random_state,
    )


def _build_impact_model(config: dict[str, Any], seed: int) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=build_model(config, seed, loss="squared_error", model_name="impact"),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def prepare_training_sample(
    train: pd.DataFrame,
    config: dict[str, Any],
    target_column: str = "target_violation_count",
    model_name: str = "hotspot",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    seed = int(config["project"]["random_seed"])
    model_config = _resolved_model_config(config, model_name)
    ratio = float(model_config.get("zero_to_positive_ratio", 2.0))
    positive = train.loc[train[target_column].gt(0)]
    negative = train.loc[train[target_column].eq(0)]
    desired_negatives = min(len(negative), int(max(len(positive), 1) * ratio))
    sampled_negative = negative.sample(n=desired_negatives, random_state=seed)
    sample = pd.concat([positive, sampled_negative], ignore_index=False).sample(
        frac=1.0, random_state=seed
    )

    negative_weight = len(negative) / max(len(sampled_negative), 1)
    weights = np.where(sample["target_violation_count"].eq(0), negative_weight, 1.0)
    info: dict[str, Any] = {
        "full_training_rows": int(len(train)),
        "positive_rows": int(len(positive)),
        "full_negative_rows": int(len(negative)),
        "sampled_negative_rows": int(len(sampled_negative)),
        "model_fit_rows": int(len(sample)),
        "negative_sample_weight": float(negative_weight),
    }
    return sample, weights.astype("float32"), info


def _fit_model_pair(
    train: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = int(config["project"]["random_seed"])
    hotspot_sample, hotspot_weights, hotspot_info = prepare_training_sample(
        train,
        config,
        target_column="target_violation_count",
        model_name="hotspot",
    )
    x_hotspot = hotspot_sample[feature_columns].astype("float32")

    hotspot_model = build_model(config, seed, loss="poisson", model_name="hotspot")
    hotspot_model.fit(
        x_hotspot,
        hotspot_sample["target_violation_count"].astype(float),
        sample_weight=hotspot_weights,
    )

    impact_config = _resolved_model_config(config, "impact")
    ensemble_members = impact_config.get("ensemble_members")
    impact_models: list[Any] = []
    impact_model_weights: list[float] = []
    impact_infos: list[dict[str, Any]] = []
    impact_caps: list[float] = []
    if isinstance(ensemble_members, list) and ensemble_members:
        for index, member in enumerate(ensemble_members):
            if not isinstance(member, dict):
                continue
            member_weight = float(member.get("weight", 1.0))
            overrides = {
                key: value
                for key, value in member.items()
                if key not in {"name", "weight"}
            }
            member_config = _config_with_model_overrides(config, "impact", overrides)
            member_sample, member_weights, member_info = prepare_training_sample(
                train,
                member_config,
                target_column="target_impact_units",
                model_name="impact",
            )
            member_model = _build_impact_model(member_config, seed + 1)
            member_model.fit(
                member_sample[feature_columns].astype("float32"),
                member_sample["target_impact_units"].astype(float),
                sample_weight=member_weights,
            )
            impact_models.append(member_model)
            impact_model_weights.append(member_weight)
            member_info["name"] = str(member.get("name", f"member_{index + 1}"))
            member_info["weight"] = member_weight
            impact_infos.append(member_info)
            impact_caps.append(float(member_sample["target_impact_units"].quantile(0.999) * 1.5))
        impact_model = WeightedModelEnsemble(impact_models, impact_model_weights)
        impact_info = {
            "ensemble_members": impact_infos,
            "model_fit_rows": int(max(info["model_fit_rows"] for info in impact_infos)),
            "positive_rows": int(max(info["positive_rows"] for info in impact_infos)),
        }
    else:
        impact_sample, impact_weights, impact_info = prepare_training_sample(
            train,
            config,
            target_column="target_impact_units",
            model_name="impact",
        )
        impact_model = _build_impact_model(config, seed + 1)
        impact_model.fit(
            impact_sample[feature_columns].astype("float32"),
            impact_sample["target_impact_units"].astype(float),
            sample_weight=impact_weights,
        )
        impact_caps.append(float(impact_sample["target_impact_units"].quantile(0.999) * 1.5))

    # Guard against implausible extrapolation in a sparse, heavy-tailed target.
    impact_model._clearlane_prediction_cap = float(max(max(impact_caps), 1.0))
    training_info = {
        "full_training_rows": int(len(train)),
        "positive_rows": int(hotspot_info["positive_rows"]),
        "model_fit_rows": int(
            max(hotspot_info["model_fit_rows"], impact_info["model_fit_rows"])
        ),
        "hotspot_sample": hotspot_info,
        "impact_sample": impact_info,
    }
    return {"hotspot": hotspot_model, "impact": impact_model}, training_info


def _temporal_validation_split(
    train: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_windows = int(config["model"].get("calibration_windows", 84))
    windows = sorted(train["target_window_start"].drop_duplicates().tolist())
    if validation_windows < 1 or len(windows) <= validation_windows:
        return train, train.iloc[0:0].copy()
    cutoff = windows[-validation_windows]
    fit = train.loc[train["target_window_start"].lt(cutoff)].copy()
    validation = train.loc[train["target_window_start"].ge(cutoff)].copy()
    if fit.empty or validation.empty:
        return train, train.iloc[0:0].copy()
    return fit, validation


def _baseline_columns(kind: str) -> list[str]:
    if kind == "count":
        return ["count_lag_84", "count_roll_mean_84"]
    return ["impact_lag_84", "impact_roll_mean_84"]


def _blend_prediction(
    frame: pd.DataFrame,
    raw_prediction: np.ndarray,
    kind: str,
    weights: dict[str, float],
) -> np.ndarray:
    prediction = raw_prediction.astype(float) * float(weights.get("model", 1.0))
    for column in _baseline_columns(kind):
        if column in frame.columns:
            prediction += frame[column].astype(float).to_numpy() * float(weights.get(column, 0.0))
    return np.clip(prediction, 0.0, None)


def _candidate_weight_sets(
    sources: list[str],
    step: float,
    minimum_model_weight: float,
) -> list[dict[str, float]]:
    units = max(int(round(1.0 / step)), 1)
    minimum_model_units = max(int(round(minimum_model_weight * units)), 0)
    candidates: list[dict[str, float]] = []
    if len(sources) == 1:
        return [{sources[0]: 1.0}]

    for allocations in product(range(units + 1), repeat=len(sources) - 1):
        model_units = allocations[0]
        if model_units < minimum_model_units:
            continue
        used = sum(allocations)
        if used > units:
            continue
        all_units = list(allocations) + [units - used]
        candidates.append(
            {
                source: round(unit / units, 4)
                for source, unit in zip(sources, all_units, strict=True)
            }
        )
    return candidates


def _weighted_capture_at_k(
    frame: pd.DataFrame,
    actual_column: str,
    predicted_values: np.ndarray,
    k: int,
) -> float:
    scored = frame[["target_window_start", actual_column]].copy()
    scored["_prediction"] = predicted_values
    captured = 0.0
    total = 0.0
    for _, group in scored.groupby("target_window_start", sort=False):
        actual_total = float(group[actual_column].sum())
        if actual_total <= 0:
            continue
        top = group.nlargest(min(k, len(group)), "_prediction")
        captured += float(top[actual_column].sum())
        total += actual_total
    return captured / total if total else 0.0


def _calibrate_blend(
    validation: pd.DataFrame,
    raw_prediction: np.ndarray,
    kind: str,
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    actual_column = "target_violation_count" if kind == "count" else "target_impact_units"
    actual = validation[actual_column].astype(float).to_numpy()
    available_sources = ["model"] + [
        column for column in _baseline_columns(kind) if column in validation.columns
    ]
    model_config = config["model"]
    candidates = _candidate_weight_sets(
        available_sources,
        step=float(model_config.get("calibration_grid_step", 0.1)),
        minimum_model_weight=float(model_config.get("minimum_model_blend_weight", 0.5)),
    )
    k_values = config.get("deployment", {}).get("top_k_metrics", [10])
    k = int(k_values[0]) if k_values else 10
    ranking_weight = float(model_config.get("calibration_ranking_weight", 0.15))
    scale = max(float(np.mean(np.abs(actual))), 1e-6)

    best_weights = {"model": 1.0}
    best_metrics = {"mae": float("inf"), "weighted_capture": 0.0, "objective": float("inf")}
    for weights in candidates:
        blended = _blend_prediction(validation, raw_prediction, kind, weights)
        mae = float(np.mean(np.abs(actual - blended)))
        capture = _weighted_capture_at_k(validation, actual_column, blended, k)
        objective = (mae / scale) - (ranking_weight * capture)
        if objective < best_metrics["objective"]:
            best_weights = weights
            best_metrics = {
                "mae": mae,
                "weighted_capture": float(capture),
                "objective": float(objective),
            }
    return best_weights, best_metrics


def train_models(
    train: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_frame, validation_frame = _temporal_validation_split(train, config)
    count_blend = {"model": 1.0}
    impact_blend = {"model": 1.0}
    calibration_info: dict[str, Any] = {
        "calibration_rows": int(len(validation_frame)),
        "calibration_windows": int(validation_frame["target_window_start"].nunique())
        if not validation_frame.empty
        else 0,
    }

    if not validation_frame.empty:
        calibration_models, calibration_fit_info = _fit_model_pair(
            fit_frame, feature_columns, config
        )
        validation_predictions = predict_models(
            calibration_models,
            validation_frame,
            feature_columns,
            apply_blend=False,
        )
        count_blend, count_metrics = _calibrate_blend(
            validation_frame,
            validation_predictions["predicted_violation_count"].to_numpy(),
            "count",
            config,
        )
        impact_blend, impact_metrics = _calibrate_blend(
            validation_frame,
            validation_predictions["predicted_impact_units"].to_numpy(),
            "impact",
            config,
        )
        calibration_info.update(
            {
                "calibration_fit_rows": calibration_fit_info["model_fit_rows"],
                "count_blend_weights": count_blend,
                "count_blend_metrics": count_metrics,
                "impact_blend_weights": impact_blend,
                "impact_blend_metrics": impact_metrics,
            }
        )

    models, training_info = _fit_model_pair(train, feature_columns, config)
    models["hotspot"]._clearlane_blend_weights = count_blend
    models["impact"]._clearlane_blend_weights = impact_blend
    training_info.update(calibration_info)
    return models, training_info


def predict_models(
    models: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    apply_blend: bool = True,
) -> pd.DataFrame:
    x = frame[feature_columns].astype("float32")
    predictions = frame.copy()
    count_prediction = np.clip(models["hotspot"].predict(x), 0.0, None)
    if apply_blend:
        count_prediction = _blend_prediction(
            frame,
            count_prediction,
            "count",
            getattr(models["hotspot"], "_clearlane_blend_weights", {"model": 1.0}),
        )
    predictions["predicted_violation_count"] = count_prediction
    impact_cap = float(getattr(models["impact"], "_clearlane_prediction_cap", np.inf))
    impact_prediction = np.clip(models["impact"].predict(x), 0.0, impact_cap)
    if apply_blend:
        impact_prediction = _blend_prediction(
            frame,
            impact_prediction,
            "impact",
            getattr(models["impact"], "_clearlane_blend_weights", {"model": 1.0}),
        )
    predictions["predicted_impact_units"] = np.clip(impact_prediction, 0.0, impact_cap)
    return predictions


def calculate_feature_importance(
    models: dict[str, Any],
    validation: pd.DataFrame,
    feature_columns: list[str],
    random_state: int,
    max_rows: int = 2_000,
) -> pd.DataFrame:
    sample = validation.sample(
        n=min(max_rows, len(validation)),
        random_state=random_state,
    )
    x = sample[feature_columns].astype("float32")
    outputs: list[pd.DataFrame] = []
    targets = {
        "hotspot": "target_violation_count",
        "impact": "target_impact_units",
    }
    for name, model in models.items():
        result = permutation_importance(
            model,
            x,
            sample[targets[name]].astype(float),
            scoring="neg_mean_absolute_error",
            n_repeats=1,
            random_state=random_state,
            n_jobs=1,
        )
        outputs.append(
            pd.DataFrame(
                {
                    "model": name,
                    "feature": feature_columns,
                    "importance": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
        )
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["model", "importance"], ascending=[True, False]
    )


def save_models(models: dict[str, Any], output_directory: str | Path) -> None:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["hotspot"], directory / "hotspot_model.joblib")
    joblib.dump(models["impact"], directory / "impact_model.joblib")


def load_models(directory: str | Path) -> dict[str, Any]:
    path = Path(directory)
    return {
        "hotspot": joblib.load(path / "hotspot_model.joblib"),
        "impact": joblib.load(path / "impact_model.joblib"),
    }
