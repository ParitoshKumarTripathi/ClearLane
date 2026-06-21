from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clearlane.config import load_config, project_root
from clearlane.evaluation import evaluate_predictions
from clearlane.features import build_zone_time_features, model_feature_columns
from clearlane.modeling import (
    calculate_feature_importance,
    predict_models,
    save_models,
    train_models,
)
from clearlane.optimizer import add_operational_scores, optimize_patrol_plan
from clearlane.preprocessing import load_and_clean_records


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _as_timestamp(value: str, timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    return timestamp


def run_pipeline(
    input_path: str | Path,
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    save_intermediate: bool = True,
) -> dict[str, Any]:
    output_root = Path(root) if root else project_root()
    config = load_config(config_path)
    artifacts_directory = output_root / "artifacts"
    models_directory = output_root / "models"
    processed_directory = output_root / "data" / "processed"
    artifacts_directory.mkdir(parents=True, exist_ok=True)
    models_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)

    print("[1/6] Cleaning raw violation records")
    records, data_summary = load_and_clean_records(input_path, config)
    if save_intermediate:
        records.to_parquet(processed_directory / "cleaned_violations.parquet", index=False)

    print("[2/6] Building spatial-temporal training table")
    feature_frame, zone_reference = build_zone_time_features(records, config)
    features = model_feature_columns(config)
    if save_intermediate:
        feature_frame.to_parquet(processed_directory / "zone_time_features.parquet", index=False)

    timezone = config["data"]["timezone"]
    test_start = _as_timestamp(config["temporal"]["test_start"], timezone)
    test_end = _as_timestamp(config["temporal"]["test_end"], timezone)
    train = feature_frame.loc[feature_frame["target_window_start"].lt(test_start)].copy()
    test = feature_frame.loc[
        feature_frame["target_window_start"].ge(test_start)
        & feature_frame["target_window_start"].lt(test_end)
    ].copy()
    if train.empty or test.empty:
        raise ValueError(
            "The configured temporal split produced an empty train or test set. "
            "Adjust temporal.test_start and temporal.test_end in config.yaml."
        )

    print(f"[3/6] Training two models on {len(train):,} zone-window rows")
    models, training_info = train_models(train, features, config)
    print(
        f"    Model fit sample: {training_info['model_fit_rows']:,} rows "
        f"({training_info['positive_rows']:,} positive windows)",
        flush=True,
    )
    save_models(models, models_directory)

    print(f"[4/6] Predicting and backtesting on {len(test):,} zone-window rows")
    predictions = predict_models(models, test, features)
    predictions = add_operational_scores(predictions)
    metrics = evaluate_predictions(predictions, config)

    print("[5/6] Calculating explainability and operational artifacts")
    importance = calculate_feature_importance(
        models,
        test,
        features,
        random_state=int(config["project"]["random_seed"]),
    )

    peak_predictions = predictions.loc[predictions["target_is_peak"].eq(1)]
    window_totals = (
        peak_predictions if not peak_predictions.empty else predictions
    ).groupby("target_window_start")["predicted_impact_units"].sum()
    default_window = window_totals.idxmax()
    default_predictions = predictions.loc[
        predictions["target_window_start"].eq(default_window)
    ].copy()
    patrol_plan = optimize_patrol_plan(
        default_predictions,
        teams=int(config["deployment"]["default_teams"]),
        minimum_distance_meters=float(config["deployment"]["minimum_team_distance_meters"]),
    )

    prediction_columns = [
        "zone_id",
        "target_window_start",
        "zone_name",
        "police_station",
        "primary_junction",
        "centroid_latitude",
        "centroid_longitude",
        "predicted_violation_count",
        "predicted_impact_units",
        "target_violation_count",
        "target_impact_units",
        "impact_score",
        "risk_level",
        "model_confidence",
        "priority_score",
        "top_reason",
        "trend_ratio",
        "junction_share_84",
        "junction_offence_share_84",
        "main_road_share_84",
        "heavy_vehicle_share_84",
        "high_severity_share_84",
        "repeat_vehicle_rate_84",
        "count_roll_mean_84",
        "impact_roll_mean_84",
        "history_observations",
        "target_is_peak",
    ]
    display_predictions = predictions[prediction_columns].copy()
    display_predictions.to_csv(
        artifacts_directory / "demo_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    history_start = test_start - pd.Timedelta(days=35)
    history = feature_frame.loc[
        feature_frame["window_start"].ge(history_start)
        & feature_frame["window_start"].lt(test_end),
        [
            "zone_id",
            "window_start",
            "violation_count",
            "impact_units",
            "police_station",
            "zone_name",
        ],
    ].copy()
    history.to_csv(
        artifacts_directory / "zone_history.csv.gz",
        index=False,
        compression="gzip",
    )
    zone_reference.to_csv(artifacts_directory / "zone_reference.csv", index=False)
    importance.to_csv(artifacts_directory / "feature_importance.csv", index=False)
    patrol_plan.to_csv(artifacts_directory / "demo_patrol_plan.csv", index=False)

    station_summary = (
        records.groupby("police_station")
        .agg(
            violation_records=("id", "size"),
            unique_zones=("zone_id", "nunique"),
            mean_obstruction_units=("obstruction_units", "mean"),
            high_severity_share=("is_high_severity", "mean"),
        )
        .reset_index()
        .sort_values("violation_records", ascending=False)
    )
    station_summary.to_csv(artifacts_directory / "police_station_summary.csv", index=False)

    data_summary.update(
        {
            "active_model_zones": int(feature_frame["zone_id"].nunique()),
            "training_rows": int(len(train)),
            "model_fit_rows": int(training_info["model_fit_rows"]),
            "test_rows": int(len(test)),
            "test_windows": int(test["target_window_start"].nunique()),
            "default_demo_window": default_window.isoformat(),
            "grid_size_meters": int(config["spatial"]["grid_size_meters"]),
            "window_hours": int(config["temporal"]["window_hours"]),
        }
    )
    _write_json(artifacts_directory / "data_summary.json", data_summary)
    _write_json(artifacts_directory / "metrics.json", metrics)

    metadata = {
        "project": config["project"]["name"],
        "model_type": "HistGradientBoostingRegressor with weighted impact ensemble",
        "hotspot_model_type": "HistGradientBoostingRegressor",
        "impact_model_type": "WeightedModelEnsemble",
        "hotspot_loss": "poisson",
        "impact_loss": "squared_error",
        "impact_target_transform": "log1p/expm1",
        "model_config": config["model"],
        "random_seed": int(config["project"]["random_seed"]),
        "feature_columns": features,
        "training_rows": int(len(train)),
        "training_sample": training_info,
        "training_end": train["target_window_start"].max().isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "grid_size_meters": int(config["spatial"]["grid_size_meters"]),
        "window_hours": int(config["temporal"]["window_hours"]),
        "validation_policy": config["data"]["validation_policy"],
        "default_demo_window": default_window.isoformat(),
    }
    _write_json(models_directory / "model_metadata.json", metadata)

    run_manifest = {
        "input_path": str(Path(input_path).resolve()),
        "artifacts": sorted(path.name for path in artifacts_directory.iterdir() if path.is_file()),
        "models": sorted(path.name for path in models_directory.iterdir() if path.is_file()),
        "data_summary": data_summary,
    }
    _write_json(artifacts_directory / "run_manifest.json", run_manifest)
    print("[6/6] Pipeline complete")
    return {"metrics": metrics, "data_summary": data_summary, "metadata": metadata}
