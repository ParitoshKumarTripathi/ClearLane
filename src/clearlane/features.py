from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from clearlane.grid import GridSpec, grid_centroid


def _mode(series: pd.Series, default: str) -> str:
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned.loc[~cleaned.str.upper().isin({"", "NULL", "NAN", "NONE"})]
    if cleaned.empty:
        return default
    modes = cleaned.mode()
    return str(modes.iloc[0]) if not modes.empty else str(cleaned.iloc[0])


def _display_name(group: pd.DataFrame) -> str:
    named = group.loc[group["is_named_junction"].eq(1), "junction_name"]
    if not named.empty:
        return _mode(named, "Named junction")
    location = _mode(group["location"], "Mapped road segment")
    return location.split(",")[0].strip()[:70] or "Mapped road segment"


def build_zone_reference(records: pd.DataFrame, active_zones: list[str], config: dict[str, Any]) -> pd.DataFrame:
    spatial = config["spatial"]
    spec = GridSpec(
        origin_latitude=float(spatial["origin_latitude"]),
        origin_longitude=float(spatial["origin_longitude"]),
        grid_size_meters=int(spatial["grid_size_meters"]),
    )
    selected = records.loc[records["zone_id"].isin(active_zones)].copy()
    grouped = selected.groupby("zone_id", sort=False)
    reference = grouped.agg(
        cell_x=("cell_x", "first"),
        cell_y=("cell_y", "first"),
        police_station=("police_station", lambda s: _mode(s, "Unknown")),
        primary_junction=("junction_name", lambda s: _mode(s, "No Junction")),
        total_records=("id", "size"),
    ).reset_index()
    names = grouped.apply(_display_name, include_groups=False).rename("zone_name").reset_index()
    reference = reference.merge(names, on="zone_id", how="left")
    centroids = grid_centroid(reference["cell_x"], reference["cell_y"], spec)
    reference = pd.concat([reference.reset_index(drop=True), centroids.reset_index(drop=True)], axis=1)
    return reference


def build_zone_time_features(
    records: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    temporal = config["temporal"]
    test_start = pd.Timestamp(temporal["test_start"])
    if test_start.tzinfo is None:
        test_start = test_start.tz_localize(config["data"]["timezone"])

    training_records = records.loc[records["window_start"].lt(test_start)]
    zone_counts = training_records.groupby("zone_id").size()
    minimum = int(config["spatial"]["min_zone_observations"])
    active_zones = zone_counts.loc[zone_counts.ge(minimum)].index.astype(str).tolist()
    if not active_zones:
        raise ValueError("No zones satisfy min_zone_observations; lower the threshold in config.yaml")

    records = records.loc[records["zone_id"].isin(active_zones)].copy()
    print(f"    Active zones: {len(active_zones):,}; retained records: {len(records):,}", flush=True)
    zone_reference = build_zone_reference(records, active_zones, config)
    print("    Zone reference built", flush=True)

    aggregate = records.groupby(["zone_id", "window_start"], sort=False).agg(
        violation_count=("id", "size"),
        impact_units=("obstruction_units", "sum"),
        high_severity_count=("is_high_severity", "sum"),
        junction_count=("is_named_junction", "sum"),
        junction_offence_count=("has_junction_offence", "sum"),
        main_road_count=("has_main_road_offence", "sum"),
        heavy_vehicle_count=("is_heavy_vehicle", "sum"),
        unique_vehicles=("canonical_vehicle_number", "nunique"),
        active_devices=("device_id", "nunique"),
        active_officers=("created_by_id", "nunique"),
        mean_record_confidence=("record_confidence", "mean"),
    )

    print(f"    Aggregated observed windows: {len(aggregate):,}", flush=True)

    window_hours = int(temporal["window_hours"])
    frequency = f"{window_hours}h"
    minimum_window = records["window_start"].min()
    maximum_window = records["window_start"].max()
    all_windows = pd.date_range(minimum_window, maximum_window, freq=frequency)
    full_index = pd.MultiIndex.from_product(
        [active_zones, all_windows], names=["zone_id", "window_start"]
    )
    panel = aggregate.reindex(full_index).reset_index()
    print(f"    Dense panel rows: {len(panel):,}", flush=True)

    zero_columns = [
        "violation_count",
        "impact_units",
        "high_severity_count",
        "junction_count",
        "junction_offence_count",
        "main_road_count",
        "heavy_vehicle_count",
        "unique_vehicles",
        "active_devices",
        "active_officers",
    ]
    panel[zero_columns] = panel[zero_columns].fillna(0.0)
    panel["mean_record_confidence"] = panel["mean_record_confidence"].fillna(0.0)
    panel = panel.merge(
        zone_reference[
            [
                "zone_id",
                "cell_x",
                "cell_y",
                "centroid_latitude",
                "centroid_longitude",
                "police_station",
                "zone_name",
                "primary_junction",
            ]
        ],
        on="zone_id",
        how="left",
        validate="many_to_one",
    )
    panel = panel.sort_values(["zone_id", "window_start"]).reset_index(drop=True)

    denominator = panel["violation_count"].replace(0, np.nan)
    panel["repeat_vehicle_rate_window"] = (
        1.0 - panel["unique_vehicles"].div(denominator)
    ).fillna(0.0).clip(0.0, 1.0)

    lag_windows = sorted({int(value) for value in temporal["lag_windows"]})
    grouped_count = panel.groupby("zone_id", sort=False)["violation_count"]
    grouped_impact = panel.groupby("zone_id", sort=False)["impact_units"]
    for lag in lag_windows:
        shift_steps = max(lag - 1, 0)
        panel[f"count_lag_{lag}"] = grouped_count.shift(shift_steps)
        panel[f"impact_lag_{lag}"] = grouped_impact.shift(shift_steps)

    print("    Lag features built", flush=True)

    rolling_windows = sorted({int(value) for value in temporal["rolling_windows"]})
    base_rolling_columns = ["violation_count", "impact_units"]
    for window in rolling_windows:
        rolled = (
            panel.groupby("zone_id", sort=False)[base_rolling_columns]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .sort_index()
        )
        panel[f"count_roll_mean_{window}"] = rolled["violation_count"].to_numpy()
        panel[f"impact_roll_mean_{window}"] = rolled["impact_units"].to_numpy()

    print("    Base rolling features built", flush=True)

    context_window = 84 if 84 in rolling_windows else rolling_windows[-1]
    sum_columns = [
        "violation_count",
        "junction_count",
        "junction_offence_count",
        "main_road_count",
        "heavy_vehicle_count",
        "high_severity_count",
    ]
    rolled_sums = (
        panel.groupby("zone_id", sort=False)[sum_columns]
        .rolling(context_window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    safe_count = rolled_sums["violation_count"].replace(0, np.nan)
    for source, output in [
        ("junction_count", "junction_share_84"),
        ("junction_offence_count", "junction_offence_share_84"),
        ("main_road_count", "main_road_share_84"),
        ("heavy_vehicle_count", "heavy_vehicle_share_84"),
        ("high_severity_count", "high_severity_share_84"),
    ]:
        panel[output] = (
            rolled_sums[source].div(safe_count).fillna(0.0).clip(0.0, 1.0).to_numpy()
        )

    mean_columns = [
        "repeat_vehicle_rate_window",
        "active_devices",
        "active_officers",
        "mean_record_confidence",
    ]
    rolled_means = (
        panel.groupby("zone_id", sort=False)[mean_columns]
        .rolling(context_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["repeat_vehicle_rate_84"] = rolled_means["repeat_vehicle_rate_window"].to_numpy()
    panel["active_devices_mean_84"] = rolled_means["active_devices"].to_numpy()
    panel["active_officers_mean_84"] = rolled_means["active_officers"].to_numpy()
    panel["record_confidence_mean_84"] = rolled_means["mean_record_confidence"].to_numpy()

    print("    Context rolling features built", flush=True)

    short_window = 84 if 84 in rolling_windows else rolling_windows[0]
    long_window = 336 if 336 in rolling_windows else rolling_windows[-1]
    panel["trend_ratio"] = (
        panel[f"count_roll_mean_{short_window}"] + 0.10
    ) / (panel[f"count_roll_mean_{long_window}"] + 0.10)
    panel["impact_trend_ratio"] = (
        panel[f"impact_roll_mean_{short_window}"] + 0.10
    ) / (panel[f"impact_roll_mean_{long_window}"] + 0.10)

    zone_group = panel.groupby("zone_id", sort=False)
    panel["history_observations"] = zone_group["violation_count"].cumsum()
    history_windows = zone_group.cumcount() + 1
    panel["zone_expanding_mean_count"] = panel["history_observations"] / history_windows
    cumulative_impact = zone_group["impact_units"].cumsum()
    panel["zone_expanding_mean_impact"] = cumulative_impact / history_windows

    panel["target_window_start"] = panel["window_start"] + pd.Timedelta(hours=window_hours)
    panel["target_violation_count"] = zone_group["violation_count"].shift(-1)
    panel["target_impact_units"] = zone_group["impact_units"].shift(-1)

    target_hour = panel["target_window_start"].dt.hour
    target_weekday = panel["target_window_start"].dt.dayofweek
    panel["target_hour_sin"] = np.sin(2 * math.pi * target_hour / 24)
    panel["target_hour_cos"] = np.cos(2 * math.pi * target_hour / 24)
    panel["target_weekday_sin"] = np.sin(2 * math.pi * target_weekday / 7)
    panel["target_weekday_cos"] = np.cos(2 * math.pi * target_weekday / 7)
    panel["target_is_weekend"] = target_weekday.ge(5).astype("int8")
    panel["target_is_peak"] = (
        ((target_hour >= 7) & (target_hour < 11))
        | ((target_hour >= 16) & (target_hour < 20))
    ).astype("int8")

    print("    Targets and calendar features built", flush=True)

    # Require at least one week of history for honest forecasting.
    weekly_lag = max(lag_windows)
    panel = panel.loc[panel[f"count_lag_{weekly_lag}"].notna()].copy()
    panel = panel.loc[panel["target_violation_count"].notna()].copy()
    panel.reset_index(drop=True, inplace=True)
    return panel, zone_reference


def model_feature_columns(config: dict[str, Any]) -> list[str]:
    lag_windows = sorted({int(value) for value in config["temporal"]["lag_windows"]})
    rolling_windows = sorted({int(value) for value in config["temporal"]["rolling_windows"]})
    features = [
        "cell_x",
        "cell_y",
        "centroid_latitude",
        "centroid_longitude",
        "target_hour_sin",
        "target_hour_cos",
        "target_weekday_sin",
        "target_weekday_cos",
        "target_is_weekend",
        "target_is_peak",
    ]
    features.extend([f"count_lag_{value}" for value in lag_windows])
    features.extend([f"impact_lag_{value}" for value in lag_windows])
    features.extend([f"count_roll_mean_{value}" for value in rolling_windows])
    features.extend([f"impact_roll_mean_{value}" for value in rolling_windows])
    features.extend(
        [
            "trend_ratio",
            "impact_trend_ratio",
            "junction_share_84",
            "junction_offence_share_84",
            "main_road_share_84",
            "heavy_vehicle_share_84",
            "high_severity_share_84",
            "repeat_vehicle_rate_84",
            "active_devices_mean_84",
            "active_officers_mean_84",
            "record_confidence_mean_84",
            "history_observations",
            "zone_expanding_mean_count",
            "zone_expanding_mean_impact",
        ]
    )
    return features
