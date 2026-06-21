from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clearlane.grid import GridSpec, assign_grid
from clearlane.impact import (
    PARKING_OFFENCE_WEIGHTS,
    is_peak_hour,
    vehicle_factor,
)

NULL_VALUES = ["NULL", "null", "None", "NONE", "nan", "NaN", ""]
REQUIRED_COLUMNS = [
    "id",
    "latitude",
    "longitude",
    "location",
    "vehicle_number",
    "vehicle_type",
    "violation_type",
    "created_datetime",
    "device_id",
    "created_by_id",
    "center_code",
    "police_station",
    "junction_name",
    "updated_vehicle_number",
    "updated_vehicle_type",
    "validation_status",
]


def _canonical_text(primary: pd.Series, fallback: pd.Series, default: str = "Unknown") -> pd.Series:
    value = primary.fillna("").astype(str).str.strip()
    fallback_value = fallback.fillna("").astype(str).str.strip()
    result = value.where(~value.str.upper().isin({"", "NULL", "NAN", "NONE"}), fallback_value)
    return result.where(~result.str.upper().isin({"", "NULL", "NAN", "NONE"}), default)


def load_and_clean_records(
    input_path: str | Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    raw = pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
        na_values=NULL_VALUES,
        keep_default_na=True,
        low_memory=False,
    )
    original_rows = len(raw)
    raw = raw.drop_duplicates(subset="id", keep="last").copy()

    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    lat_min, lat_max = config["data"]["valid_latitude"]
    lon_min, lon_max = config["data"]["valid_longitude"]
    coordinate_mask = raw["latitude"].between(lat_min, lat_max) & raw["longitude"].between(
        lon_min, lon_max
    )
    raw = raw.loc[coordinate_mask].copy()

    raw["validation_status"] = (
        raw["validation_status"].fillna("unreviewed").astype(str).str.strip().str.lower()
    )
    policy = config["data"].get("validation_policy", "exclude_rejected_duplicate")
    if policy == "approved_only":
        raw = raw.loc[raw["validation_status"].eq("approved")].copy()
    elif policy == "exclude_rejected_duplicate":
        raw = raw.loc[~raw["validation_status"].isin({"rejected", "duplicate"})].copy()
    else:
        raise ValueError(f"Unknown validation policy: {policy}")

    raw["created_at"] = pd.to_datetime(raw["created_datetime"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["created_at"]).copy()
    raw["created_at"] = raw["created_at"].dt.tz_convert(config["data"]["timezone"])

    raw["canonical_vehicle_type"] = _canonical_text(
        raw["updated_vehicle_type"], raw["vehicle_type"]
    ).str.upper()
    raw["canonical_vehicle_number"] = _canonical_text(
        raw["updated_vehicle_number"], raw["vehicle_number"], default="Unknown vehicle"
    )

    # The source stores JSON-like offence lists. Exact vectorized token matching is
    # much faster than parsing hundreds of thousands of rows in Python and is safe
    # because the competition categories are fixed and known.
    violation_text = raw["violation_type"].fillna("").astype(str).str.upper()
    offence_flags = pd.DataFrame(
        {
            offence: violation_text.str.contains(offence, regex=False).astype("int8")
            for offence in PARKING_OFFENCE_WEIGHTS
        },
        index=raw.index,
    )
    raw = raw.loc[offence_flags.sum(axis=1).gt(0)].copy()
    offence_flags = offence_flags.loc[raw.index]

    impact_config = config["impact"]
    weights = pd.Series(PARKING_OFFENCE_WEIGHTS, dtype=float)
    weighted = offence_flags.mul(weights, axis=1)
    maximum_weight = weighted.max(axis=1)
    raw["offence_severity"] = maximum_weight + float(
        impact_config["additional_offence_fraction"]
    ) * (weighted.sum(axis=1) - maximum_weight)
    raw["vehicle_factor"] = raw["canonical_vehicle_type"].map(vehicle_factor)

    location_text = raw["location"].fillna("").astype(str)
    junction_text = raw["junction_name"].fillna("").astype(str).str.strip()
    raw["is_named_junction"] = (~junction_text.str.upper().isin(
        {"", "NULL", "NO JUNCTION", "NAN", "NONE"}
    )).astype("int8")
    raw["is_main_road"] = location_text.str.contains(
        r"\b(?:main road|highway|ring road|flyover|service road|arterial)\b",
        case=False,
        regex=True,
    ).astype("int8")
    raw["is_sensitive_place"] = location_text.str.contains(
        r"\b(?:metro|market|bus stand|bus stop|school|hospital|mall|station|terminal)\b",
        case=False,
        regex=True,
    ).astype("int8")
    raw["location_factor"] = np.maximum.reduce(
        [
            np.ones(len(raw), dtype=float),
            np.where(
                raw["is_named_junction"].eq(1),
                float(impact_config["named_junction_factor"]),
                1.0,
            ),
            np.where(
                raw["is_main_road"].eq(1),
                float(impact_config["main_road_factor"]),
                1.0,
            ),
            np.where(
                raw["is_sensitive_place"].eq(1),
                float(impact_config["sensitive_place_factor"]),
                1.0,
            ),
        ]
    )
    raw["is_peak_hour"] = raw["created_at"].dt.hour.map(is_peak_hour).astype("int8")
    raw["peak_factor"] = np.where(
        raw["is_peak_hour"].eq(1), float(impact_config["peak_hour_factor"]), 1.0
    )
    raw["obstruction_units"] = (
        raw["offence_severity"]
        * raw["vehicle_factor"]
        * raw["location_factor"]
        * raw["peak_factor"]
    )

    raw["is_high_severity"] = raw["offence_severity"].ge(4.0).astype("int8")
    raw["is_heavy_vehicle"] = raw["vehicle_factor"].ge(1.4).astype("int8")
    raw["has_main_road_offence"] = offence_flags["PARKING IN A MAIN ROAD"].astype("int8")
    raw["has_junction_offence"] = (
        offence_flags["PARKING NEAR ROAD CROSSING"]
        | offence_flags["PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS"]
    ).astype("int8")
    confidence_map = {
        "approved": 1.0,
        "unreviewed": 0.75,
        "created1": 0.65,
        "processing": 0.65,
    }
    raw["record_confidence"] = raw["validation_status"].map(confidence_map).fillna(0.70)

    spatial = config["spatial"]
    spec = GridSpec(
        origin_latitude=float(spatial["origin_latitude"]),
        origin_longitude=float(spatial["origin_longitude"]),
        grid_size_meters=int(spatial["grid_size_meters"]),
    )
    grid = assign_grid(raw["latitude"], raw["longitude"], spec)
    raw = pd.concat([raw.reset_index(drop=True), grid.reset_index(drop=True)], axis=1)

    window_hours = int(config["temporal"]["window_hours"])
    raw["window_start"] = raw["created_at"].dt.floor(f"{window_hours}h")
    raw["police_station"] = raw["police_station"].fillna("Unknown").astype(str).str.strip()
    raw["junction_name"] = raw["junction_name"].fillna("No Junction").astype(str).str.strip()
    raw["location"] = raw["location"].fillna("Location unavailable").astype(str).str.strip()
    raw["device_id"] = raw["device_id"].fillna("Unknown device").astype(str)
    raw["created_by_id"] = raw["created_by_id"].fillna("Unknown officer").astype(str)

    summary = {
        "source_rows": int(original_rows),
        "clean_rows": int(len(raw)),
        "removed_rows": int(original_rows - len(raw)),
        "date_min": raw["created_at"].min().isoformat(),
        "date_max": raw["created_at"].max().isoformat(),
        "zones_before_threshold": int(raw["zone_id"].nunique()),
        "police_stations": int(raw["police_station"].nunique()),
        "named_junctions": int(raw.loc[raw["is_named_junction"].eq(1), "junction_name"].nunique()),
        "parking_offence_categories": int(len(PARKING_OFFENCE_WEIGHTS)),
        "validation_policy": policy,
    }
    return raw, summary
