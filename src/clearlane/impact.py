from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable

import numpy as np

PARKING_OFFENCE_WEIGHTS: dict[str, float] = {
    "DOUBLE PARKING": 5.0,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 5.0,
    "PARKING NEAR ROAD CROSSING": 5.0,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 4.0,
    "PARKING IN A MAIN ROAD": 4.0,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 3.0,
    "PARKING OTHER THAN BUS STOP": 3.0,
    "PARKING ON FOOTPATH": 2.0,
    "NO PARKING": 1.5,
    "WRONG PARKING": 1.0,
}

VEHICLE_FACTORS: dict[str, float] = {
    "HGV": 1.50,
    "LORRY/GOODS VEHICLE": 1.50,
    "TANKER": 1.50,
    "PRIVATE BUS": 1.50,
    "TOURIST BUS": 1.50,
    "FACTORY BUS": 1.50,
    "BUS (BMTC/KSRTC)": 1.50,
    "SCHOOL VEHICLE": 1.45,
    "MINI LORRY": 1.40,
    "TRACTOR": 1.40,
    "LGV": 1.25,
    "TEMPO": 1.25,
    "VAN": 1.25,
    "MAXI-CAB": 1.20,
    "GOODS AUTO": 1.15,
    "CAR": 1.00,
    "JEEP": 1.00,
    "PASSENGER AUTO": 0.85,
    "MOTOR CYCLE": 0.60,
    "SCOOTER": 0.60,
    "MOPED": 0.55,
    "OTHERS": 1.00,
}

MAIN_ROAD_PATTERN = re.compile(r"\b(main road|highway|ring road|flyover|service road|arterial)\b", re.I)
SENSITIVE_PLACE_PATTERN = re.compile(
    r"\b(metro|market|bus stand|bus stop|school|hospital|mall|station|terminal)\b", re.I
)


def normalize_offence(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().upper())


def parse_violation_types(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        parsed: Iterable[object] = value
    else:
        text = str(value).strip()
        if not text or text.upper() in {"NULL", "NAN", "NONE"}:
            return []
        parsed = []
        for parser in (json.loads, ast.literal_eval):
            try:
                candidate = parser(text)
                parsed = candidate if isinstance(candidate, list) else [candidate]
                break
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
        if not parsed:
            parsed = [segment for segment in re.split(r"[,;|]", text) if segment.strip()]
    return [normalize_offence(item) for item in parsed if str(item).strip()]


def parking_offences(offences: Iterable[str]) -> list[str]:
    return [offence for offence in offences if offence in PARKING_OFFENCE_WEIGHTS]


def offence_severity(offences: Iterable[str], additional_fraction: float = 0.25) -> float:
    scores = sorted(
        (PARKING_OFFENCE_WEIGHTS[item] for item in offences if item in PARKING_OFFENCE_WEIGHTS),
        reverse=True,
    )
    if not scores:
        return 0.0
    return float(scores[0] + additional_fraction * sum(scores[1:]))


def vehicle_factor(vehicle_type: object) -> float:
    key = str(vehicle_type).strip().upper()
    return VEHICLE_FACTORS.get(key, 1.0)


def location_flags(location: object, junction_name: object) -> dict[str, int]:
    location_text = "" if location is None else str(location)
    junction_text = "" if junction_name is None else str(junction_name).strip()
    named_junction = bool(junction_text) and junction_text.upper() not in {
        "NULL",
        "NO JUNCTION",
        "NAN",
        "NONE",
    }
    return {
        "is_named_junction": int(named_junction),
        "is_main_road": int(bool(MAIN_ROAD_PATTERN.search(location_text))),
        "is_sensitive_place": int(bool(SENSITIVE_PLACE_PATTERN.search(location_text))),
    }


def location_factor(
    is_named_junction: int,
    is_main_road: int,
    is_sensitive_place: int,
    named_junction_factor: float,
    main_road_factor: float,
    sensitive_place_factor: float,
) -> float:
    factors = [1.0]
    if is_named_junction:
        factors.append(named_junction_factor)
    if is_main_road:
        factors.append(main_road_factor)
    if is_sensitive_place:
        factors.append(sensitive_place_factor)
    return max(factors)


def is_peak_hour(hour: int) -> int:
    return int(7 <= hour < 11 or 16 <= hour < 20)


def dominant_reason(row: object) -> str:
    candidates = {
        "Recurring hotspot": float(getattr(row, "trend_ratio", 0.0) or 0.0),
        "Junction obstruction pattern": float(getattr(row, "junction_share_84", 0.0) or 0.0) * 2,
        "Main-road parking pattern": float(getattr(row, "main_road_share_84", 0.0) or 0.0) * 2,
        "Large-vehicle obstruction": float(getattr(row, "heavy_vehicle_share_84", 0.0) or 0.0) * 2,
        "High recent violation volume": float(getattr(row, "count_roll_mean_84", 0.0) or 0.0),
    }
    return max(candidates, key=candidates.get)
