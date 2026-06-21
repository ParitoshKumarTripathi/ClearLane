from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

METERS_PER_DEGREE_LATITUDE = 111_320.0


@dataclass(frozen=True)
class GridSpec:
    origin_latitude: float
    origin_longitude: float
    grid_size_meters: int

    @property
    def meters_per_degree_longitude(self) -> float:
        return METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(self.origin_latitude))


def assign_grid(
    latitude: pd.Series,
    longitude: pd.Series,
    spec: GridSpec,
) -> pd.DataFrame:
    cell_x = np.floor(
        (longitude.astype(float) - spec.origin_longitude)
        * spec.meters_per_degree_longitude
        / spec.grid_size_meters
    ).astype("int32")
    cell_y = np.floor(
        (latitude.astype(float) - spec.origin_latitude)
        * METERS_PER_DEGREE_LATITUDE
        / spec.grid_size_meters
    ).astype("int32")
    zone_id = "Z" + cell_x.astype(str) + "_" + cell_y.astype(str)
    return pd.DataFrame({"cell_x": cell_x, "cell_y": cell_y, "zone_id": zone_id})


def grid_centroid(cell_x: pd.Series, cell_y: pd.Series, spec: GridSpec) -> pd.DataFrame:
    longitude = spec.origin_longitude + (
        (cell_x.astype(float) + 0.5) * spec.grid_size_meters / spec.meters_per_degree_longitude
    )
    latitude = spec.origin_latitude + (
        (cell_y.astype(float) + 0.5) * spec.grid_size_meters / METERS_PER_DEGREE_LATITUDE
    )
    return pd.DataFrame({"centroid_latitude": latitude, "centroid_longitude": longitude})


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))
