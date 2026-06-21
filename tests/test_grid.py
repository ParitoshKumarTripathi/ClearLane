import pandas as pd

from clearlane.grid import GridSpec, assign_grid, grid_centroid, haversine_meters


def test_grid_assignment_is_deterministic() -> None:
    spec = GridSpec(12.97, 77.59, 500)
    result = assign_grid(pd.Series([12.971]), pd.Series([77.591]), spec)
    repeated = assign_grid(pd.Series([12.971]), pd.Series([77.591]), spec)
    assert result.loc[0, "zone_id"] == repeated.loc[0, "zone_id"]


def test_centroid_returns_valid_coordinates() -> None:
    spec = GridSpec(12.97, 77.59, 500)
    centroid = grid_centroid(pd.Series([0]), pd.Series([0]), spec)
    assert 12.97 < centroid.loc[0, "centroid_latitude"] < 12.98
    assert 77.59 < centroid.loc[0, "centroid_longitude"] < 77.60


def test_haversine_zero_distance() -> None:
    assert haversine_meters(12.97, 77.59, 12.97, 77.59) == 0
