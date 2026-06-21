import pandas as pd

from clearlane.optimizer import optimize_patrol_plan


def test_optimizer_returns_requested_number_of_teams() -> None:
    frame = pd.DataFrame(
        {
            "zone_id": ["A", "B", "C"],
            "zone_name": ["A road", "B road", "C road"],
            "police_station": ["Station", "Station", "Station"],
            "centroid_latitude": [12.97, 12.98, 12.99],
            "centroid_longitude": [77.59, 77.60, 77.61],
            "target_window_start": pd.to_datetime(
                ["2024-04-01T08:00:00+05:30"] * 3
            ),
            "priority_score": [90, 80, 70],
            "predicted_impact_units": [20.0, 15.0, 10.0],
            "predicted_violation_count": [8.0, 6.0, 4.0],
            "impact_score": [95, 85, 75],
            "risk_level": ["Critical", "Critical", "High"],
            "model_confidence": [0.9, 0.8, 0.7],
            "top_reason": [
                "Main-road parking pattern",
                "Recurring hotspot",
                "High recent violation volume",
            ],
        }
    )
    result = optimize_patrol_plan(frame, teams=2, minimum_distance_meters=100)
    assert len(result) == 2
    assert result["zone_id"].is_unique
