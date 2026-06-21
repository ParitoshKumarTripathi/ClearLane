from __future__ import annotations

import numpy as np
import pandas as pd

from clearlane.grid import haversine_meters
from clearlane.impact import dominant_reason


def add_operational_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["impact_percentile"] = frame.groupby("target_window_start")[
        "predicted_impact_units"
    ].rank(pct=True, method="average")
    frame["impact_score"] = (100 * frame["impact_percentile"]).round().clip(0, 100).astype(int)

    by_window = frame.groupby("target_window_start", sort=False)
    history = frame["history_observations"].clip(lower=0)
    history_cap = by_window["history_observations"].transform(
        lambda values: max(float(values.max()), 1.0)
    )
    history_evidence = (np.log1p(history) / np.log1p(history_cap)).clip(0.0, 1.0)

    recent_activity = frame.get(
        "count_roll_mean_84", pd.Series(0.0, index=frame.index)
    ).clip(lower=0)
    recent_cap = by_window["count_roll_mean_84"].transform(
        lambda values: max(float(values.max()), 0.1)
    ) if "count_roll_mean_84" in frame.columns else pd.Series(0.1, index=frame.index)
    recent_evidence = (np.log1p(recent_activity) / np.log1p(recent_cap)).clip(0.0, 1.0)

    trend_ratio = frame.get("trend_ratio", pd.Series(1.0, index=frame.index)).clip(lower=0.05)
    trend_stability = (
        1.0 - (np.abs(np.log(trend_ratio)) / np.log(3.0)).clip(0.0, 1.0) * 0.30
    ).clip(0.75, 1.0)

    frame["model_confidence"] = (
        0.34
        + 0.42 * history_evidence
        + 0.18 * recent_evidence
        + 0.06 * trend_stability
    ).clip(0.35, 0.94)
    frame["priority_score"] = (
        frame["impact_score"] * frame["model_confidence"]
    ).round(1)
    frame["risk_level"] = pd.cut(
        frame["impact_score"],
        bins=[-1, 29, 59, 79, 100],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)
    frame["top_reason"] = [dominant_reason(row) for row in frame.itertuples(index=False)]
    return frame


def _action_for_reason(reason: str) -> str:
    actions = {
        "Junction obstruction pattern": "Clear the junction approaches and monitor turning lanes",
        "Main-road parking pattern": "Prioritize carriageway clearance and tow-away readiness",
        "Large-vehicle obstruction": "Deploy a heavy-vehicle capable enforcement unit",
        "Recurring hotspot": "Schedule a visible repeat patrol during the full window",
        "High recent violation volume": "Deploy an enforcement team before the expected peak",
    }
    return actions.get(reason, "Deploy a targeted parking-enforcement patrol")


def optimize_patrol_plan(
    predictions: pd.DataFrame,
    teams: int,
    minimum_distance_meters: float,
    police_station: str | None = None,
) -> pd.DataFrame:
    if teams < 1:
        raise ValueError("teams must be at least 1")
    candidates = predictions.copy()
    if police_station and police_station != "All stations":
        candidates = candidates.loc[candidates["police_station"].eq(police_station)].copy()
    candidates = candidates.sort_values(
        ["priority_score", "predicted_impact_units"], ascending=False
    )
    if candidates.empty:
        return candidates

    selected_indices: list[int] = []
    for index, row in candidates.iterrows():
        sufficiently_far = all(
            haversine_meters(
                float(row["centroid_latitude"]),
                float(row["centroid_longitude"]),
                float(candidates.loc[chosen, "centroid_latitude"]),
                float(candidates.loc[chosen, "centroid_longitude"]),
            )
            >= minimum_distance_meters
            for chosen in selected_indices
        )
        if sufficiently_far:
            selected_indices.append(index)
        if len(selected_indices) >= teams:
            break

    if len(selected_indices) < teams:
        for index in candidates.index:
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= teams:
                break

    plan = candidates.loc[selected_indices].copy().reset_index(drop=True)
    plan.insert(0, "team", [f"Team {number}" for number in range(1, len(plan) + 1)])
    plan.insert(1, "rank", range(1, len(plan) + 1))
    start = pd.to_datetime(plan["target_window_start"])
    if getattr(start.dt, "tz", None) is None:
        end = start + pd.Timedelta(hours=2)
    else:
        end = start + pd.Timedelta(hours=2)
    plan["recommended_window"] = start.dt.strftime("%d %b, %I:%M %p") + "–" + end.dt.strftime(
        "%I:%M %p"
    )
    plan["recommended_action"] = plan["top_reason"].map(_action_for_reason)
    columns = [
        "team",
        "rank",
        "zone_id",
        "zone_name",
        "police_station",
        "centroid_latitude",
        "centroid_longitude",
        "recommended_window",
        "predicted_violation_count",
        "predicted_impact_units",
        "impact_score",
        "risk_level",
        "model_confidence",
        "top_reason",
        "recommended_action",
    ]
    return plan[columns]
