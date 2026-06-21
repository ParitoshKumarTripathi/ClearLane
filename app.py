from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clearlane.optimizer import optimize_patrol_plan  # noqa: E402

st.set_page_config(
    page_title="ClearLane — Parking intelligence",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --cl-blue: #155EEF;
  --cl-ink: #17212B;
  --cl-muted: #667085;
  --cl-border: #E4E7EC;
  --cl-surface: #FFFFFF;
  --cl-canvas: #F7F8FA;
}
.block-container {max-width: 1440px; padding-top: 3.25rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {border-right: 1px solid var(--cl-border);}
[data-testid="stMetric"] {
  background: var(--cl-surface);
  border: 1px solid var(--cl-border);
  border-radius: 14px;
  padding: 14px 16px;
}
.cl-brand {display:flex; align-items:center; gap:12px; margin: 0 0 6px 0; min-height: 40px;}
.cl-mark {
  width:34px; height:34px; border-radius:9px; background:var(--cl-blue);
  display:grid; place-items:center; color:white; font-weight:800; letter-spacing:0; line-height:1;
  flex: 0 0 34px;
}
.cl-title {font-size:1.65rem; font-weight:720; letter-spacing:0; line-height:1.2; color:var(--cl-ink);}
.cl-subtitle {color:var(--cl-muted); margin-top:0; margin-bottom:1.1rem;}
.cl-kicker {font-size:.76rem; font-weight:700; color:var(--cl-blue); text-transform:uppercase; letter-spacing:0;}
.cl-note {
  border:1px solid var(--cl-border); border-radius:12px; padding:12px 14px;
  background:var(--cl-surface); color:var(--cl-muted); font-size:.88rem;
}
.cl-help {
  border:1px solid var(--cl-border); border-radius:10px; padding:10px 12px;
  background:#F9FAFB; color:var(--cl-muted); font-size:.86rem; margin:.6rem 0 1rem 0;
}
.cl-help strong {color:var(--cl-ink);}
.cl-win {
  border:1px solid #B7E4C7; border-radius:10px; padding:10px 12px;
  background:#F1FBF5; color:#14532D; font-size:.9rem; margin:.65rem 0 1rem 0;
}
.cl-section {font-size:1.12rem; font-weight:700; color:var(--cl-ink); margin-top:.4rem;}
.cl-map-legend {display:flex; flex-wrap:wrap; gap:8px 14px; margin:.35rem 0 .25rem 0; color:var(--cl-muted); font-size:.84rem;}
.cl-map-legend span {display:inline-flex; align-items:center; gap:6px;}
.cl-swatch {width:10px; height:10px; border-radius:50%; display:inline-block;}
.stButton > button, .stDownloadButton > button {border-radius:10px; font-weight:650;}
[data-testid="stDataFrame"] {border:1px solid var(--cl-border); border-radius:12px; overflow:hidden;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    artifacts = ROOT / "artifacts"
    predictions_path = artifacts / "demo_predictions.csv.gz"
    history_path = artifacts / "zone_history.csv.gz"
    importance_path = artifacts / "feature_importance.csv"
    metrics_path = artifacts / "metrics.json"
    summary_path = artifacts / "data_summary.json"
    required = [predictions_path, history_path, importance_path, metrics_path, summary_path]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing generated artifacts: " + ", ".join(missing) + ". Run the pipeline first."
        )

    predictions = pd.read_csv(predictions_path)
    predictions["target_window_start"] = pd.to_datetime(
        predictions["target_window_start"], utc=True
    ).dt.tz_convert("Asia/Kolkata")
    history = pd.read_csv(history_path)
    history["window_start"] = pd.to_datetime(history["window_start"], utc=True).dt.tz_convert(
        "Asia/Kolkata"
    )
    importance = pd.read_csv(importance_path)
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return predictions, history, importance, metrics, summary


def format_window(timestamp: pd.Timestamp) -> str:
    end = timestamp + pd.Timedelta(hours=2)
    return f"{timestamp:%a, %d %b · %I:%M %p}–{end:%I:%M %p}"


def map_color(score: int) -> list[int]:
    if score >= 95:
        return [127, 29, 29, 230]
    if score >= 85:
        return [220, 38, 38, 220]
    if score >= 70:
        return [234, 88, 12, 210]
    if score >= 50:
        return [217, 119, 6, 195]
    return [37, 99, 235, 170]


def render_map(frame: pd.DataFrame, show_rank_labels: bool = False) -> None:
    map_frame = frame.copy()
    map_frame["color"] = map_frame["impact_score"].map(map_color)
    map_frame["radius"] = 80 + np.sqrt(map_frame["predicted_impact_units"].clip(lower=0)) * 32
    if show_rank_labels:
        ranks = map_frame["rank"] if "rank" in map_frame.columns else pd.Series(
            range(1, len(map_frame) + 1), index=map_frame.index
        )
        map_frame["map_label"] = ranks.astype(int).astype(str)
    center_lat = float(map_frame["centroid_latitude"].mean())
    center_lon = float(map_frame["centroid_longitude"].mean())
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=map_frame,
            get_position="[centroid_longitude, centroid_latitude]",
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=6,
            radius_max_pixels=26,
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255, 180],
            line_width_min_pixels=1,
        )
    ]
    if show_rank_labels:
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=map_frame,
                get_position="[centroid_longitude, centroid_latitude]",
                get_text="map_label",
                get_size=14,
                get_color=[255, 255, 255, 240],
                get_alignment_baseline="'center'",
                get_text_anchor="'middle'",
                pickable=False,
            )
        )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=10.5,
            pitch=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": "<b>{zone_name}</b><br/>"
            "Station: {police_station}<br/>"
            "Obstruction risk: {impact_score}/100 ({risk_level})<br/>"
            "Expected obstruction: {predicted_impact_units}<br/>"
            "Expected violations: {predicted_violation_count}<br/>"
            "Reason: {top_reason}",
            "style": {"backgroundColor": "#17212B", "color": "white"},
        },
    )
    st.pydeck_chart(deck, width="stretch", height=540)


st.markdown(
    """
<div class="cl-brand">
  <div class="cl-mark">CL</div>
  <div class="cl-title">ClearLane</div>
</div>
<p class="cl-subtitle">Predict where illegal parking will obstruct traffic and direct patrols where they matter most.</p>
""",
    unsafe_allow_html=True,
)

try:
    predictions, history, importance, metrics, summary = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.code(
        'PYTHONPATH=src python -m clearlane.cli pipeline --input "data/raw/police_violations.csv"',
        language="bash",
    )
    st.stop()

available_windows = sorted(predictions["target_window_start"].drop_duplicates().tolist())
default_value = pd.Timestamp(summary.get("default_demo_window"))
if default_value.tzinfo is None:
    default_value = default_value.tz_localize("Asia/Kolkata")
else:
    default_value = default_value.tz_convert("Asia/Kolkata")
default_index = min(
    range(len(available_windows)),
    key=lambda index: abs((available_windows[index] - default_value).total_seconds()),
)

with st.sidebar:
    st.markdown("### Patrol controls")
    selected_window = st.selectbox(
        "Forecast window",
        available_windows,
        index=default_index,
        format_func=format_window,
    )
    stations = ["All stations"] + sorted(predictions["police_station"].dropna().unique().tolist())
    selected_station = st.selectbox("Police station", stations)
    teams = st.slider("Teams available", min_value=1, max_value=20, value=8)
    map_focus = st.radio(
        "Map focus",
        ["Assigned stops", "Top 50 risk zones", "All selected zones"],
    )
    risk_levels = st.multiselect(
        "Show risk levels",
        ["Critical", "High", "Moderate", "Low"],
        default=["Critical", "High"],
    )
    st.divider()
    st.caption("Historical simulation · April 2024 holdout")
    st.caption(f"{summary['grid_size_meters']} m zones · {summary['window_hours']}-hour windows")

window_frame = predictions.loc[predictions["target_window_start"].eq(selected_window)].copy()
if selected_station != "All stations":
    window_frame = window_frame.loc[window_frame["police_station"].eq(selected_station)].copy()

if window_frame.empty:
    st.warning("No model zones are available for the selected station and time window.")
    st.stop()

plan = optimize_patrol_plan(
    window_frame,
    teams=teams,
    minimum_distance_meters=400,
    police_station=None,
)

operations_tab, performance_tab, method_tab = st.tabs(
    ["Operations", "Model performance", "Methodology"]
)

with operations_tab:
    predicted_violations = float(window_frame["predicted_violation_count"].sum())
    predicted_impact = float(window_frame["predicted_impact_units"].sum())
    critical_zones = int(window_frame["risk_level"].eq("Critical").sum())
    plan_capture = (
        float(plan["predicted_impact_units"].sum()) / predicted_impact if predicted_impact > 0 else 0.0
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Expected violations",
        f"{predicted_violations:,.0f}",
        help="Forecasted parking violation records in the selected patrol window.",
    )
    metric_columns[1].metric(
        "Expected obstruction",
        f"{predicted_impact:,.1f} units",
        help="Relative obstruction estimate. It is used for ranking, not claimed as measured traffic delay.",
    )
    metric_columns[2].metric(
        "Critical zones",
        f"{critical_zones}",
        help="Zones with obstruction risk score of 80 or above in this patrol window.",
    )
    metric_columns[3].metric(
        "Covered by teams",
        f"{plan_capture:.0%}",
        help="Share of expected obstruction covered by the current team allocation.",
    )
    st.markdown(
        '<div class="cl-help"><strong>Critical zone:</strong> obstruction risk score 80 or above. '
        '<strong>Obstruction units:</strong> a relative estimate based on offence severity, vehicle type, '
        "road context and peak-hour timing.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="cl-section">Priority map</div>', unsafe_allow_html=True)
        if map_focus == "Assigned stops":
            visible_map = plan.loc[plan["risk_level"].isin(risk_levels)].copy()
            show_rank_labels = True
        elif map_focus == "Top 50 risk zones":
            visible_map = window_frame.loc[window_frame["risk_level"].isin(risk_levels)].copy()
            visible_map = visible_map.nlargest(50, "priority_score")
            show_rank_labels = False
        else:
            visible_map = window_frame.loc[window_frame["risk_level"].isin(risk_levels)].copy()
            visible_map = visible_map.nlargest(250, "priority_score")
            show_rank_labels = False
        map_limit = len(visible_map)
        if visible_map.empty:
            st.info("Select at least one risk level with available zones.")
        else:
            render_map(visible_map, show_rank_labels=show_rank_labels)
            st.markdown(
                """
<div class="cl-map-legend">
  <span><i class="cl-swatch" style="background:#7F1D1D"></i>Risk 95-100</span>
  <span><i class="cl-swatch" style="background:#DC2626"></i>85-94</span>
  <span><i class="cl-swatch" style="background:#EA580C"></i>70-84</span>
  <span><i class="cl-swatch" style="background:#D97706"></i>50-69</span>
  <span><i class="cl-swatch" style="background:#2563EB"></i>&lt;50</span>
</div>
""",
                unsafe_allow_html=True,
            )
        if map_focus == "Assigned stops":
            st.caption(
                f"Showing {map_limit:,} assigned stop(s). Numbers match dispatch rank; larger dots mean more expected obstruction."
            )
        else:
            st.caption(
                f"Showing {map_limit:,} zone(s). Larger dots mean more expected obstruction."
            )

    with right:
        st.markdown('<div class="cl-section">Team dispatch</div>', unsafe_allow_html=True)
        display_plan = plan[
            [
                "team",
                "zone_name",
                "police_station",
                "impact_score",
                "model_confidence",
                "predicted_violation_count",
                "top_reason",
            ]
        ].copy()
        display_plan["predicted_violation_count"] = display_plan[
            "predicted_violation_count"
        ].round(1)
        display_plan["model_confidence"] = (display_plan["model_confidence"] * 100).round(0)
        st.dataframe(
            display_plan,
            hide_index=True,
            width="stretch",
            height=470,
            column_config={
                "team": "Team",
                "zone_name": "Location",
                "police_station": "Station",
                "impact_score": st.column_config.ProgressColumn(
                    "Risk score", min_value=0, max_value=100, format="%d"
                ),
                "model_confidence": st.column_config.ProgressColumn(
                    "Evidence", min_value=0, max_value=100, format="%d%%"
                ),
                "predicted_violation_count": "Expected violations",
                "top_reason": "Why prioritize",
            },
        )
        st.download_button(
            "Download dispatch list",
            plan.to_csv(index=False).encode("utf-8"),
            file_name=f"clearlane_patrol_plan_{selected_window:%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
        )

    st.divider()
    st.markdown('<div class="cl-section">Zone detail</div>', unsafe_allow_html=True)
    selected_zone = st.selectbox(
        "Inspect an assigned stop",
        plan["zone_id"].tolist(),
        format_func=lambda zone: plan.set_index("zone_id").loc[zone, "zone_name"],
    )
    detail = plan.loc[plan["zone_id"].eq(selected_zone)].iloc[0]
    detail_columns = st.columns([1, 1, 2])
    detail_columns[0].metric(
        "Obstruction risk",
        f"{int(detail['impact_score'])}/100",
        help="0-100 rank for expected obstruction in this patrol window.",
    )
    detail_columns[1].metric(
        "Evidence strength",
        f"{float(detail['model_confidence']):.0%}",
        help="How much recent and historical evidence supports this recommendation.",
    )
    detail_columns[2].markdown(
        f"**Action**  \n{detail['recommended_action']}  \n\n"
        f"**Why this stop**  \n{detail['top_reason']}"
    )

    zone_history = history.loc[history["zone_id"].eq(selected_zone)].copy()
    if not zone_history.empty:
        daily_history = (
            zone_history.set_index("window_start")[["violation_count", "impact_units"]]
            .resample("1D")
            .sum()
            .rename(columns={"violation_count": "Violations", "impact_units": "Obstruction units"})
        )
        st.line_chart(daily_history, width="stretch", height=260)

with performance_tab:
    st.markdown("### Holdout validation")
    st.caption(
        "Models are trained only on records before 1 April 2024 and evaluated as rolling one-step-ahead "
        "forecasts on the April holdout period."
    )
    clearlane = metrics["clearlane"]
    baseline = metrics["same_window_last_week"]
    seven_day = metrics["seven_day_average"]
    count_mae_gain = (
        (baseline["count"]["mae"] - clearlane["count"]["mae"]) / baseline["count"]["mae"]
    )
    impact_mae_gain = (
        (baseline["impact"]["mae"] - clearlane["impact"]["mae"]) / baseline["impact"]["mae"]
    )
    impact_capture_10_gain = (
        (clearlane["impact"]["weighted_capture_at_10"] - baseline["impact"]["weighted_capture_at_10"])
        / baseline["impact"]["weighted_capture_at_10"]
    )
    impact_capture_25_gain = (
        (clearlane["impact"]["weighted_capture_at_25"] - baseline["impact"]["weighted_capture_at_25"])
        / baseline["impact"]["weighted_capture_at_25"]
    )
    cols = st.columns(4)
    cols[0].metric(
        "Count MAE",
        f"{clearlane['count']['mae']:.2f}",
        delta=f"{count_mae_gain:.0%} lower than last week",
        help="Mean absolute error for violation counts. Lower is better.",
    )
    cols[1].metric(
        "Impact MAE",
        f"{clearlane['impact']['mae']:.2f}",
        delta=f"{impact_mae_gain:.0%} lower than last week",
        help="Mean absolute error for obstruction units. Lower is better.",
    )
    cols[2].metric(
        "Impact capture@10",
        f"{clearlane['impact']['weighted_capture_at_10']:.1%}",
        delta=f"{impact_capture_10_gain:.0%} higher",
        help="Share of actual obstruction found in the top 10 recommended zones. Higher is better.",
    )
    cols[3].metric(
        "Impact capture@25",
        f"{clearlane['impact']['weighted_capture_at_25']:.1%}",
        delta=f"{impact_capture_25_gain:.0%} higher",
        help="Share of actual obstruction found in the top 25 recommended zones. Higher is better.",
    )
    st.markdown(
        f'<div class="cl-win"><strong>Validation result:</strong> ClearLane reduces impact error by '
        f"{impact_mae_gain:.0%} versus the same-window-last-week baseline and captures "
        f"{impact_capture_10_gain:.0%} more actual obstruction in the top 10 zones.</div>",
        unsafe_allow_html=True,
    )

    comparison = pd.DataFrame(
        [
            {
                "System": "ClearLane ML",
                "Count MAE ↓": clearlane["count"]["mae"],
                "Impact MAE ↓": clearlane["impact"]["mae"],
                "Impact capture@10 ↑": clearlane["impact"]["weighted_capture_at_10"] * 100,
                "Impact capture@25 ↑": clearlane["impact"]["weighted_capture_at_25"] * 100,
            },
            {
                "System": "Same window last week",
                "Count MAE ↓": baseline["count"]["mae"],
                "Impact MAE ↓": baseline["impact"]["mae"],
                "Impact capture@10 ↑": baseline["impact"]["weighted_capture_at_10"] * 100,
                "Impact capture@25 ↑": baseline["impact"]["weighted_capture_at_25"] * 100,
            },
            {
                "System": "Seven-day average",
                "Count MAE ↓": seven_day["count"]["mae"],
                "Impact MAE ↓": seven_day["impact"]["mae"],
                "Impact capture@10 ↑": seven_day["impact"]["weighted_capture_at_10"] * 100,
                "Impact capture@25 ↑": seven_day["impact"]["weighted_capture_at_25"] * 100,
            },
        ]
    )
    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        column_config={
            "Count MAE ↓": st.column_config.NumberColumn(format="%.2f"),
            "Impact MAE ↓": st.column_config.NumberColumn(format="%.2f"),
            "Impact capture@10 ↑": st.column_config.NumberColumn(format="%.1f%%"),
            "Impact capture@25 ↑": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption("Arrows show what better means: lower error, higher obstruction capture.")

    st.markdown("### What drives the predictions")
    model_choice = st.segmented_control(
        "Model",
        options=["hotspot", "impact"],
        default="impact",
        format_func=lambda value: "Impact model" if value == "impact" else "Hotspot model",
    )
    selected_importance = (
        importance.loc[importance["model"].eq(model_choice)]
        .nlargest(12, "importance")
        .set_index("feature")[["importance"]]
    )
    st.bar_chart(selected_importance, horizontal=True, width="stretch", height=380)

with method_tab:
    st.markdown("### How ClearLane decides")
    st.markdown(
        """
**Where violations may happen**  
ClearLane forecasts expected parking violations for each 500 m zone in the next two-hour patrol window.

**Where they may block traffic**  
Obstruction units are a relative risk score based on offence severity, vehicle type, junction or main-road context, and peak-hour timing.

**Where teams should go first**  
The dispatch list picks high-risk zones, respects the number of teams available, and avoids sending multiple teams to nearby duplicate stops.
"""
    )
    st.markdown(
        '<div class="cl-note"><strong>Interpretation:</strong> The dataset contains enforcement records, '
        "not measured vehicle speeds or queue lengths. ClearLane therefore reports relative traffic-obstruction "
        "risk, not an unsupported percentage reduction in traffic speed.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Data and model facts")
    facts = pd.DataFrame(
        {
            "Item": [
                "Clean parking records",
                "Active model zones",
                "Grid resolution",
                "Forecast horizon",
                "Training rows",
                "Evaluation windows",
            ],
            "Value": [
                f"{summary['clean_rows']:,}",
                f"{summary['active_model_zones']:,}",
                f"{summary['grid_size_meters']} metres",
                f"{summary['window_hours']} hours",
                f"{summary['training_rows']:,}",
                f"{summary['test_windows']:,}",
            ],
        }
    )
    st.dataframe(facts, hide_index=True, width="stretch")
