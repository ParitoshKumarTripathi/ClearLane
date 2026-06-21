# Limitations and safeguards

## Traffic impact is a proxy

The source data contains violations, not road speed, queue length or lane capacity. ClearLane reports relative obstruction units and an impact percentile. It does not claim a measured speed reduction.

**Future validation:** join probe-speed, signal, lane-count or travel-time data and calibrate obstruction units against observed flow degradation.

## Enforcement bias

Observed violations depend on both illegal parking and where personnel already patrol.

**Current mitigation:** active-device, active-officer and record-confidence histories are available to the model and dashboard analysis. Rejected and duplicate records are excluded.

**Future mitigation:** normalize by patrol exposure hours and run controlled enforcement experiments.

## No post-intervention outcome

The source `action_taken_timestamp` and `closed_datetime` fields are empty. The project cannot establish causal enforcement effectiveness.

**Future validation:** record arrival time, vehicles cleared, tow actions, lane recovery time and post-enforcement speed.

## Historical simulation

The deployed demo replays April holdout forecasts. A production deployment requires an incremental event feed and a scheduled feature update.

## Spatial approximation

A 500 m grid is a practical hackathon abstraction. It does not follow precise road geometry and can combine parallel roads.

**Future improvement:** map-match coordinates to road segments and intersections using an authoritative road network.
