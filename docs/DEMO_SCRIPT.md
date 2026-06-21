# Three-minute demo script

## 0:00–0:25 — Problem

“Parking heatmaps tell enforcement where violations happened. ClearLane predicts where illegal parking will obstruct traffic in the next patrol window and creates a team plan.”

## 0:25–0:55 — Select an operating context

Choose a forecast window, police station and number of available teams in the sidebar.

Explain that each point is a 500 m operational zone and that the model was trained only on earlier records.

## 0:55–1:30 — Show impact-aware prioritization

Point out a zone with fewer predicted violations but a higher impact score because it has junction, main-road or heavy-vehicle patterns.

Use the key line:

“ClearLane does not ask only where violations are common; it asks which violations are most likely to obstruct traffic.”

## 1:30–2:05 — Generate the patrol plan

Show team assignments, recommended windows, reasons and actions. Change the team count and demonstrate that the plan updates immediately.

## 2:05–2:35 — Prove it is ML

Open Model performance. Compare ClearLane against the same-window-last-week baseline. Show ranking capture and feature importance.

## 2:35–3:00 — Close honestly

Explain that obstruction units are a transparent proxy because direct traffic-speed data is absent. State that a production pilot would calibrate the score against probe speeds and post-enforcement lane recovery.
