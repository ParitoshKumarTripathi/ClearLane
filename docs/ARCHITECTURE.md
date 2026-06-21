# Architecture

## Runtime architecture

ClearLane is deployed as one Streamlit application. The app reads compact prediction and history artifacts and executes the patrol optimizer in-process. There is no separate API dependency during judging.

```text
Browser
  ↕
Streamlit server
  ├── demo_predictions.csv.gz
  ├── zone_history.csv.gz
  ├── metrics.json
  ├── feature_importance.csv
  └── patrol optimizer
```

This minimizes integration failure and keeps the deployed service stateless.

## Offline training architecture

The offline pipeline is intentionally separate from the demo runtime:

1. `preprocessing.py` validates coordinates, statuses and timestamps.
2. `impact.py` parses offences and creates transparent obstruction units.
3. `features.py` constructs a dense zone-time panel, including zero-event windows.
4. `modeling.py` trains the hotspot and impact models.
5. `evaluation.py` compares ClearLane with same-window-last-week and seven-day-average baselines.
6. `optimizer.py` converts a selected forecast window into a team plan.
7. `pipeline.py` exports all deployable artifacts.

## Design choices

- Fixed metric grid instead of an external geospatial API
- Numeric-only features to avoid fragile categorical encoders
- Poisson regression loss for non-negative count-like targets
- Time-based holdout instead of random splitting
- Precomputed demo predictions to avoid retraining on application startup
