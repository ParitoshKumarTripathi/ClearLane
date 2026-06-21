# ClearLane

**ClearLane predicts where illegal parking will obstruct traffic and tells enforcement teams where and when to act.**

ClearLane is an impact-aware parking intelligence system built for a 24-hour ML hackathon. It transforms historical enforcement records into two-hour, 500 m zone forecasts, estimates relative traffic-obstruction risk, and produces a capacity-aware patrol plan.

## Why it is different

A conventional heatmap shows where violations happened. ClearLane:

1. forecasts the next patrol window;
2. separates expected violation volume from obstruction impact;
3. corrects the operational story for enforcement intensity and record confidence;
4. converts model output into a deployable team plan with explanations.

## End-to-end architecture

```text
Competition CSV
    ↓
Cleaning, validation filtering, canonical vehicle fields
    ↓
Parking offence parsing + ClearLane Impact Engine
    ↓
500 m spatial zones × two-hour time windows
    ↓
Lag, rolling, trend, road-context, vehicle and enforcement features
    ↓
┌──────────────────────────┬───────────────────────────┐
│ Hotspot volume model     │ Obstruction impact model  │
│ next-window violations   │ next-window impact units  │
└──────────────────────────┴───────────────────────────┘
    ↓
Temporal holdout evaluation
    ↓
Capacity-aware patrol optimizer
    ↓
Streamlit operations dashboard
```

## Repository

```text
clearlane/
├── app.py                         # Streamlit dashboard
├── config.yaml                    # Reproducible experiment parameters
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Test and lint dependencies
├── Dockerfile                     # Render/container deployment
├── render.yaml                    # Render blueprint
├── Makefile                       # Common commands
├── .streamlit/config.toml         # Accessible, minimal visual theme
├── src/clearlane/
│   ├── preprocessing.py           # Data cleaning and record construction
│   ├── impact.py                  # Domain-specific obstruction proxy
│   ├── grid.py                    # 500 m spatial grid utilities
│   ├── features.py                # Zone-time panel and leakage-safe features
│   ├── modeling.py                # Deterministic model training and inference
│   ├── evaluation.py              # Regression and ranking evaluation
│   ├── optimizer.py               # Patrol allocation
│   ├── pipeline.py                # End-to-end artifact generation
│   └── cli.py                     # Command-line interface
├── tests/                         # Unit and smoke tests
├── docs/                          # Method, architecture, deployment and demo
├── models/                        # Generated trained model artifacts
└── artifacts/                     # Generated dashboard and evaluation files
```

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Add the competition data

Place the CSV at:

```text
data/raw/police_violations.csv
```

### 3. Rebuild everything

```bash
PYTHONPATH=src python -m clearlane.cli pipeline \
  --input "data/raw/police_violations.csv"
```

To avoid saving the large intermediate Parquet files:

```bash
PYTHONPATH=src python -m clearlane.cli pipeline \
  --input "data/raw/police_violations.csv" \
  --skip-intermediate
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

### 5. Test

```bash
PYTHONPATH=src pytest -q
```

## Reproducibility

- Fixed random seed in `config.yaml`
- Exact runtime dependency versions
- No random train/test split
- Training data ends before the April holdout period
- Every lag and rolling feature uses only data available before its target window
- Baselines and model outputs are evaluated on identical zone-window rows
- Model metadata records the features, split, grid resolution and horizon

## Models

Both models use compact scikit-learn histogram gradient boosting:

- **Hotspot model:** Poisson-loss regression for observed parking-violation counts.
- **Impact model:** squared-error regression on a `log1p` transformed obstruction target, which is inverted at prediction time and bounded against implausible sparse-data extrapolation.

The deployment deliberately avoids native boosting dependencies so Streamlit and Render builds are less fragile.

## ClearLane Impact Engine

The dataset does not include traffic speed, queue length or travel time. ClearLane therefore estimates **relative obstruction risk** using a transparent proxy:

```text
obstruction units = offence severity
                  × vehicle size factor
                  × road/junction context
                  × peak-hour factor
```

The UI never presents this proxy as an observed percentage reduction in traffic speed.

## Main artifacts

| File | Purpose |
|---|---|
| `models/hotspot_model.joblib` | Trained next-window volume model |
| `models/impact_model.joblib` | Trained next-window impact model |
| `models/model_metadata.json` | Reproducibility metadata |
| `artifacts/demo_predictions.csv.gz` | Holdout predictions consumed by the dashboard |
| `artifacts/zone_history.csv.gz` | Recent history for zone trend charts |
| `artifacts/metrics.json` | Model and baseline evaluation |
| `artifacts/feature_importance.csv` | Permutation importance |
| `artifacts/demo_patrol_plan.csv` | Default operational plan |

## Deployment

### Streamlit Community Cloud

1. Push the repository, including `models/` and `artifacts/`, to GitHub.
2. Create a Streamlit Community Cloud app from the repository.
3. Select `app.py` as the entry point.
4. Deploy. The raw competition CSV is not needed at runtime.

### Render

The included `Dockerfile` and `render.yaml` are deployment-ready. Create a Render Blueprint from the repository or deploy the Docker service manually.

See `docs/DEPLOYMENT.md` for the checklist.

## Limitations

- The model predicts **observed enforcement records**, which reflect both parking behavior and patrol activity.
- Obstruction units are an explainable proxy, not measured traffic flow.
- The data does not contain enforcement outcomes after a patrol.
- April is used as a historical simulation because no live violation feed is connected.

See `docs/LIMITATIONS.md` for mitigation and future work.
