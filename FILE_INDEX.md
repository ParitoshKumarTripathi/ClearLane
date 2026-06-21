# ClearLane file index

## Entry points

| File | Purpose |
|---|---|
| `app.py` | Streamlit operations dashboard used in the deployed demo |
| `config.yaml` | Single source of truth for data, grid, time, model and deployment parameters |
| `Makefile` | Short commands for installation, pipeline execution, tests and local app startup |

## Reproducible ML pipeline

| File | Purpose |
|---|---|
| `src/clearlane/preprocessing.py` | Reads the competition CSV, removes invalid/rejected records and creates record-level features |
| `src/clearlane/impact.py` | ClearLane offence severity, vehicle-size and road-context rules |
| `src/clearlane/grid.py` | Metric spatial grid, centroids and distance calculations |
| `src/clearlane/features.py` | Dense zone-time panel, lags, rolling history, trends and prediction targets |
| `src/clearlane/modeling.py` | Deterministic sampling, model training, target transformation and inference |
| `src/clearlane/evaluation.py` | Error, ranking, capture and baseline evaluation |
| `src/clearlane/optimizer.py` | Converts one forecast window into a capacity-aware team plan |
| `src/clearlane/pipeline.py` | Runs the entire workflow and exports models and artifacts |
| `src/clearlane/cli.py` | Command-line interface for inspection and full rebuilds |

## Deployment

| File | Purpose |
|---|---|
| `requirements.txt` | Exact runtime and model-rebuild dependencies |
| `requirements-dev.txt` | Test and lint dependencies |
| `Dockerfile` | Reproducible Render/container image |
| `render.yaml` | Render Blueprint service definition |
| `runtime.txt` | Python runtime hint for hosted deployment |
| `.streamlit/config.toml` | Streamlit theme, server and privacy settings |
| `.dockerignore` | Keeps raw data, caches and local files out of the image |
| `.github/workflows/ci.yml` | Automated tests, linting and UI smoke test |

## Generated model files

| File | Purpose |
|---|---|
| `models/hotspot_model.joblib` | Next-window parking-volume model |
| `models/impact_model.joblib` | Log-target obstruction-impact model |
| `models/model_metadata.json` | Features, split, seed, losses, sampling and model horizon |

## Generated dashboard artifacts

| File | Purpose |
|---|---|
| `artifacts/demo_predictions.csv.gz` | April holdout predictions shown on the map |
| `artifacts/zone_history.csv.gz` | Recent zone history used by detail charts |
| `artifacts/demo_patrol_plan.csv` | Default example deployment plan |
| `artifacts/metrics.json` | Full ClearLane and baseline metrics |
| `artifacts/feature_importance.csv` | Permutation importance for both models |
| `artifacts/data_summary.json` | Cleaned data and model panel summary |
| `artifacts/zone_reference.csv` | Model zone coordinates and readable names |
| `artifacts/police_station_summary.csv` | Station-level descriptive summary |
| `artifacts/run_manifest.json` | Inventory of generated outputs |

## Quality and documentation

| File | Purpose |
|---|---|
| `tests/test_impact.py` | Validates offence and vehicle scoring |
| `tests/test_grid.py` | Validates spatial calculations |
| `tests/test_features.py` | Validates required temporal features |
| `tests/test_optimizer.py` | Validates patrol selection behavior |
| `scripts/smoke_app.py` | Executes the Streamlit app headlessly before deployment |
| `docs/ARCHITECTURE.md` | Offline and deployed system architecture |
| `docs/METHODOLOGY.md` | Feature, target, leakage and evaluation methodology |
| `docs/RESULTS.md` | Judge-ready April holdout results |
| `docs/LIMITATIONS.md` | Honest constraints and future validation plan |
| `docs/DEPLOYMENT.md` | Streamlit and Render deployment checklist |
| `docs/DEMO_SCRIPT.md` | Three-minute judging walkthrough |
| `docs/UI_DESIGN.md` | Dashboard interaction and accessibility rationale |
