# Deployment

## Pre-deployment checklist

- Confirm `artifacts/demo_predictions.csv.gz` exists.
- Confirm `artifacts/zone_history.csv.gz` exists.
- Confirm `artifacts/metrics.json` exists.
- Confirm both `.joblib` files and `model_metadata.json` exist under `models/`.
- Run `PYTHONPATH=src pytest -q`.
- Run `streamlit run app.py` locally.
- Do not commit the raw competition CSV unless the competition explicitly permits it.

## Streamlit Community Cloud

- Entry point: `app.py`
- Python version: select `3.12` in Advanced settings when creating the app.
- Python dependencies: `requirements.txt` contains only dashboard runtime packages.
- Training and test dependencies: `requirements-dev.txt`.
- Runtime artifacts: committed `artifacts/` and `models/`
- No secret or API key is required.

The app inserts `src/` into `sys.path`, so no package installation step is required beyond normal dependency installation.

## Render

Use the included Blueprint or Dockerfile.

Health endpoint:

```text
/_stcore/health
```

The Docker image exposes port 8501 and binds Streamlit to `0.0.0.0`.

## Production extension

For a real-time deployment, separate the system into:

- scheduled ingestion and feature computation;
- model service;
- prediction store;
- Streamlit or web client;
- monitoring for data drift, coverage and patrol exposure.
