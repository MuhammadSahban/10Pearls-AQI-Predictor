# Pearls AQI Predictor

Predicts US AQI for the next 24h, 48h, and 72h for Karachi, Lahore, and
Islamabad. Architecture: GitHub Actions (hourly features, daily
training) + Hopsworks (feature store + model registry) + a single
Streamlit dashboard that computes predictions live.

## Architecture

```
                 hourly cron                    daily cron
GitHub Actions -----------------> Hopsworks <----------------- GitHub Actions
(feature_pipeline.py)          Feature Store      (training_pipeline.py
  fetches Open-Meteo data      + Model Registry     trains 3 candidate
  -> writes features                                 models per horizon,
                                                       keeps the best,
                                                       pushes new version)
                                      |
                                      | latest model version, on demand
                                      v
                              Streamlit app
                    (streamlit_app.py, deployed separately)
              - fetches LIVE Open-Meteo data on each cache refresh
              - loads the newest model version from Hopsworks
              - predicts 24h/48h/72h AQI, city picker, alerts
```

There's no separate backend server (Flask was removed) — the Streamlit
app IS the website. It talks to Hopsworks directly and computes
predictions on demand, cached for an hour so it doesn't hammer
Open-Meteo/Hopsworks on every click, with a manual "Refresh now" button
to force an immediate recompute.

## Why predictions update automatically after daily retraining

`storage.ModelRegistry.load_model_and_columns()` always checks Hopsworks
for the *highest version number* registered under a model's name before
ever touching any local cache. `training_pipeline.py` pushes a new
version every time it runs (once a day), so the very next time the
Streamlit app's cache expires (at most an hour later), it will pick up
that new version with zero redeploys or manual steps.

## Why weather-at-target-time matters (avoids the leakage bug)

For a training row at time t with target "AQI at t+horizon", the weather
features are also taken **at t+horizon** (shifted forward from history),
not at t — because in production you won't have live weather for the
future, you'll have Open-Meteo's *forecast* for it. Training and
inference therefore use the same kind of input at every step.

## Project layout

```
src/
  config.py               cities, horizons, feature settings
  open_meteo.py            Open-Meteo API client (recent-window + archive modes)
  storage.py               Hopsworks-or-local storage abstraction
  feature_pipeline.py      hourly: fetch + engineer features -> feature store
  backfill_historical.py   one-off: deep historical backfill
  training_pipeline.py     daily: trains 3 candidate models per horizon,
                             keeps the best (by RMSE) -> model registry
  inference_pipeline.py    shared prediction logic, imported by the
                             Streamlit app (and runnable standalone/CLI
                             for local testing)
  shap_analysis.py         feature importance plots
streamlit_app.py           the dashboard/website: city picker, live
                             24h/48h/72h forecasts, hazard alerts, AQI history chart
.github/workflows/
  feature_pipeline.yml      hourly: refresh the feature store
  training_pipeline.yml     daily: retrain + select best model per horizon
local_store/                fallback storage when Hopsworks isn't configured
```

## Storage backend

Three backends, auto-selected by which credentials are present
(checked in this order — see `src/storage.py`):

1. **Hopsworks** — if `HOPSWORKS_API_KEY` is set.
2. **Backblaze B2** — if `B2_KEY_ID`, `B2_APPLICATION_KEY`,
   `B2_BUCKET_NAME`, and `B2_ENDPOINT_URL` are all set. A real cloud
   object store, no credit card required at signup, 10GB free. Feature
   parquet files and versioned model bundles are stored under a
   `b2-store/` prefix in your bucket (e.g.
   `b2-store/models/aqi_model_24h/v3/model.joblib`), so the backend
   that produced a file is visible in its own path.
3. **Local disk** (`local_store/`) — fallback if neither of the above
   is configured.

Set whichever one you're using:
- Locally: as environment variables.
- In GitHub Actions: as repo secrets (Settings -> Secrets -> Actions).
- In Streamlit Community Cloud: as app secrets (Settings -> Secrets),
  same key names, so the deployed dashboard also reads from the same backend.

If using B2, also install its extra dependency:
```bash
pip install -r requirements-b2.txt
```

## First-time bootstrap

```bash
cd src
python backfill_historical.py --start 2024-01-01 --end 2026-08-01   # deep history, once
python feature_pipeline.py                                          # recent features
python training_pipeline.py                                         # trains + selects best models
cd ..
```

## Run the dashboard

```bash
streamlit run streamlit_app.py
```

## Deploying

1. Push this repo to GitHub.
2. On GitHub: add `HOPSWORKS_API_KEY` as a repo secret so the two
   scheduled workflows can reach Hopsworks.
3. On Streamlit Community Cloud: connect the repo, set `streamlit_app.py`
   as the entry point, and add `HOPSWORKS_API_KEY` as an app secret.

That's the whole deployment — one repo, one Streamlit app, two cron
workflows, no server to manage.

## Notes / known limitations (worth putting in your final report)

- Accuracy naturally degrades with horizon — 72h predictions are bounded
  by how good Open-Meteo's own 72h weather forecast is, not just your model.
- `local_store/aqi_model_{horizon}h_comparison.json` (written by
  training_pipeline.py) has all 3 candidate models' metrics side by
  side, not just the winner — useful evidence for the report's
  "multiple forecasting models" requirement.
- Evaluation uses a time-based (not random) train/test split, since
  shuffling time series data leaks future information into training.
