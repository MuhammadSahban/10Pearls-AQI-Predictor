10PEARLS AQI PREDICTOR
=====================

WHAT THIS PROJECT DOES
-----------------------
Predicts US AQI for the next 24, 48, and 72 hours for three cities:
Karachi, Lahore, and Islamabad. Data comes from the Open-Meteo Weather
and Air Quality APIs. Three candidate models (Random Forest, Gradient
Boosting, Ridge) are trained per horizon and the best one (by RMSE) is
kept automatically. Everything is served through a live Streamlit
dashboard.


TECH STACK
-----------
- Python 3.11
- pandas, numpy, scikit-learn -- feature engineering and modeling
- SHAP -- feature importance explanations
- Backblaze B2 (S3-compatible) -- feature store + model registry
- GitHub Actions -- pipeline execution (hourly features, daily
  training, one-time manual backfill), triggered by an external cron
  service rather than GitHub's own native schedule (see SCHEDULING
  below for why)
- cron-job.org -- external scheduler that reliably
  triggers the GitHub Actions workflows on time
- Streamlit -- live dashboard, deployed on Streamlit Community Cloud


PROJECT STRUCTURE
-------------------
src/config.py               cities, horizons, feature settings
src/open_meteo.py           Open-Meteo API client
src/storage.py              B2 / Hopsworks / local storage abstraction
                             (FeatureStore, ModelRegistry, ReportStore)
src/feature_pipeline.py     hourly: fetch + engineer features
src/backfill_historical.py  one-off: deep historical backfill
src/training_pipeline.py    daily: train 3 candidates per horizon,
                             keep the best
src/inference_pipeline.py   prediction logic, used live by the
                             Streamlit app
src/shap_analysis.py        feature importance plots
src/baseline_check.py       compares trained models against a naive
                             persistence baseline
streamlit_app.py            the dashboard (city picker, live forecasts,
                             hazard alerts, SHAP panel, pipeline status)
.github/workflows/          feature_pipeline.yml (hourly),
                             training_pipeline.yml (daily),
                             backfill.yml (manual, one-time)
requirements.txt            core dependencies (includes boto3 for B2)
.gitignore                  excludes .venv, .env, __pycache__, .idea,
                             and local_store/ contents


HOW PREDICTION ACTUALLY WORKS (TRAIN/SERVE CONSISTENCY)
----------------------------------------------------------
For a training row at time T predicting AQI at T+horizon, the weather
features used are ALSO taken at T+horizon (shifted forward from
history), not at T. This matters because in production you don't have
real weather for the future -- you have Open-Meteo's forecast for it.
Training uses shifted historical weather as a stand-in for what a
forecast would have said; live inference uses Open-Meteo's actual
forecast for that same future timestamp. Same feature, same column,
consistent structure between training and serving.

AQI lag/rolling features are always computed using only values from
before the current row (shift >= 1), so they never leak future
information into training.

Live predictions are anchored to the TRUE current time (a fixed GMT+5
offset, since all three cities are in Pakistan and observe no daylight
saving), not simply "the last row of fetched data." This matters
because each live fetch pulls ~3 days of past data plus ~4 days of
Open-Meteo's own forecast, so "the last row" is actually several days
in the future -- using it as "now" would have silently shifted every
prediction and the "current AQI" reading forward by days.


WHERE EACH PIECE ACTUALLY RUNS
---------------------------------
- backfill_historical.py   -> GitHub Actions, triggered manually once
- feature_pipeline.py      -> GitHub Actions, triggered hourly by an
                               external cron service (see SCHEDULING)
- training_pipeline.py     -> GitHub Actions, triggered daily by the
                               same external cron service
- shap_analysis.py         -> GitHub Actions, runs right after
                               training in the same workflow
- baseline_check.py        -> run manually / on demand
- Prediction (inference)   -> runs LIVE inside the Streamlit app's own
                               process, cached for about an hour.


SCHEDULING (EXTERNAL CRON, NOT GITHUB'S NATIVE SCHEDULE)
-------------------------------------------------------------
GitHub Actions' built-in `schedule:` trigger is best-effort and can be
delayed significantly, especially around the top of every hour when
huge numbers of repositories worldwide are all scheduled at once. In
this project that meant the hourly/daily jobs were not reliably firing
on time (manual `workflow_dispatch` runs always worked fine and fast,
confirming the workflows themselves were correct -- it was purely a
native-scheduling reliability issue).

Fix: the `schedule:` trigger was removed from feature_pipeline.yml and
training_pipeline.yml entirely. `workflow_dispatch: {}` was kept in
both, and an external free cron service
(cron-job.org) is configured to call GitHub's REST API on a schedule:

    POST https://api.github.com/repos/<you>/<repo>/actions/workflows/<workflow-file>.yml/dispatches
    Authorization: Bearer <fine-grained personal access token>
    Body: {"ref": "main"}

The token is a fine-grained PAT scoped to ONLY this repository, with
ONLY "Actions: Read and write" permission -- nothing broader. This
swaps "GitHub's queued, sometimes-delayed native cron" for "an
external, reliable trigger of the exact same manual dispatch mechanism
that was already confirmed to work."


STORAGE POLICY
----------------
If B2 credentials are configured, EVERYTHING lives in B2 only: models,
features, SHAP plots, training summaries, and pipeline run-status logs.
Nothing is written to local disk and nothing is committed to git in
that mode -- models are streamed to/from B2 purely in memory.

Only when no cloud backend is configured at all does local_store/
become the real persistent store, and in that fallback mode GitHub
Actions commits it back into the repo after each run.


SETUP
-------
1. pip install -r requirements.txt
2. Set your B2 credentials as environment variables (never commit
   real credentials to the repo).
3. Run once locally to bootstrap:
     python src/backfill_historical.py --start YYYY-MM-DD --end YYYY-MM-DD
     python src/feature_pipeline.py
     python src/training_pipeline.py
     python src/shap_analysis.py
4. Run the dashboard locally:
     streamlit run streamlit_app.py


SEEING WHAT HAPPENED ON EACH RUN
-----------------------------------
- Full logs: GitHub repo -> Actions tab -> pick a workflow run.
- At-a-glance status: the "Pipeline status / logs" panel inside the
  Streamlit dashboard itself, reading the same status logs.


KNOWN LIMITATIONS
--------------------
- Accuracy naturally degrades with horizon -- 72h predictions are
  bounded by how good Open-Meteo's own 72h weather forecast is, not
  just the model.
- Evaluation uses a time-based (not random) train/test split, since
  shuffling time series data leaks future information into training.
- The persistence-baseline check (src/baseline_check.py) exists
  specifically to verify the trained models add real value beyond
  simply assuming AQI doesn't change -- see the project report for the
  actual comparison results.
