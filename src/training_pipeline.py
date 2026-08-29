"""
Training pipeline. For EACH horizon (24h, 48h, 72h), trains multiple
candidate models, evaluates them on a time-based holdout split, and
keeps only the best one (by lowest RMSE) — saved to Hopsworks (if
HOPSWORKS_API_KEY is set) or local_store/ otherwise.

Candidate models per horizon:
  - RandomForestRegressor      (bagged trees, robust baseline)
  - GradientBoostingRegressor  (boosted trees, often stronger on tabular data)
  - Ridge                      (regularized linear model, cheap sanity-check baseline)

This is "direct multi-step" forecasting: one independently-trained model
per horizon, so no error compounding across 24h -> 48h -> 72h.

Critical detail (this is exactly what caused the mismatch you hit
before): for a row at time t, the label is us_aqi at t+horizon. The
weather features used for that row are the weather AT t+horizon
(shifted forward), NOT the weather at t -> because at inference time
you won't have live weather for t+horizon, you'll have Open-Meteo's
FORECAST for t+horizon, which these shifted historical values stand in
for during training. AQI lag/rolling features stay anchored at t, since
those come from real observed history, never future data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import HORIZONS, WEATHER_VARS, FEATURE_STORE_NAME
from storage import FeatureStore, ModelRegistry, ReportStore, log_run_status

# Candidate models to try for every horizon. Add/remove entries here to
# change what gets compared — everything downstream picks it up.
CANDIDATE_MODELS = {
    "random_forest": lambda: RandomForestRegressor(
        # Tuned down from n_estimators=300/max_depth=14/min_samples_leaf=3.
        # On a 1-year/3-city backfill this cut the serialized model from
        # ~77MB to ~6MB (further to ~2MB with joblib compression, see
        # storage.py) with no measurable RMSE/R2 change -- the original
        # settings were mostly building bigger, deeper trees than the
        # data actually supports, not more accurate ones.
        n_estimators=100, max_depth=10, min_samples_leaf=10,
        n_jobs=-1, random_state=42,
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42,
    ),
    "ridge": lambda: Ridge(alpha=1.0),
}


def build_horizon_dataset(df, horizon):
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    out = []
    for city, g in df.groupby("city"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["target"] = g["us_aqi"].shift(-horizon)

        for var in WEATHER_VARS:
            g[f"{var}_target"] = g[var].shift(-horizon)

        target_time = g["timestamp"] + pd.to_timedelta(horizon, unit="h")
        g["target_hour"] = target_time.dt.hour
        g["target_dayofweek"] = target_time.dt.dayofweek
        g["target_month"] = target_time.dt.month
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def train_and_select_best(df, horizon):
    """Trains every candidate model for this horizon, evaluates them all
    on the same time-based holdout split, and returns the best one plus
    a full comparison table (so you can put it straight in your report)."""
    data = build_horizon_dataset(df, horizon)
    data = pd.get_dummies(data, columns=["city"], prefix="city")

    feature_cols = [c for c in data.columns if c not in ("timestamp", "target")]
    data = data.dropna(subset=["target"] + feature_cols)

    data = data.sort_values("timestamp")
    split_idx = int(len(data) * 0.85)
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    comparison = {}
    trained_models = {}

    for model_name, build_fn in CANDIDATE_MODELS.items():
        model = build_fn()
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        metrics["n_train"] = int(len(train))
        metrics["n_test"] = int(len(test))
        comparison[model_name] = metrics
        trained_models[model_name] = model
        print(f"[training]     {model_name:<18} "
              f"RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}")

    best_name = min(comparison, key=lambda name: comparison[name]["rmse"])
    best_model = trained_models[best_name]
    best_metrics = dict(comparison[best_name])
    best_metrics["selected_model"] = best_name

    return best_model, best_metrics, comparison, feature_cols


def run():
    fs = FeatureStore()
    mr = ModelRegistry()
    rs = ReportStore()

    df = fs.load_features(name=FEATURE_STORE_NAME)
    if df.empty:
        raise RuntimeError("No features found. Run feature_pipeline.py "
                            "(and ideally backfill_historical.py) first.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    summary = {}
    for horizon in HORIZONS:
        print(f"[training] horizon +{horizon}h — comparing {list(CANDIDATE_MODELS)} ...")
        best_model, best_metrics, comparison, feature_cols = train_and_select_best(df, horizon)
        print(f"[training]   -> best for +{horizon}h: {best_metrics['selected_model']} "
              f"(RMSE={best_metrics['rmse']:.2f}, R2={best_metrics['r2']:.3f})")

        mr.save_model(best_model, name=f"aqi_model_{horizon}h",
                       metrics=best_metrics, feature_cols=feature_cols)

        # Full comparison table (all candidates, not just the winner) --
        # goes to the same place as everything else (B2 if configured,
        # else local_store/reports/), never a stray local-only file.
        rs.save_json(f"aqi_model_{horizon}h_comparison.json", comparison)

        summary[f"{horizon}h"] = {"selected_model": best_metrics["selected_model"],
                                   "metrics": best_metrics, "all_candidates": comparison}

    rs.save_json("training_summary.json", summary)
    print(f"[training] done. Summary saved via ReportStore (backend: {rs.backend}).")
    return summary


if __name__ == "__main__":
    try:
        result = run()
        log_run_status("training_pipeline", "success",
                        {h: v["selected_model"] for h, v in result.items()})
    except Exception as e:
        log_run_status("training_pipeline", "failed", {"error": str(e)})
        raise
