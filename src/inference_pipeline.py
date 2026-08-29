"""
Inference pipeline. For each city, fetches the latest recent-window data
(past few days + Open-Meteo's own forecast for the next few days), builds
one feature row per horizon using:
  - AQI lag/rolling features computed at "now" (real observed data)
  - weather at target_time = now + horizon, taken directly from
    Open-Meteo's forecast (the real forecasted weather, not a stand-in
    like in training -> the actual thing)
  - target_hour/target_dayofweek/target_month for the target time
...then loads the matching horizon model and predicts.

Run this after every feature_pipeline.py run (hourly), or on-demand from
the dashboard's "Refresh" button.
"""
import json
import pathlib
from datetime import datetime, timezone

import pandas as pd

from config import CITIES, HORIZONS, WEATHER_VARS
from open_meteo import fetch_weather, fetch_air_quality
from feature_pipeline import add_time_features, add_lag_features
from storage import ModelRegistry

LOCAL_STORE = pathlib.Path(__file__).resolve().parent.parent / "local_store"


def aqi_category(aqi):
    if aqi is None or pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_feature_row(city_df, horizon, feature_cols):
    latest = city_df.sort_values("timestamp").iloc[[-1]].copy()
    now_ts = latest["timestamp"].iloc[0]
    target_ts = now_ts + pd.Timedelta(hours=horizon)

    row = latest.copy()
    target_row = city_df[city_df["timestamp"] == target_ts]
    for var in WEATHER_VARS:
        if len(target_row):
            row[f"{var}_target"] = target_row[var].values[0]
        else:
            row[f"{var}_target"] = city_df[var].iloc[-1]

    row["target_hour"] = target_ts.hour
    row["target_dayofweek"] = target_ts.dayofweek
    row["target_month"] = target_ts.month

    row = pd.get_dummies(row, columns=["city"], prefix="city")
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    return row[feature_cols], now_ts, target_ts


def run():
    mr = ModelRegistry()
    results = []

    for city, coords in CITIES.items():
        print(f"[inference] {city} ...")
        weather = fetch_weather(coords["lat"], coords["lon"], past_days=3, forecast_days=4)
        aqi = fetch_air_quality(coords["lat"], coords["lon"], past_days=3, forecast_days=4)
        df = pd.merge(weather, aqi, on="timestamp", how="inner")
        df["city"] = city
        df = add_time_features(df)
        df = add_lag_features(df, "us_aqi")

        observed = aqi.dropna(subset=["us_aqi"]).sort_values("timestamp")
        current_aqi = float(observed["us_aqi"].iloc[-1]) if len(observed) else None

        city_result = {
            "city": city,
            "current_aqi": current_aqi,
            "current_category": aqi_category(current_aqi),
            "current_time": str(observed["timestamp"].iloc[-1]) if len(observed) else None,
            "forecasts": [],
            "recent_history": observed.tail(72)[["timestamp", "us_aqi"]]
                .assign(timestamp=lambda d: d["timestamp"].astype(str))
                .to_dict(orient="records"),
        }

        for horizon in HORIZONS:
            model, feature_cols = mr.load_model_and_columns(f"aqi_model_{horizon}h")
            if model is None or feature_cols is None:
                print(f"[inference]   no trained model for +{horizon}h yet, skipping")
                continue
            X, now_ts, target_ts = get_feature_row(df, horizon, feature_cols)
            pred = float(model.predict(X)[0])
            city_result["forecasts"].append({
                "horizon_hours": horizon,
                "target_time": str(target_ts),
                "predicted_aqi": round(pred, 1),
                "category": aqi_category(pred),
            })

        results.append(city_result)

    LOCAL_STORE.mkdir(parents=True, exist_ok=True)
    out_path = LOCAL_STORE / "latest_predictions.json"
    payload = {"generated_at": str(datetime.now(timezone(timedelta(hours=5)))), "predictions": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[inference] wrote {out_path}")
    return payload


if __name__ == "__main__":
    run()
