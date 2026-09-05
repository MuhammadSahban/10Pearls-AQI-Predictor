"""
Feature pipeline. Run this hourly (see .github/workflows/feature_pipeline.yml).
For each city it fetches recent weather + AQI, computes lag/rolling/time
features, and upserts into the feature store. Horizon-specific target
alignment is deliberately NOT done here -> that lives in
training_pipeline.py / inference_pipeline.py so it's defined in one place.
"""
import pandas as pd

from config import CITIES, LAG_HOURS, ROLLING_WINDOWS, FEATURE_STORE_NAME
from open_meteo import fetch_weather, fetch_air_quality
from storage import FeatureStore, log_run_status


def add_time_features(df, ts_col="timestamp"):
    df = df.copy()
    df["hour"] = df[ts_col].dt.hour
    df["dayofweek"] = df[ts_col].dt.dayofweek
    df["day"] = df[ts_col].dt.day
    df["month"] = df[ts_col].dt.month
    return df


def add_lag_features(df, col="us_aqi"):
    """All lag/rolling features use shift(>=1): only values strictly
    BEFORE the current row -> never leaks same-timestamp or future
    values into a feature."""
    df = df.sort_values("timestamp").copy()
    for lag in LAG_HOURS:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    for win in ROLLING_WINDOWS:
        df[f"{col}_roll_mean_{win}h"] = df[col].shift(1).rolling(win).mean()
        df[f"{col}_roll_std_{win}h"] = df[col].shift(1).rolling(win).std()
    df[f"{col}_change_rate_1h"] = df[col].diff()
    return df


def build_city_features(city, lat, lon, past_days=60, forecast_days=1):
    weather = fetch_weather(lat, lon, past_days=past_days, forecast_days=forecast_days)
    aqi = fetch_air_quality(lat, lon, past_days=past_days, forecast_days=forecast_days)
    df = pd.merge(weather, aqi, on="timestamp", how="inner")
    df["city"] = city
    df = add_time_features(df)
    df = add_lag_features(df, "us_aqi")
    return df


def run(past_days=5):
    fs = FeatureStore()
    frames = []
    for city, coords in CITIES.items():
        print(f"[feature_pipeline] fetching {city} ...")
        df = build_city_features(city, coords["lat"], coords["lon"], past_days=past_days)
        frames.append(df)
        print(f"[feature_pipeline] {city}: {len(df)} rows")

    all_df = pd.concat(frames, ignore_index=True)
    saved_df=fs.save_features(all_df, name=FEATURE_STORE_NAME)
    city_counts = saved_df["city"].value_counts().to_dict()
    print(f"[feature_pipeline] saved {len(saved_df)} rows total to '{FEATURE_STORE_NAME}'")
    return {"total_rows": len(saved_df), "rows_by_city": city_counts}


if __name__ == "__main__":
    try:
        result = run()
        log_run_status("feature_pipeline", "success", result)
    except Exception as e:
        log_run_status("feature_pipeline", "failed", {"error": str(e)})
        raise
