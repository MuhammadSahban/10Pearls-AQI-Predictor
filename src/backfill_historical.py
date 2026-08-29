"""
Run this ONCE to backfill deep history before your first training run.
The recent-window API only gives ~92 days; this uses archive endpoints
to pull a longer date range.

Usage:
    python backfill_historical.py --start 2024-01-01 --end 2026-08-01
"""
import argparse
import pandas as pd

from config import CITIES, FEATURE_STORE_NAME
from open_meteo import fetch_weather, fetch_air_quality
from feature_pipeline import add_time_features, add_lag_features
from storage import FeatureStore


def run(start_date, end_date):
    fs = FeatureStore()
    frames = []
    for city, coords in CITIES.items():
        print(f"[backfill] {city}: {start_date} -> {end_date}")
        weather = fetch_weather(coords["lat"], coords["lon"],
                                 start_date=start_date, end_date=end_date)
        aqi = fetch_air_quality(coords["lat"], coords["lon"],
                                 start_date=start_date, end_date=end_date)
        df = pd.merge(weather, aqi, on="timestamp", how="inner")
        df["city"] = city
        df = add_time_features(df)
        df = add_lag_features(df, "us_aqi")
        frames.append(df)
        print(f"[backfill] {city}: {len(df)} rows")

    all_df = pd.concat(frames, ignore_index=True)
    fs.save_features(all_df, name=FEATURE_STORE_NAME)
    print(f"[backfill] saved {len(all_df)} rows total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    run(args.start, args.end)
