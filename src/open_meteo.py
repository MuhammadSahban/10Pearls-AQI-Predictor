"""
Thin client around Open-Meteo's Weather and Air Quality APIs.

Two modes are supported for each fetch function:
  - "recent window" mode: pass past_days / forecast_days -> hits the live
    forecast endpoints, which return recent history AND future forecast
    from the SAME model family. Use this for daily feature updates and
    for inference, so train/serve data comes from a consistent source.
  - "date range" mode: pass start_date / end_date -> hits the archive
    endpoints. Use this ONCE to backfill deep history for initial training.
"""
import requests
import pandas as pd

from config import WEATHER_VARS, POLLUTANT_VARS

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

TIMEOUT = 30


def _to_df(hourly_json):
    df = pd.DataFrame(hourly_json)
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"time": "timestamp"})


def fetch_weather(lat, lon, past_days=None, forecast_days=None,
                   start_date=None, end_date=None):
    if start_date and end_date:
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(WEATHER_VARS),
            "start_date": start_date, "end_date": end_date,
            "timezone": "auto",
        }
        url = WEATHER_ARCHIVE_URL
    else:
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(WEATHER_VARS),
            "past_days": past_days if past_days is not None else 3,
            "forecast_days": forecast_days if forecast_days is not None else 4,
            "timezone": "auto",
        }
        url = WEATHER_FORECAST_URL

    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return _to_df(r.json()["hourly"])


def fetch_air_quality(lat, lon, past_days=None, forecast_days=None,
                       start_date=None, end_date=None):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(POLLUTANT_VARS),
        "timezone": "auto",
    }
    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        params["past_days"] = past_days if past_days is not None else 3
        params["forecast_days"] = forecast_days if forecast_days is not None else 4

    r = requests.get(AQI_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return _to_df(r.json()["hourly"])
