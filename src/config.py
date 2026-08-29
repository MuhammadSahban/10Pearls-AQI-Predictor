"""
Central config for the AQI predictor project.
Edit CITIES to add/remove cities. Everything else picks it up automatically.
"""

CITIES = {
    "Karachi":   {"lat": 24.8607, "lon": 67.0011},
    "Lahore":    {"lat": 31.5497, "lon": 74.3436},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
}

HORIZONS = [24, 48, 72]

LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]

ROLLING_WINDOWS = [6, 24, 72]

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
]

AQI_TARGET = "us_aqi"

POLLUTANT_VARS = ["us_aqi", "pm2_5", "pm10", "carbon_monoxide",
                   "nitrogen_dioxide", "ozone", "sulphur_dioxide"]

FEATURE_STORE_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
