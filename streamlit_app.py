"""
Pearls AQI Predictor dashboard.

Design choice: this app computes predictions LIVE on each load (cached
for CACHE_TTL_SECONDS), rather than only reading a static JSON file that
some other process wrote earlier. Why: when HOPSWORKS_API_KEY is set,
storage.ModelRegistry.load_model_and_columns() always fetches the newest
model version straight from Hopsworks, so as soon as
training_pipeline.py finishes its daily run and pushes a new version,
the NEXT time this app's cache expires it will automatically start
serving the new model's predictions -- no redeploy, no manual step.

If you're not using Hopsworks, this still works: it just loads whatever
model is sitting in local_store/ (which GitHub Actions commits back to
the repo after each training run -- see .github/workflows/training_pipeline.yml).
"""
import sys
import pathlib
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from config import CITIES  # noqa: E402
import inference_pipeline as inf  # noqa: E402
from storage import ModelRegistry, ReportStore  # noqa: E402

st.set_page_config(page_title="10Pearls AQI Predictor", page_icon="🌫️", layout="wide")

CACHE_TTL_SECONDS = 60 * 60  # re-check for a new model / new data hourly

CATEGORY_COLOR = {
    "Good": "#639922",
    "Moderate": "#BA7517",
    "Unhealthy for Sensitive Groups": "#D85A30",
    "Unhealthy": "#993C1D",
    "Very Unhealthy": "#A32D2D",
    "Hazardous": "#501313",
    "Unknown": "#888780",
}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_prediction_for_city(city, coords, _cache_bust):
    """One city at a time, so switching cities in the picker doesn't
    force recomputation of cities you're not currently viewing.
    `_cache_bust` is unused inside but lets the Refresh button force a
    fresh computation by changing its value (Streamlit caches on args)."""
    mr = ModelRegistry()

    weather = inf.fetch_weather(coords["lat"], coords["lon"], past_days=3, forecast_days=4)
    aqi = inf.fetch_air_quality(coords["lat"], coords["lon"], past_days=3, forecast_days=4)
    df = pd.merge(weather, aqi, on="timestamp", how="inner")
    df["city"] = city
    df = inf.add_time_features(df)
    df = inf.add_lag_features(df, "us_aqi")

    observed = aqi.dropna(subset=["us_aqi"]).sort_values("timestamp")
    current_aqi = float(observed["us_aqi"].iloc[-1]) if len(observed) else None

    result = {
        "city": city,
        "current_aqi": current_aqi,
        "current_category": inf.aqi_category(current_aqi),
        "current_time": str(observed["timestamp"].iloc[-1]) if len(observed) else None,
        "forecasts": [],
        "recent_history": observed.tail(72)[["timestamp", "us_aqi"]]
            .assign(timestamp=lambda d: d["timestamp"].astype(str))
            .to_dict(orient="records"),
    }

    for horizon in [24, 48, 72]:
        model, feature_cols = mr.load_model_and_columns(f"aqi_model_{horizon}h")
        if model is None or feature_cols is None:
            continue
        X, now_ts, target_ts = inf.get_feature_row(df, horizon, feature_cols)
        pred = float(model.predict(X)[0])
        result["forecasts"].append({
            "horizon_hours": horizon,
            "target_time": str(target_ts),
            "predicted_aqi": round(pred, 1),
            "category": inf.aqi_category(pred),
        })

    return result


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_reports(_cache_bust):
    """One shared, cached read of everything ReportStore holds: SHAP
    plot bytes, training summary, and each pipeline's last-run status.
    This is the answer to "how does the dashboard see files that only
    exist in the cloud (or only get committed to git)?" -> it doesn't
    read local_store/ paths directly at all anymore. It goes through
    the exact same ReportStore that feature_pipeline.py,
    training_pipeline.py, and shap_analysis.py write to -> if B2 is
    configured, that's B2 for everyone; if not, it's local_store/reports/,
    which is what GitHub Actions commits back into the repo that this
    Streamlit app is deployed from (same checkout, same files)."""
    rs = ReportStore()
    shap_images = {}
    for horizon in [24, 48, 72]:
        data = rs.load_bytes(f"shap_{horizon}h.png")
        if data:
            shap_images[horizon] = data
    training_summary = rs.load_json("training_summary.json")
    run_statuses = {}
    for pipeline in ["feature_pipeline", "training_pipeline", "shap_analysis"]:
        run_statuses[pipeline] = rs.load_json(f"status_{pipeline}.json")
    return {
        "backend": rs.backend,
        "shap_images": shap_images,
        "training_summary": training_summary,
        "run_statuses": run_statuses,
    }


def render_city(city_result):
    left, right = st.columns([1, 3])

    with left:
        cat = city_result["current_category"]
        color = CATEGORY_COLOR.get(cat, "#888780")
        st.markdown(
            f"<div style='padding:14px;border-radius:8px;background:{color}22;"
            f"border:1px solid {color};'>"
            f"<div style='font-size:13px;color:gray;'>Current AQI</div>"
            f"<div style='font-size:32px;font-weight:600;'>{city_result['current_aqi']}</div>"
            f"<div style='color:{color};font-weight:500;'>{cat}</div>"
            f"<div style='font-size:12px;color:gray;'>as of {city_result['current_time']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if cat in ("Unhealthy", "Very Unhealthy", "Hazardous"):
            st.error(f"⚠️ Hazardous AQI alert: {cat}")

    with right:
        if not city_result["forecasts"]:
            st.info("No trained model available yet for this city — run "
                    "training_pipeline.py at least once.")
        else:
            fcols = st.columns(len(city_result["forecasts"]))
            for i, fc in enumerate(city_result["forecasts"]):
                with fcols[i]:
                    color = CATEGORY_COLOR.get(fc["category"], "#888780")
                    st.markdown(f"**+{fc['horizon_hours']}h**")
                    st.markdown(
                        f"<div style='padding:10px;border-radius:8px;background:{color}22;"
                        f"border:1px solid {color};'>"
                        f"<div style='font-size:22px;font-weight:600;'>{fc['predicted_aqi']}</div>"
                        f"<div style='color:{color};font-size:13px;'>{fc['category']}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(fc["target_time"])

        if city_result["recent_history"]:
            hist_df = pd.DataFrame(city_result["recent_history"])
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
            st.line_chart(hist_df.set_index("timestamp")["us_aqi"], height=220)


# ---- Page ----

st.title("10Pearls AQI Predictor")
st.caption("Air quality forecasts for the next 24h / 48h / 72h")

top_left, top_mid, top_right = st.columns([2, 2, 1])
with top_left:
        selected_city = st.selectbox("City", list(CITIES.keys()))
with top_right:
    st.write("")  # vertical spacer to align button with selectbox
    refresh_clicked = st.button("🔄 Refresh now")

if "cache_bust" not in st.session_state:
    st.session_state.cache_bust = 0
if refresh_clicked:
    get_prediction_for_city.clear()
    load_reports.clear()
    st.session_state.cache_bust += 1

with st.spinner(f"Loading {selected_city}'s forecast..."):
    city_result = get_prediction_for_city(
        selected_city, CITIES[selected_city], st.session_state.cache_bust
    )

render_city(city_result)

reports = load_reports(st.session_state.cache_bust)

with st.expander("📊 What drives these predictions? (feature importance)"):
    st.caption(
        "Computed once daily right after training (not live -- SHAP is too "
        "slow to recompute on every page load), so this may lag the very "
        "latest model by up to a day. Read from the same storage backend "
        f"your pipelines write to (currently: **{reports['backend']}**)."
    )
    if reports["shap_images"]:
        for horizon, img_bytes in sorted(reports["shap_images"].items()):
            st.image(img_bytes, caption=f"+{horizon}h model")
    else:
        st.write("Not generated yet — run `python src/shap_analysis.py` after training.")

with st.expander("🛠️ Pipeline status / logs"):
    st.caption(
        "Lightweight status written by each pipeline's last run. For "
        "full step-by-step logs, see your repo's GitHub Actions tab "
        "(Actions -> pick a workflow run)."
    )
    for pipeline_name, status in reports["run_statuses"].items():
        if status is None:
            st.write(f"**{pipeline_name}**: no run recorded yet")
            continue
        icon = "✅" if status["status"] == "success" else (
            "⚠️" if status["status"] == "partial" else "❌"
        )
        st.write(f"{icon} **{pipeline_name}** — {status['status']} at {status['timestamp']}")
        if status.get("details"):
            st.json(status["details"], expanded=False)

    if reports["training_summary"]:
        st.write("**Latest training summary** (best model per horizon):")
        summary_rows = []
        for horizon_label, info in reports["training_summary"].items():
            summary_rows.append({
                "horizon": horizon_label,
                "selected_model": info["selected_model"],
                "rmse": round(info["metrics"]["rmse"], 2),
                "r2": round(info["metrics"]["r2"], 3),
            })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Data: Open-Meteo Weather & Air Quality APIs. Models: best-of-3 "
    "(RandomForest / GradientBoosting / Ridge) per horizon, trained on "
    "lagged AQI + forecasted weather features, selected by RMSE. "
    "Predictions refresh automatically ~hourly and pick up new model "
    "versions after each daily retrain. This is a personal/educational "
    "project, not a substitute for official air quality advisories."
)
