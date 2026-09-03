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
import plotly.graph_objects as go
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

STATUS_COLOR = {"success": "#639922", "partial": "#BA7517", "failed": "#993C1D"}

# (low, high, category) -> used for both the gauge bands and the sidebar legend
AQI_BANDS = [
    (0, 50, "Good"),
    (50, 100, "Moderate"),
    (100, 150, "Unhealthy for Sensitive Groups"),
    (150, 200, "Unhealthy"),
    (200, 300, "Very Unhealthy"),
    (300, 400, "Hazardous"),
]


def hex_to_rgba(hex_color, alpha):
    """Plotly's color properties only accept 6-digit hex or rgb()/rgba()
    strings -- NOT 8-digit alpha-hex like '#63992230'. This converts a
    plain hex color + a 0-1 alpha into a valid rgba() string instead."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def svg_dot(color, size=10):
    """A small inline SVG circle used as a status/legend marker in place
    of emoji -- renders identically across every OS/browser, unlike
    emoji glyphs which vary by platform font."""
    r = size / 2
    return (f'<svg width="{size}" height="{size}" style="vertical-align:middle;">'
            f'<circle cx="{r}" cy="{r}" r="{r}" fill="{color}" /></svg>')


def build_gauge(value, category):
    """A semicircle gauge for the current AQI reading, colored by
    category, with the full EPA band structure shown as background
    zones so the reading is immediately contextualized."""
    color = CATEGORY_COLOR.get(category, "#888780")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value if value is not None else 0,
        number={"font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 400], "tickcolor": "#888", "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [{"range": [lo, hi], "color": hex_to_rgba(CATEGORY_COLOR.get(cat, "#888780"), 0.19)}
                      for lo, hi, cat in AQI_BANDS],
        },
    ))
    fig.update_layout(
        height=210, margin=dict(l=25, r=25, t=15, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_trend_chart(recent_history, forecasts, current_time, current_aqi):
    """Solid line for observed history, dashed line + markers for the
    forecast, sharing the current point so they connect visually, with
    a 'now' marker separating the two."""
    fig = go.Figure()

    if recent_history:
        hist_df = pd.DataFrame(recent_history)
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
        fig.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["us_aqi"],
            mode="lines", name="Observed",
            line=dict(color="#4FA8E0", width=2.2),
        ))

    if forecasts and current_time is not None:
        fx = [pd.to_datetime(current_time)] + [pd.to_datetime(f["target_time"]) for f in forecasts]
        fy = [current_aqi] + [f["predicted_aqi"] for f in forecasts]
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines+markers", name="Forecast",
            line=dict(color="#F2A93B", width=2.2, dash="dash"),
            marker=dict(size=9, color="#F2A93B", line=dict(width=1.5, color="white")),
        ))
        # NOTE: fig.add_vline() is intentionally NOT used here -- with
        # pandas>=2.x + a datetime x-value, its internal annotation-
        # positioning code performs Timestamp + int arithmetic that
        # pandas no longer allows, raising "Addition/subtraction of
        # integers and integer-arrays with Timestamp is no longer
        # supported." Building the line + label manually via add_shape/
        # add_annotation avoids that code path entirely.
        now_ts = pd.to_datetime(current_time)
        fig.add_shape(
            type="line", x0=now_ts, x1=now_ts, y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color="#888"),
        )
        fig.add_annotation(
            x=now_ts, y=1, yref="paper", yanchor="bottom",
            text="now", showarrow=False, font=dict(color="#888", size=11),
        )

    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="AQI", showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    )
    return fig


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

    now_ts = inf.get_now_anchor(df)
    observed = aqi[aqi["timestamp"] <= now_ts].dropna(subset=["us_aqi"]).sort_values("timestamp")
    current_aqi = float(observed["us_aqi"].iloc[-1]) if len(observed) else None

    conditions = {}
    current_row = df[df["timestamp"] == now_ts]
    if len(current_row):
        r = current_row.iloc[0]
        for key in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pm2_5", "pm10"):
            val = r.get(key)
            conditions[key] = round(float(val), 2) if pd.notna(val) else None

    result = {
        "city": city,
        "current_aqi": current_aqi,
        "current_category": inf.aqi_category(current_aqi),
        "current_time": str(observed["timestamp"].iloc[-1]) if len(observed) else None,
        "conditions": conditions,
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
    cat = city_result["current_category"]
    color = CATEGORY_COLOR.get(cat, "#888780")

    gauge_col, info_col = st.columns([1, 2])
    with gauge_col:
        st.plotly_chart(build_gauge(city_result["current_aqi"], cat),
                         use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<span style='color:{color};font-weight:600;font-size:16px;'>{cat}</span><br>"
            f"<span style='color:gray;font-size:12px;'>as of {city_result['current_time']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if cat in ("Unhealthy", "Very Unhealthy", "Hazardous"):
            st.error(f"Hazardous AQI alert: {cat}")

    with info_col:
        if city_result["forecasts"]:
            fcols = st.columns(len(city_result["forecasts"]))
            for i, fc in enumerate(city_result["forecasts"]):
                with fcols[i]:
                    delta = None
                    if city_result["current_aqi"] is not None:
                        delta = round(fc["predicted_aqi"] - city_result["current_aqi"], 1)
                    st.metric(
                        label=f"+{fc['horizon_hours']}h",
                        value=fc["predicted_aqi"],
                        delta=delta,
                        # AQI increasing is bad -> show increases in red, not green
                        delta_color="inverse",
                    )
                    st.caption(f"{fc['category']}  \u00b7  {fc['target_time'][:16]}")
        else:
            st.info("No trained model available yet for this city -- run "
                    "training_pipeline.py at least once.")

    # Current conditions get the FULL page width (not squeezed into the
    # 2/3-width info_col above) -> each of the 5 metrics gets enough
    # room that values like "30.80 \u00b0C" no longer truncate with "...".
    cond = city_result.get("conditions", {})
    if cond:
        st.markdown("&nbsp;")
        st.caption("Current conditions")
        ccols = st.columns(5)
        fields = [
            ("Temp", "temperature_2m", "\u00b0C"),
            ("Humidity", "relative_humidity_2m", "%"),
            ("Wind", "wind_speed_10m", "m/s"),
            ("PM2.5", "pm2_5", "\u00b5g/m\u00b3"),
            ("PM10", "pm10", "\u00b5g/m\u00b3"),
        ]
        for col, (label, key, unit) in zip(ccols, fields):
            val = cond.get(key)
            col.metric(label, f"{val:.2f} {unit}" if val is not None else "\u2014")

        if city_result["forecasts"]:
            st.markdown("&nbsp;")
            fcols = st.columns(len(city_result["forecasts"]))
            for i, fc in enumerate(city_result["forecasts"]):
                with fcols[i]:
                    delta = None
                    if city_result["current_aqi"] is not None:
                        delta = round(fc["predicted_aqi"] - city_result["current_aqi"], 1)
                    st.metric(
                        label=f"+{fc['horizon_hours']}h",
                        value=fc["predicted_aqi"],
                        delta=delta,
                        # AQI increasing is bad -> show increases in red, not green
                        delta_color="inverse",
                    )
                    st.caption(f"{fc['category']}  \u00b7  {fc['target_time'][:16]}")
        else:
            st.info("No trained model available yet for this city -- run "
                    "training_pipeline.py at least once.")

    if city_result["recent_history"] or city_result["forecasts"]:
        st.plotly_chart(
            build_trend_chart(city_result["recent_history"], city_result["forecasts"],
                               city_result["current_time"], city_result["current_aqi"]),
            use_container_width=True, config={"displayModeBar": False},
        )


# ---- Page ----

if "cache_bust" not in st.session_state:
    st.session_state.cache_bust = 0

top_left, top_mid, top_right = st.columns([2, 2, 1])
with top_left:
    st.title("10Pearls AQI Predictor")
    st.caption("Air quality forecasts for the next 24h / 48h / 72h")
with top_right:
    st.write("")  # vertical spacer to align button with selectbox
    refresh_clicked = st.button("Refresh now")

if refresh_clicked:
    get_prediction_for_city.clear()
    load_reports.clear()
    st.session_state.cache_bust += 1

reports = load_reports(st.session_state.cache_bust)

with st.sidebar:
    st.markdown("### Pearls AQI Predictor")
    st.caption(
        "24h / 48h / 72h AQI forecasts for Karachi, Lahore, and Islamabad. "
        "Live inference, best-of-3 models per horizon, storage backend: "
        f"**{reports['backend']}**."
    )
    st.divider()
    st.markdown("**EPA AQI scale**")
    for lo, hi, cat in AQI_BANDS:
        c = CATEGORY_COLOR.get(cat, "#888780")
        label = f"{lo}\u2013{hi}" if hi < 400 else f"{lo}+"
        st.markdown(
            f"{svg_dot(c)} **{label}** &nbsp;{cat}",
            unsafe_allow_html=True,
        )
    if reports["training_summary"]:
        st.divider()
        st.markdown("**Current models**")
        for horizon_label, info in sorted(reports["training_summary"].items()):
            st.markdown(
                f"**{horizon_label}** \u2014 {info['selected_model']} "
                f"(R\u00b2 {info['metrics']['r2']:.2f})"
            )

selected_city = st.selectbox("City", list(CITIES.keys()))

with st.spinner(f"Loading {selected_city}'s forecast..."):
    city_result = get_prediction_for_city(
        selected_city, CITIES[selected_city], st.session_state.cache_bust
    )

render_city(city_result)

with st.expander("What drives these predictions? (feature importance)"):
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

with st.expander("Pipeline status / logs"):
    st.caption(
        "Lightweight status written by each pipeline's last run. For "
        "full step-by-step logs, see your repo's GitHub Actions tab "
        "(Actions -> pick a workflow run)."
    )
    for pipeline_name, status in reports["run_statuses"].items():
        if status is None:
            st.markdown(f"{svg_dot('#888780')} **{pipeline_name}**: no run recorded yet",
                        unsafe_allow_html=True)
            continue
        dot_color = STATUS_COLOR.get(status["status"], "#888780")
        st.markdown(
            f"{svg_dot(dot_color)} **{pipeline_name}** \u2014 {status['status']} "
            f"at {status['timestamp']}",
            unsafe_allow_html=True,
        )
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
