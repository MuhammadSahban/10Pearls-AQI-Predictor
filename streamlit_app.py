"""
10Pearls AQI Predictor dashboard.

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
import base64

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from config import CITIES  # noqa: E402
import inference_pipeline as inf  # noqa: E402
from storage import ModelRegistry, ReportStore  # noqa: E402

st.set_page_config(page_title="10Pearls AQI Predictor", layout="wide")

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

AQI_BANDS = [
    (0, 50, "Good"),
    (50, 100, "Moderate"),
    (100, 150, "Unhealthy for Sensitive Groups"),
    (150, 200, "Unhealthy"),
    (200, 300, "Very Unhealthy"),
    (300, 400, "Hazardous"),
]

POLLUTANT_FIELDS = [
    ("pm2_5", "PM2.5"), ("pm10", "PM10"), ("nitrogen_dioxide", "NO\u2082"),
    ("ozone", "O\u2083"), ("sulphur_dioxide", "SO\u2082"), ("carbon_monoxide", "CO"),
]

THEMES = {
    "light": {"bg": "#FFFFFF", "text": "#1A1A1A", "muted": "#6B7280",
              "card": "#F7F7F8", "border": "#E3E3E6", "grid": "rgba(0,0,0,0.08)"},
    "dark": {"bg": "#0E1117", "text": "#E8E8E8", "muted": "#9CA3AF",
             "card": "#1B1F27", "border": "#2C313C", "grid": "rgba(255,255,255,0.10)"},
}


def hex_to_rgba(hex_color, alpha):
    """Plotly's color properties only accept 6-digit hex or rgb()/rgba()
    strings -- NOT 8-digit alpha-hex like '#63992230'."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def svg_dot(color, size=10):
    r = size / 2
    return (f'<svg width="{size}" height="{size}" style="vertical-align:middle;">'
            f'<circle cx="{r}" cy="{r}" r="{r}" fill="{color}" /></svg>')


def init_theme():
    """Default to the browser/device's current color-scheme preference
    on first load, remembered afterward via session_state.

    IMPORTANT: this must return immediately if session_state.theme is
    already set, from EITHER the initial query-param handoff below OR a
    later manual toggle -- otherwise, since the query param still holds
    whatever the page loaded with, re-reading it on every rerun would
    silently overwrite a manual toggle back to the stale original value
    on the very next interaction."""
    if "theme" in st.session_state:
        return
    qp_theme = st.query_params.get("theme")
    if qp_theme in ("light", "dark"):
        st.session_state.theme = qp_theme
        return
    st.session_state.theme = "light"  # sensible fallback before JS reports back
    components.html(
        """
        <script>
        const params = new URLSearchParams(window.parent.location.search);
        if (!params.has('theme')) {
            const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            params.set('theme', dark ? 'dark' : 'light');
            window.parent.location.search = params.toString();
        }
        </script>
        """,
        height=0,
    )


def inject_theme_css(theme):
    """The previous version only styled the outer container + metric
    boxes -- Streamlit's own title/caption/button/selectbox text is
    rendered using STREAMLIT'S OWN internal theme CSS variables
    (--text-color, --background-color, etc.), which our earlier CSS
    never touched. That's why some text stayed dark-on-dark (title,
    captions, labels) while other elements we DID directly control
    (Plotly charts, our custom divs) switched correctly.

    Fix: override Streamlit's actual CSS custom properties at :root --
    that's the same mechanism Streamlit's own config.toml theme setting
    uses internally, so every native widget that reads var(--text-color)
    etc. picks up the new value automatically. Explicit !important
    overrides are added on top for the specific elements (title,
    caption, buttons, selectbox) confirmed broken, as a safety net."""
    p = THEMES[theme]
    st.markdown(
        f"""
        <style>
        :root {{
            --primary-color: #F2A93B;
            --background-color: {p['bg']};
            --secondary-background-color: {p['card']};
            --text-color: {p['text']};
        }}
        .stApp, .stAppHeader {{
            background-color: {p['bg']} !important;
            color: {p['text']} !important;
            transition: background-color 0.35s ease, color 0.35s ease;
        }}
        [data-testid="stSidebar"] {{
            background-color: {p['card']} !important;
            transition: background-color 0.35s ease;
        }}
        [data-testid="stSidebar"] * {{
            color: {p['text']} !important;
        }}
        h1, h2, h3, .stMarkdown, .stMarkdown p, label,
        [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] {{
            color: {p['text']} !important;
        }}
        .stButton > button {{
            background-color: {p['card']} !important;
            color: {p['text']} !important;
            border: 1px solid {p['border']} !important;
            transition: background-color 0.35s ease, color 0.35s ease;
        }}
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background-color: {p['card']} !important;
            color: {p['text']} !important;
            border-color: {p['border']} !important;
        }}
        [data-testid="stMetric"] {{
            background-color: {p['card']} !important;
            border: 1px solid {p['border']};
            border-radius: 10px;
            padding: 10px 14px;
            transition: background-color 0.35s ease, border-color 0.35s ease;
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
            color: {p['text']} !important;
        }}
        /* stMetricDelta is intentionally left alone -- it carries its
           own semantic red/green coloring (rising/falling AQI), which
           must not be flattened to the theme's plain text color. */
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_gauge(value, category, theme):
    color = CATEGORY_COLOR.get(category, "#888780")
    axis_color = THEMES[theme]["text"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value if value is not None else 0,
        number={"font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 400], "tickcolor": axis_color, "tickfont": {"size": 10, "color": axis_color}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [{"range": [lo, hi], "color": hex_to_rgba(CATEGORY_COLOR.get(cat, "#888780"), 0.19)}
                      for lo, hi, cat in AQI_BANDS],
        },
    ))
    fig.update_layout(height=210, margin=dict(l=25, r=25, t=15, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", font=dict(color=axis_color))
    return fig


def build_trend_chart(recent_history, forecasts, current_time, current_aqi, theme):
    p = THEMES[theme]
    fig = go.Figure()
    if recent_history:
        hist_df = pd.DataFrame(recent_history)
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
        fig.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df["us_aqi"], mode="lines",
                                  name="Observed", line=dict(color="#4FA8E0", width=2.2)))
    if forecasts and current_time is not None:
        fx = [pd.to_datetime(current_time)] + [pd.to_datetime(f["target_time"]) for f in forecasts]
        fy = [current_aqi] + [f["predicted_aqi"] for f in forecasts]
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines+markers", name="Forecast",
                                  line=dict(color="#F2A93B", width=2.2, dash="dash"),
                                  marker=dict(size=9, color="#F2A93B", line=dict(width=1.5, color="white"))))
        # add_vline() is intentionally NOT used -- with pandas>=2.x + a
        # datetime x-value its internal code does Timestamp + int
        # arithmetic that pandas no longer allows. add_shape/
        # add_annotation avoids that code path entirely.
        now_ts = pd.to_datetime(current_time)
        fig.add_shape(type="line", x0=now_ts, x1=now_ts, y0=0, y1=1, yref="paper",
                      line=dict(dash="dot", color=p["muted"]))
        fig.add_annotation(x=now_ts, y=1, yref="paper", yanchor="bottom", text="now",
                           showarrow=False, font=dict(color=p["muted"], size=11))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=p["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0,
                    font=dict(color=p["text"])),
        xaxis=dict(title=None, showgrid=False, tickfont=dict(color=p["text"])),
        yaxis=dict(title="AQI", showgrid=True, gridcolor=p["grid"],
                   tickfont=dict(color=p["text"]), title_font=dict(color=p["text"])),
    )
    return fig


def build_pollutant_chart(pollutants, theme):
    """Horizontal bar chart of raw pollutant concentrations. Log x-axis
    because CO is typically reported in the hundreds (\u00b5g/m\u00b3)
    while the others are single/double digits -- a linear axis would
    make everything except CO invisible."""
    p = THEMES[theme]
    labels, values, colors = [], [], []
    palette = ["#4FA8E0", "#63A375", "#D8A13B", "#C97BB0", "#E0776A", "#9B8ED6"]
    for i, (key, label) in enumerate(POLLUTANT_FIELDS):
        val = pollutants.get(key)
        if val is not None:
            labels.append(label)
            values.append(max(val, 0.01))  # log axis can't plot 0
            colors.append(palette[i % len(palette)])
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                            text=[f"{v:.2f}" for v in values], textposition="outside",
                            textfont=dict(color=p["text"]),cliponaxis=False))
    fig.update_layout(
        height=300, margin=dict(l=10, r=30, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=p["text"]),
        xaxis=dict(title="\u00b5g/m\u00b3 (log scale)", type="log", gridcolor=p["grid"],
                   tickfont=dict(color=p["text"]), title_font=dict(color=p["text"])),
        yaxis=dict(autorange="reversed", tickfont=dict(color=p["text"])),
    )
    return fig


def build_city_comparison_chart(city_aqis, theme):
    """Current AQI across all three cities side by side, each bar
    colored by its own EPA category."""
    p = THEMES[theme]
    cities = list(city_aqis.keys())
    values = [city_aqis[c]["aqi"] if city_aqis[c]["aqi"] is not None else 0 for c in cities]
    colors = [CATEGORY_COLOR.get(city_aqis[c]["category"], "#888780") for c in cities]
    fig = go.Figure(go.Bar(x=cities, y=values, marker_color=colors,
                            text=[f"{v:.1f}" for v in values], textposition="outside",
                            textfont=dict(color=p["text"]),cliponaxis=False))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=p["text"]),
        xaxis=dict(showgrid=False, tickfont=dict(color=p["text"])),
        yaxis=dict(title="Current AQI", gridcolor=p["grid"],
                   tickfont=dict(color=p["text"]), title_font=dict(color=p["text"])),
    )
    return fig


def render_image_on_white(img_bytes, caption):
    """SHAP plots are pre-rendered PNGs (matplotlib, white background)
    coming from ReportStore/B2 -- their background can't be changed
    after the fact. Rather than let a stark white rectangle float
    directly on a dark theme, wrap it in an explicit white card with
    padding/border so it reads as an intentional design choice in
    both themes, not a visual glitch."""
    b64 = base64.b64encode(img_bytes).decode()
    st.markdown(
        f'<div style="background:#ffffff;padding:14px;border-radius:10px;'
        f'border:1px solid #e3e3e6;margin-bottom:10px;">'
        f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:4px;" />'
        f'<div style="text-align:center;color:#555;font-size:12px;padding-top:6px;">{caption}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_prediction_for_city(city, coords, _cache_bust):
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

    conditions, pollutants = {}, {}
    current_row = df[df["timestamp"] == now_ts]
    if len(current_row):
        r = current_row.iloc[0]
        for key in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pm2_5", "pm10"):
            val = r.get(key)
            conditions[key] = round(float(val), 2) if pd.notna(val) else None
        for key, _ in POLLUTANT_FIELDS:
            val = r.get(key)
            pollutants[key] = round(float(val), 2) if pd.notna(val) else None

    result = {
        "city": city,
        "current_aqi": current_aqi,
        "current_category": inf.aqi_category(current_aqi),
        "current_time": str(observed["timestamp"].iloc[-1]) if len(observed) else None,
        "conditions": conditions,
        "pollutants": pollutants,
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
    return {"backend": rs.backend, "shap_images": shap_images,
            "training_summary": training_summary, "run_statuses": run_statuses}


def render_city(city_result, theme):
    cat = city_result["current_category"]
    color = CATEGORY_COLOR.get(cat, "#888780")

    with st.container(border=True):
        gauge_col, info_col = st.columns([1, 2])
        with gauge_col:
            st.plotly_chart(build_gauge(city_result["current_aqi"], cat, theme),
                             use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span style='color:{color};font-weight:600;font-size:16px;'>{cat}</span><br>"
                f"<span style='color:gray;font-size:12px;'>as of {city_result['current_time']}</span>"
                f"</div>", unsafe_allow_html=True,
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
                        st.metric(label=f"+{fc['horizon_hours']}h", value=fc["predicted_aqi"],
                                  delta=delta, delta_color="inverse")
                        st.caption(f"{fc['category']}  \u00b7  {fc['target_time'][:16]}")
            else:
                st.info("No trained model available yet for this city -- run "
                        "training_pipeline.py at least once.")

    cond = city_result.get("conditions", {})
    if cond:
        st.markdown("&nbsp;")
        with st.container(border=True):
            st.caption("Current conditions")
            ccols = st.columns(5)
            fields = [("Temp", "temperature_2m", "\u00b0C"), ("Humidity", "relative_humidity_2m", "%"),
                      ("Wind", "wind_speed_10m", "m/s"), ("PM2.5", "pm2_5", "\u00b5g/m\u00b3"),
                      ("PM10", "pm10", "\u00b5g/m\u00b3")]
            for col, (label, key, unit) in zip(ccols, fields):
                val = cond.get(key)
                col.metric(label, f"{val:.2f} {unit}" if val is not None else "\u2014")

    chart_col, pollutant_col = st.columns([2, 1])
    with chart_col:
        if city_result["recent_history"] or city_result["forecasts"]:
            with st.container(border=True):
                st.caption("AQI trend: recent history & forecast")
                st.plotly_chart(
                    build_trend_chart(city_result["recent_history"], city_result["forecasts"],
                                       city_result["current_time"], city_result["current_aqi"], theme),
                    use_container_width=True, config={"displayModeBar": False},
                )
    with pollutant_col:
        if city_result.get("pollutants"):
            with st.container(border=True):
                st.caption("Pollutant concentrations")
                st.plotly_chart(build_pollutant_chart(city_result["pollutants"], theme),
                                 use_container_width=True, config={"displayModeBar": False})


# ---- Page ----

init_theme()

if "cache_bust" not in st.session_state:
    st.session_state.cache_bust = 0

top_left, top_mid, top_right = st.columns([2, 2, 1])

# Theme toggle is handled BEFORE inject_theme_css() runs below, and
# deliberately WITHOUT an explicit st.rerun() -- clicking any Streamlit
# widget (this button included) already triggers Streamlit's own
# automatic rerun, same as the Refresh button or the city selector,
# neither of which resets the sidebar. Calling st.rerun() explicitly
# on top of that automatic rerun was the actual cause of the sidebar
# snapping back open on every theme toggle: a forced rerun behaves
# differently from the automatic one every other widget already gets.
# Updating session_state.theme here, before inject_theme_css() is
# called further down in this same script pass, is enough on its own
# to make the CSS reflect the change immediately -- no rerun needed.
def toggle_theme():
    st.session_state.theme = (
        "dark"
        if st.session_state.theme == "light"
        else "light"
    )

inject_theme_css(st.session_state.theme)
with top_right:
    st.write("")
    other_theme = "Dark" if st.session_state.theme == "light" else "Light"
    theme_icon = ":material/dark_mode:" if st.session_state.theme == "light" else ":material/light_mode:"
    if st.button(f"Switch to {other_theme}", icon=theme_icon, use_container_width=True,on_click=toggle_theme):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    refresh_clicked = st.button("Refresh now", icon=":material/refresh:", use_container_width=True)


with top_left:
    st.title("10Pearls AQI Predictor")
    st.caption("Air quality forecasts for the next 24h / 48h / 72h")

if refresh_clicked:
    get_prediction_for_city.clear()
    load_reports.clear()
    st.session_state.cache_bust += 1

reports = load_reports(st.session_state.cache_bust)
theme = st.session_state.theme

with st.sidebar:
    st.markdown("### 10Pearls AQI Predictor")
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
        st.markdown(f"{svg_dot(c)} **{label}** &nbsp;{cat}", unsafe_allow_html=True)
    if reports["training_summary"]:
        st.divider()
        st.markdown("**Current models**")
        for horizon_label, info in sorted(reports["training_summary"].items()):
            st.markdown(f"**{horizon_label}** \u2014 {info['selected_model']} "
                        f"(R\u00b2 {info['metrics']['r2']:.2f})")

selected_city = st.selectbox("City", list(CITIES.keys()))

with st.spinner(f"Loading {selected_city}'s forecast..."):
    city_result = get_prediction_for_city(selected_city, CITIES[selected_city], st.session_state.cache_bust)

render_city(city_result, theme)

# Multi-city comparison -- computes/reuses cached predictions for all
# three cities (not just the selected one), so this is cheap after the
# first load of each city within the current cache window.
st.markdown("&nbsp;")
with st.container(border=True):
    st.caption("Current AQI across all cities")
    city_aqis = {}
    for c, coords in CITIES.items():
        r = get_prediction_for_city(c, coords, st.session_state.cache_bust)
        city_aqis[c] = {"aqi": r["current_aqi"], "category": r["current_category"]}
    st.plotly_chart(build_city_comparison_chart(city_aqis, theme),
                     use_container_width=True, config={"displayModeBar": False})

with st.expander("What drives these predictions? (feature importance)"):
    st.caption(
        "Computed once daily right after training (not live -- SHAP is too "
        "slow to recompute on every page load), so this may lag the very "
        "latest model by up to a day. Read from the same storage backend "
        f"your pipelines write to (currently: **{reports['backend']}**)."
    )
    if reports["shap_images"]:
        for horizon, img_bytes in sorted(reports["shap_images"].items()):
            render_image_on_white(img_bytes, f"+{horizon}h model")
    else:
        st.write("Not generated yet -- run `python src/shap_analysis.py` after training.")

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
        st.markdown(f"{svg_dot(dot_color)} **{pipeline_name}** \u2014 {status['status']} "
                    f"at {status['timestamp']}", unsafe_allow_html=True)
        if status.get("details"):
            st.json(status["details"], expanded=False)

    if reports["training_summary"]:
        st.write("**Latest training summary** (best model per horizon):")
        summary_rows = []
        for horizon_label, info in reports["training_summary"].items():
            summary_rows.append({"horizon": horizon_label, "selected_model": info["selected_model"],
                                  "rmse": round(info["metrics"]["rmse"], 2),
                                  "r2": round(info["metrics"]["r2"], 3)})
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
