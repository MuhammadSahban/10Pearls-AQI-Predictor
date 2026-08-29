"""
Feature importance explanations using SHAP, one plot per horizon.
Run after training_pipeline.py: python src/shap_analysis.py
Plots are saved via ReportStore -> B2 if configured (nothing local/git),
else local_store/reports/ as the fallback.
"""
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from config import HORIZONS, FEATURE_STORE_NAME
from storage import FeatureStore, ModelRegistry, ReportStore, log_run_status
from training_pipeline import build_horizon_dataset


def run():
    fs = FeatureStore()
    mr = ModelRegistry()
    rs = ReportStore()

    df = fs.load_features(name=FEATURE_STORE_NAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    results = {}
    for horizon in HORIZONS:
        try:
            model, feature_cols = mr.load_model_and_columns(f"aqi_model_{horizon}h")
            if model is None or feature_cols is None:
                print(f"[shap] no model for +{horizon}h, skipping")
                results[f"{horizon}h"] = "skipped (no model)"
                continue

            data = build_horizon_dataset(df, horizon)
            data = pd.get_dummies(data, columns=["city"], prefix="city")
            data = data.dropna(subset=["target"] + feature_cols)
            sample = data[feature_cols].sample(min(200, len(data)), random_state=42)

            # pd.get_dummies() produces bool-dtype columns for the city
            # one-hot encoding. Mixed bool+float64 columns in the same
            # DataFrame make SHAP's tree background-data array come out
            # as dtype=object internally, which its C extension can't
            # safely cast to float64 -> "Found a NULL input array" /
            # "Cannot cast array data from dtype('O') to dtype('float64')".
            # Casting everything to float64 up front avoids this for
            # every model type (Ridge/RandomForest/GradientBoosting all
            # need pure-numeric input here regardless).
            sample = sample.astype(np.float64)

            # shap.Explainer (the unified API) auto-dispatches to the
            # right algorithm for whatever model type won this horizon
            # -> TreeExplainer for RandomForest/GradientBoosting,
            # LinearExplainer for Ridge, etc. This matters here
            # specifically because training_pipeline.py picks a
            # DIFFERENT model per horizon based on which has the best
            # RMSE, so this script can't assume it's always a tree model.
            model_type = type(model).__name__
            explainer = shap.Explainer(model, sample)
            explanation = explainer(sample)
            print(f"[shap]   +{horizon}h: model={model_type}, "
                  f"explainer={type(explainer).__name__}")

            plt.figure()
            shap.plots.bar(explanation, show=False)
            plt.tight_layout()

            # Render to an in-memory buffer rather than a local file, so
            # this never touches disk -> goes straight to ReportStore
            # (B2 if configured, local_store/reports/ otherwise).
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120)
            plt.close()
            rs.save_bytes(f"shap_{horizon}h.png", buf.getvalue())
            print(f"[shap] saved shap_{horizon}h.png via ReportStore (backend: {rs.backend})")
            results[f"{horizon}h"] = f"ok ({model_type})"

        except Exception as e:
            # One horizon's model failing SHAP (unexpected model type,
            # a version incompatibility you can't reproduce locally,
            # etc.) shouldn't take down the plots for the other two
            # horizons -- log it clearly and keep going.
            print(f"[shap] FAILED for +{horizon}h (model type: "
                  f"{type(model).__name__ if 'model' in dir() else 'unknown'}): "
                  f"{type(e).__name__}: {e}")
            results[f"{horizon}h"] = f"failed: {type(e).__name__}: {e}"
            plt.close("all")
            continue

    return results


if __name__ == "__main__":
    try:
        result = run()
        overall = "success" if all(v.startswith("ok") for v in result.values()) else "partial"
        log_run_status("shap_analysis", overall, result)
    except Exception as e:
        log_run_status("shap_analysis", "failed", {"error": str(e)})
        raise
