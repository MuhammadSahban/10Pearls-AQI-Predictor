"""
Persistence baseline check.

Answers one question per horizon: "does my trained model actually beat
just guessing that AQI doesn't change?" -> computes the naive baseline
(predicted AQI(t+horizon) = AQI(t)) on the EXACT SAME time-based test
split your real training used, so the comparison is apples-to-apples,
not a rough guess.

Run after training_pipeline.py: python src/baseline_check.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from config import HORIZONS, FEATURE_STORE_NAME
from storage import FeatureStore,ModelRegistry,ReportStore,log_run_status
from training_pipeline import build_horizon_dataset


def evaluate_persistence_baseline(df, horizon, split_frac=0.85):
    """Same dataset construction and same time-based split as
    train_and_select_best() in training_pipeline.py, so the test rows
    are IDENTICAL to what your real model was scored on."""
    data = build_horizon_dataset(df, horizon)
    data = pd.get_dummies(data,columns=["city"],prefix="city")
    feature_cols = [column for column in data.columns if column not in ("timestamp", "target")]
    data = data.dropna(subset=["target"] + feature_cols)
    data = data.sort_values("timestamp")

    split_idx = int(len(data) * split_frac)
    test = data.iloc[split_idx:]

    # The naive forecast: "AQI at t+horizon will be whatever it is right
    # now" -> that's literally the un-shifted us_aqi column on these rows.
    baseline_pred = test["us_aqi"]
    y_test = test["target"]

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test,baseline_pred))),
        "mae": float(mean_absolute_error(y_test,baseline_pred)),
        "r2": float(r2_score(y_test,baseline_pred)),
        "n_test": int(len(test)),
    }


def run():
    fs = FeatureStore()
    mr = ModelRegistry()
    rs = ReportStore()

    df = fs.load_features(name=FEATURE_STORE_NAME)
    if df.empty:
        raise RuntimeError("No features found. Run feature_pipeline.py first.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    results = {}
    for horizon in HORIZONS:
        baseline = evaluate_persistence_baseline(df,horizon)

        model_metrics = None
        summary = rs.load_json("training_summary.json")
        if summary and f"{horizon}h" in summary:
            model_metrics = summary[f"{horizon}h"]["metrics"]

        print(f"\n[baseline] +{horizon}h  (n_test={baseline['n_test']})")
        print(f"[baseline]   persistence baseline: RMSE={baseline['rmse']:.2f}  "
              f"MAE={baseline['mae']:.2f}  R2={baseline['r2']:.3f}")

        entry = {"persistence_baseline": baseline}

        if model_metrics:
            improvement_pct = 100 * (baseline["rmse"] - model_metrics["rmse"]) / baseline["rmse"]
            verdict = (
                "BEATS baseline meaningfully" if improvement_pct > 10 else
                "barely beats baseline" if improvement_pct > 0 else
                "DOES NOT beat baseline -- model adds no value over persistence"
            )
            print(f"[baseline]   your model ({model_metrics.get('selected_model', '?')}): "
                  f"RMSE={model_metrics['rmse']:.2f}  R2={model_metrics['r2']:.3f}")
            print(f"[baseline]   RMSE improvement over baseline: {improvement_pct:+.1f}%  -> {verdict}")
            entry["trained_model"] = model_metrics
            entry["rmse_improvement_pct"] = round(improvement_pct, 1)
            entry["verdict"] = verdict
        else:
            print(f"[baseline]   no trained model metrics found for +{horizon}h yet "
                  f"(run training_pipeline.py first for a full comparison)")
            entry["verdict"] = "no trained model to compare yet"

        results[f"{horizon}h"] = entry

    rs.save_json("baseline_comparison.json", results)
    print(f"\n[baseline] saved comparison via ReportStore (backend: {rs.backend})")
    return results


if __name__ == "__main__":
    try:
        result = run()
        log_run_status("baseline_check", "success", result)
    except Exception as e:
        log_run_status("baseline_check", "failed", {"error": str(e)})
        raise
