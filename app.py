# -*- coding: utf-8 -*-
"""churn_prediction_app.py - Standalone Churn Prediction App"""

import os, sys, json
import streamlit as st
from joblib import load, dump
import numpy as np
import pandas as pd

# ---- Auto-export: bundle models if .joblib not found ----
BUNDLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "churn_model_bundle.joblib")
PROJECT_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def _needs_bundle_refresh(bundle_path, model_dir):
    if not os.path.exists(bundle_path):
        return True
    bundle_mtime = os.path.getmtime(bundle_path)
    required = [
        "model_lr.pkl", "model_knn.pkl", "model_dt.pkl", "model_rf.pkl",
        "model_voting.pkl", "scaler.pkl", "encoder_info.pkl",
        "optimal_thresholds.pkl", "metrics.json",
    ]
    for fn in required:
        p = os.path.join(model_dir, fn)
        if os.path.exists(p) and os.path.getmtime(p) > bundle_mtime:
            return True
    return False

if os.path.exists(PROJECT_MODELS) and _needs_bundle_refresh(BUNDLE_PATH, PROJECT_MODELS):
    bundle = {}
    model_files = {
        "Logistic Regression": "model_lr.pkl", "KNN": "model_knn.pkl",
        "Decision Tree": "model_dt.pkl", "Random Forest": "model_rf.pkl",
        "Ensemble Voting": "model_voting.pkl",
    }
    for name, fn in model_files.items():
        p = os.path.join(PROJECT_MODELS, fn)
        if os.path.exists(p):
            bundle[name] = load(p)
    for key, fn in [("scaler", "scaler.pkl"), ("encoder_info", "encoder_info.pkl"),
                    ("healthy_profile", "healthy_profile.pkl"), ("optimal_thresholds", "optimal_thresholds.pkl")]:
        p = os.path.join(PROJECT_MODELS, fn)
        if os.path.exists(p):
            bundle[key] = load(p)
    metrics_path = os.path.join(PROJECT_MODELS, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            bundle["metrics"] = json.load(f)
    dump(bundle, BUNDLE_PATH, compress=3)

# ---- Load the model bundle ----
if not os.path.exists(BUNDLE_PATH):
    st.error(
        "Model bundle not found. Run `python train.py` in the project "
        "root first, or copy `churn_model_bundle.joblib` into this folder.")
    st.stop()

try:
    bundle = load(BUNDLE_PATH)
except Exception as exc:
    st.error(f"Failed to load model bundle: {exc}")
    st.stop()

required_keys = ["scaler", "encoder_info"]
missing_keys = [k for k in required_keys if k not in bundle]
if missing_keys:
    st.error(
        "Model bundle is incomplete. Missing key(s): "
        + ", ".join(missing_keys))
    st.stop()

models = {k: v for k, v in bundle.items() if k in [
    "Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Ensemble Voting"]}
if not models:
    st.error("No trained models were found in the bundle.")
    st.stop()

scaler = bundle["scaler"]
enc = bundle["encoder_info"]
metrics = bundle.get("metrics", {})
champion = metrics.get("_champion", "Random Forest")
if champion not in models:
    champion = list(models.keys())[0]
thresholds = bundle.get("optimal_thresholds", {})

BASELINE_MODEL = "Logistic Regression"
BASE_MODEL_NAMES = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]

def _model_label(name):
    tags = []
    if name == champion:
        tags.append("Champion")
    if name == BASELINE_MODEL:
        tags.append("Baseline")
    if name == "Ensemble Voting":
        tags.append("Extra - Team Collaboration")
    if tags:
        return f"{name} ({', '.join(tags)})"
    return name

# ---- Feature config ----
NUM_FEATURES_RAW = ["tenure", "MonthlyCharges", "TotalCharges"]
NUM_FEATURES = NUM_FEATURES_RAW + ["AvgMonthlySpend", "ChargeRatio", "ServiceCount"]
CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]
CAT_OPTIONS = {
    "gender": ["Male", "Female"], "SeniorCitizen": ["No", "Yes"],
    "Partner": ["Yes", "No"], "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"], "PaperlessBilling": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

# ---- Build input ----
def prepare_input(ud):
    encoded = enc["encoded_feature_names"]
    row = pd.DataFrame(np.zeros((1, len(encoded))), columns=encoded)
    for f in NUM_FEATURES_RAW:
        if f in row.columns:
            row[f] = ud[f]
    t, m, tc = ud["tenure"], ud["MonthlyCharges"], ud["TotalCharges"]
    if "AvgMonthlySpend" in row.columns: row["AvgMonthlySpend"] = tc / (t + 1)
    if "ChargeRatio" in row.columns: row["ChargeRatio"] = m / (tc + 1)
    if "ServiceCount" in row.columns:
        row["ServiceCount"] = sum(1 for c in ["PhoneService","MultipleLines","InternetService",
            "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
            if ud.get(c) in ("Yes","Fiber optic","DSL"))
    if "HasProtectionBundle" in row.columns:
        row["HasProtectionBundle"] = int(all(ud.get(x)=="Yes" for x in ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport"]))
    if "HighRiskContract" in row.columns:
        row["HighRiskContract"] = int(ud.get("Contract")=="Month-to-month" and ud.get("PaymentMethod")=="Electronic check")
    if "HasStreaming" in row.columns:
        row["HasStreaming"] = int(ud.get("StreamingTV")=="Yes" or ud.get("StreamingMovies")=="Yes")
    for f in CAT_FEATURES:
        col = f"{f}_{ud[f]}"
        if col in row.columns:
            row[col] = 1
    sel = enc.get("selected_features")
    if sel is not None:
        row = row[sel]
    return row

# ---- Streamlit App ----
st.title("📱 Telco Customer Churn Prediction")

# Collect user inputs
st.sidebar.header("Customer Information")
ud = {}
ud["tenure"] = st.sidebar.slider("Tenure (months)", 0, 72, 12, 1)
ud["MonthlyCharges"] = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
ud["TotalCharges"] = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1500.0, 10.0)
for f in CAT_FEATURES:
    ud[f] = st.sidebar.selectbox(f, CAT_OPTIONS[f])

# Predict
if st.button("Predict Churn Risk"):
    inp = prepare_input(ud)
    results = []
    for name, model in models.items():
        prob = float(model.predict_proba(inp)[0][1])
        thr = float(thresholds.get(name, 0.5))
        results.append({"Model": _model_label(name),
                        "Churn Probability": f"{prob*100:.1f}%",
                        "Threshold": f"{thr:.3f}",
                        "Prediction": "Churn" if prob >= thr else "Stay"})

    champ_prob = float(models[champion].predict_proba(inp)[0][1])
    champ_thr = float(thresholds.get(champion, 0.5))
    if champ_prob >= champ_thr:
        st.error(f"⚠️ HIGH RISK — {champion} predicts {champ_prob*100:.1f}% churn probability")
    else:
        st.success(f"✅ LOW RISK — {champion} predicts {champ_prob*100:.1f}% churn probability")

    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
