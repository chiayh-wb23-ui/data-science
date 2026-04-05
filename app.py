# app.py - BMDS2003

import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib

from config import (
    BASE_DIR, DATA_PATH, MODELS_DIR, PLOTS_DIR,
    TARGET_COL, TARGET_POSITIVE, TARGET_NEGATIVE,
    NUMERIC_FEATURES, NUMERIC_FEATURES_RAW,
    ENGINEERED_NUMERIC_FEATURES, ENGINEERED_BINARY_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    CATEGORY_OPTIONS, NUMERIC_RANGES, FEATURE_LABELS,
    BASE_MODEL_NAMES, BASELINE_MODEL,
    MODEL_NAMES, MODEL_COLORS, MODEL_FILE_KEYS, get_device_info,
)

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    }
    .hero-banner h1 { font-family:'Inter',sans-serif; font-size:2rem; font-weight:700; margin-bottom:0.3rem; letter-spacing:-0.5px; }
    .hero-banner p { font-family:'Inter',sans-serif; font-size:1rem; opacity:0.85; margin:0; }

    .metric-card {
        background:white; border-radius:12px; padding:1.2rem 1.5rem;
        box-shadow:0 2px 12px rgba(0,0,0,0.06); border-left:4px solid #3498db;
        margin-bottom:1rem; transition:transform 0.2s ease;
    }
    .metric-card:hover { transform:translateY(-2px); }
    .metric-card.danger { border-left-color:#e74c3c; }
    .metric-card.success { border-left-color:#27ae60; }
    .metric-card.warning { border-left-color:#f39c12; }
    .metric-card .metric-label { font-family:'Inter',sans-serif; font-size:0.8rem; color:#7f8c8d; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; }
    .metric-card .metric-value { font-family:'Inter',sans-serif; font-size:1.8rem; font-weight:700; color:#2c3e50; margin-top:0.2rem; }

    .result-high-risk {
        background:linear-gradient(135deg,#e74c3c 0%,#c0392b 100%);
        border-radius:16px; padding:1.5rem 2rem; color:white; text-align:center;
        box-shadow:0 4px 20px rgba(231,76,60,0.3); margin-bottom:1rem;
    }
    .result-low-risk {
        background:linear-gradient(135deg,#27ae60 0%,#2ecc71 100%);
        border-radius:16px; padding:1.5rem 2rem; color:white; text-align:center;
        box-shadow:0 4px 20px rgba(39,174,96,0.3); margin-bottom:1rem;
    }
    .result-high-risk h2,.result-low-risk h2 { font-family:'Inter',sans-serif; font-size:1.6rem; font-weight:700; margin:0 0 0.3rem 0; }
    .result-high-risk p,.result-low-risk p { font-family:'Inter',sans-serif; font-size:1rem; opacity:0.9; margin:0; }

    .section-header { font-family:'Inter',sans-serif; font-size:1.3rem; font-weight:600; color:#2c3e50;
        margin-top:1.5rem; margin-bottom:0.8rem; padding-bottom:0.4rem; border-bottom:2px solid #ecf0f1; }

    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#f8f9fa 0%,#e9ecef 100%); }
    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _card(label, value, variant=""):
    cls = "metric-card " + variant if variant else "metric-card"
    return (
        '<div class="%s">'
        '<div class="metric-label">%s</div>'
        '<div class="metric-value">%s</div></div>' % (cls, label, value)
    )


def _section(title):
    st.markdown(
        '<p class="section-header">%s</p>' % title,
        unsafe_allow_html=True,
    )


def _hero(title, subtitle):
    st.markdown(
        '<div class="hero-banner"><h1>%s</h1>'
        '<p>%s</p></div>' % (title, subtitle),
        unsafe_allow_html=True,
    )


def _model_label(name, champ):
    tags = []
    if name == champ:
        tags.append("Champion")
    if name == BASELINE_MODEL:
        tags.append("Baseline")
    if name == "Ensemble Voting":
        tags.append("Extra - Team Collaboration")
    if tags:
        return f"{name} ({', '.join(tags)})"
    return name


def _display_plot(filename, caption=""):
    fpath = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(fpath):
        st.info(f"{filename} not found.")
        return
    if caption:
        st.image(fpath, caption=caption)
    else:
        st.image(fpath)


def _safe_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else None


@st.cache_resource
def _load_models_cached(_signature):
    out = {}
    for name, fn in MODEL_FILE_KEYS.items():
        p = os.path.join(MODELS_DIR, fn)
        if os.path.exists(p):
            out[name] = joblib.load(p)
    return out


@st.cache_resource
def _load_scaler_cached(_mtime):
    p = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(p):
        return None
    return joblib.load(p)


@st.cache_resource
def _load_encoder_info_cached(_mtime):
    p = os.path.join(MODELS_DIR, "encoder_info.pkl")
    return joblib.load(p) if os.path.exists(p) else None


@st.cache_data
def _load_metrics_cached(_mtime):
    p = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


@st.cache_data
def _load_loyal_profile_cached(_mtime):
    p = os.path.join(MODELS_DIR, "healthy_profile.pkl")
    return joblib.load(p) if os.path.exists(p) else None


@st.cache_data
def _load_optimal_thresholds_cached(_mtime):
    p = os.path.join(MODELS_DIR, "optimal_thresholds.pkl")
    return joblib.load(p) if os.path.exists(p) else None


@st.cache_data
def _load_dataset_cached(_mtime):
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def load_models():
    sig = tuple(
        (fn, _safe_mtime(os.path.join(MODELS_DIR, fn)))
        for fn in MODEL_FILE_KEYS.values()
    )
    return _load_models_cached(sig)


def load_scaler():
    p = os.path.join(MODELS_DIR, "scaler.pkl")
    return _load_scaler_cached(_safe_mtime(p))


def load_encoder_info():
    p = os.path.join(MODELS_DIR, "encoder_info.pkl")
    return _load_encoder_info_cached(_safe_mtime(p))


def load_metrics():
    p = os.path.join(MODELS_DIR, "metrics.json")
    return _load_metrics_cached(_safe_mtime(p))


def load_loyal_profile():
    p = os.path.join(MODELS_DIR, "healthy_profile.pkl")
    return _load_loyal_profile_cached(_safe_mtime(p))


def load_optimal_thresholds():
    p = os.path.join(MODELS_DIR, "optimal_thresholds.pkl")
    return _load_optimal_thresholds_cached(_safe_mtime(p))


def load_dataset():
    return _load_dataset_cached(_safe_mtime(DATA_PATH))


def _count_services(user_data):
    svc = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    return sum(
        1 for c in svc
        if user_data.get(c) in ("Yes", "Fiber optic", "DSL")
    )


def prepare_input(user_data, scaler, encoder_info):
    encoded = encoder_info["encoded_feature_names"]
    row = pd.DataFrame(np.zeros((1, len(encoded))), columns=encoded)

    for feat in NUMERIC_FEATURES_RAW:
        if feat in user_data and feat in row.columns:
            row[feat] = user_data[feat]

    tenure = user_data.get("tenure", 0)
    monthly = user_data.get("MonthlyCharges", 0)
    total = user_data.get("TotalCharges", 0)

    if "AvgMonthlySpend" in row.columns:
        row["AvgMonthlySpend"] = total / (tenure + 1)
    if "ChargeRatio" in row.columns:
        row["ChargeRatio"] = monthly / (total + 1)

    if "ServiceCount" in row.columns:
        row["ServiceCount"] = _count_services(user_data)

    if "HasProtectionBundle" in row.columns:
        row["HasProtectionBundle"] = int(
            user_data.get("OnlineSecurity") == "Yes" and
            user_data.get("OnlineBackup") == "Yes" and
            user_data.get("DeviceProtection") == "Yes" and
            user_data.get("TechSupport") == "Yes"
        )

    if "HighRiskContract" in row.columns:
        row["HighRiskContract"] = int(
            user_data.get("Contract") == "Month-to-month"
            and user_data.get("PaymentMethod") == "Electronic check"
        )

    if "HasStreaming" in row.columns:
        row["HasStreaming"] = int(
            user_data.get("StreamingTV") == "Yes"
            or user_data.get("StreamingMovies") == "Yes"
        )

    for f in ALL_CATEGORICAL_FEATURES:
        if f in user_data:
            col = f"{f}_{user_data[f]}"
            if col in row.columns:
                row[col] = 1

    selected = encoder_info.get("selected_features")
    if selected is not None:
        row = row[selected]

    return row


def _add_engineered_cols(df):
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    svc = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = sum(
        (df[c].isin(["Yes", "Fiber optic", "DSL"])).astype(int)
        for c in svc if c in df.columns
    )

    df["HasProtectionBundle"] = (
        (df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes")
        & (df["DeviceProtection"] == "Yes") & (df["TechSupport"] == "Yes")
    ).astype(int)

    df["HighRiskContract"] = (
        (df["Contract"] == "Month-to-month")
        & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    df["HasStreaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    return df


def prepare_batch(df_raw, scaler, encoder_info, return_source_rows=False):
    df = df_raw.copy()
    df["_source_row"] = np.arange(len(df), dtype=int)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    source_rows = df["_source_row"].astype(int).tolist()
    df = df.drop(columns=["_source_row"])

    if "SeniorCitizen" in df.columns:
        if df["SeniorCitizen"].dtype in [np.int64, np.int32, int]:
            df["SeniorCitizen"] = df["SeniorCitizen"].map(
                {0: "No", 1: "Yes"})

    df = _add_engineered_cols(df)

    cat_cols = [c for c in ALL_CATEGORICAL_FEATURES if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    encoded_names = encoder_info["encoded_feature_names"]
    result = pd.DataFrame(
        np.zeros((len(df), len(encoded_names))), columns=encoded_names)
    for col in encoded_names:
        if col in df.columns:
            result[col] = df[col].values

    selected = encoder_info.get("selected_features")
    if selected is not None:
        result = result[selected]

    if return_source_rows:
        return result, source_rows
    return result


def _shap_waterfall(models, inp):
    try:
        import shap
        rf_pipe = models.get("Random Forest")
        if rf_pipe is None or not hasattr(rf_pipe, "named_steps"):
            st.info("SHAP explanation requires the Random Forest model.")
            return

        rf_model = rf_pipe.named_steps["model"]
        scaler_step = rf_pipe.named_steps.get("scaler")

        if scaler_step is not None:
            input_arr = scaler_step.transform(
                inp.values).astype(np.float64)
        else:
            input_arr = np.array(inp.values, dtype=np.float64)

        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(input_arr)

        if isinstance(shap_vals, list):
            sv_raw = np.asarray(shap_vals[1], dtype=np.float64)
        else:
            sv_raw = np.asarray(shap_vals, dtype=np.float64)
            if sv_raw.ndim == 3:
                sv_raw = sv_raw[:, :, 1]
        while sv_raw.ndim > 1:
            sv_raw = sv_raw[0]
        sv = sv_raw

        feature_names = [str(c) for c in inp.columns]
        n_feat = len(feature_names)
        abs_sv = np.abs(sv[:n_feat])
        top_k = min(10, n_feat)
        sorted_idx = np.argsort(abs_sv)
        top_idx = [int(i) for i in sorted_idx[-top_k:][::-1]]

        top_features = [feature_names[i] for i in top_idx]
        top_values = [float(sv[i]) for i in top_idx]
        top_colors = ["#e74c3c" if v > 0 else "#27ae60"
                      for v in top_values]

        fig = go.Figure(go.Bar(
            x=top_values, y=top_features, orientation="h",
            marker=dict(color=top_colors, opacity=0.85),
            text=[f"{v:+.4f}" for v in top_values],
            textposition="outside",
            textfont=dict(size=11, family="Inter"),
        ))
        fig.update_layout(
            title="Top Feature Contributions to Churn Prediction (SHAP)",
            xaxis_title="SHAP Value (impact on churn probability)",
            height=max(300, top_k * 38),
            margin=dict(t=50, b=40, l=200, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        pos = [(feature_names[i], float(sv[i]))
               for i in top_idx if sv[i] > 0]
        neg = [(feature_names[i], float(sv[i]))
               for i in top_idx if sv[i] < 0]
        parts = []
        if pos:
            risk_str = ", ".join(
                [f"**{f}** (+{v:.3f})" for f, v in pos[:3]])
            parts.append(f"Increasing churn risk: {risk_str}")
        if neg:
            prot_str = ", ".join(
                [f"**{f}** ({v:.3f})" for f, v in neg[:3]])
            parts.append(f"Decreasing churn risk: {prot_str}")
        if parts:
            st.markdown("**Key Factors:**\n- " + "\n- ".join(parts))
        st.caption(
            "Red bars increase churn probability, green bars decrease it. "
            "Based on Random Forest SHAP values.")

    except ImportError:
        st.warning(
            "SHAP library not installed. "
            "Run `pip install shap` for model explanations.")
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {e}")


def _radar_chart(user_data, loyal):
    feats = [f for f in NUMERIC_FEATURES_RAW if f in loyal]
    p_vals, l_vals = [], []
    for f in feats:
        mn, mx = NUMERIC_RANGES[f][0], NUMERIC_RANGES[f][1]
        rng = mx - mn if mx != mn else 1
        p_vals.append(round((user_data[f] - mn) / rng * 100, 1))
        l_vals.append(round((loyal[f] - mn) / rng * 100, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=l_vals + [l_vals[0]], theta=feats + [feats[0]], fill="toself",
        fillcolor="rgba(46,204,113,0.15)",
        line=dict(color="#27ae60", width=2.5),
        name="Loyal Customer Avg", marker=dict(size=6)))
    fig.add_trace(go.Scatterpolar(
        r=p_vals + [p_vals[0]], theta=feats + [feats[0]], fill="toself",
        fillcolor="rgba(52,152,219,0.15)",
        line=dict(color="#3498db", width=2.5),
        name="Current Customer", marker=dict(size=6)))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, xanchor="center",
                    x=0.5, font=dict(size=12)),
        height=450,
        margin=dict(t=40, b=60, l=80, r=80),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_prediction():
    _hero(
        "Customer Churn Risk Assessment",
        "AI-powered churn prediction using 5 ML models with SHAP "
        "explainability — identify at-risk customers before they leave",
    )

    models = load_models()
    scaler = load_scaler()
    enc = load_encoder_info()
    loyal = load_loyal_profile()
    opt_thresholds = load_optimal_thresholds()

    if not models or scaler is None or enc is None:
        st.error("Models not found. Run `python train.py` first or use setup.bat.")
        return

    with st.sidebar:
        st.markdown("### Customer Information")
        st.markdown("---")
        user_data = {}

        st.markdown("**Account Details**")
        for f in NUMERIC_FEATURES_RAW:
            mn, mx, dv, step = NUMERIC_RANGES[f]
            label = FEATURE_LABELS.get(f, f)
            if isinstance(step, float):
                user_data[f] = st.slider(
                    label, float(mn), float(mx), float(dv), step,
                    key=f"s_{f}")
            else:
                user_data[f] = st.slider(
                    label, int(mn), int(mx), int(dv), int(step),
                    key=f"s_{f}")

        st.markdown("---")
        st.markdown("**Services & Profile**")
        for f in ALL_CATEGORICAL_FEATURES:
            opts = CATEGORY_OPTIONS.get(f, ["Yes", "No"])
            label = FEATURE_LABELS.get(f, f)
            user_data[f] = st.selectbox(label, opts, key=f"sel_{f}")

        st.markdown("---")
        predict_btn = st.button(
            "Predict Churn Risk", type="primary",
            use_container_width=True)

    if not predict_btn:
        _section("How to Use")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                '<div class="metric-card"><div class="metric-label">Step 1</div>'
                '<div class="metric-value" style="font-size:1.2rem;">Enter Customer Data</div>'
                '<p style="color:#7f8c8d;font-size:0.85rem;margin-top:0.3rem;">'
                "Use the sidebar to input customer's account details, "
                "services, and profile.</p></div>",
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                '<div class="metric-card warning"><div class="metric-label">Step 2</div>'
                '<div class="metric-value" style="font-size:1.2rem;">Click Predict</div>'
                '<p style="color:#7f8c8d;font-size:0.85rem;margin-top:0.3rem;">'
                'Press "Predict Churn Risk" at the bottom of the sidebar.'
                '</p></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                '<div class="metric-card success"><div class="metric-label">Step 3</div>'
                '<div class="metric-value" style="font-size:1.2rem;">View Results</div>'
                '<p style="color:#7f8c8d;font-size:0.85rem;margin-top:0.3rem;">'
                "Review churn risk, model consensus, SHAP explanation, "
                "and profile comparison.</p></div>",
                unsafe_allow_html=True)

        _section("Awaiting Customer Data...")
        fig = go.Figure(go.Indicator(
            mode="gauge", value=0,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [
                    {"range": [0, 30], "color": "#d4efdf"},
                    {"range": [30, 60], "color": "#fdebd0"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "#bdc3c7", "width": 4},
                              "thickness": 0.8, "value": 0},
            },
            title={"text": "Churn Risk Score", "font": {"size": 18}},
        ))
        fig.update_layout(height=280,
                          margin=dict(t=60, b=20, l=40, r=40))
        st.plotly_chart(fig, use_container_width=True)
        return

    inp = prepare_input(user_data, scaler, enc)
    preds, probs, thresholds = {}, {}, {}
    for name, model in models.items():
        prob = float(model.predict_proba(inp)[0][1])
        thr = float(opt_thresholds.get(name, 0.5)) if opt_thresholds else 0.5
        probs[name] = prob
        thresholds[name] = thr
        preds[name] = 1 if prob >= thr else 0

    md = load_metrics()
    champ = md.get("_champion", "Random Forest") if md else "Random Forest"
    champ_prob = probs.get(champ, 0.5)
    champ_pred = preds.get(champ, 0)
    champ_thr = thresholds.get(champ, 0.5)
    risk_pct = champ_prob * 100
    high_cnt = sum(1 for p in preds.values() if p == 1)
    total = len(preds)

    if champ_pred == 1:
        st.markdown(
            f'<div class="result-high-risk"><h2>HIGH CHURN RISK</h2>'
            f'<p>{champ} predicts {risk_pct:.1f}% churn probability. '
            f'(Threshold: {champ_thr:.3f}) '
            f'{high_cnt}/{total} models agree. '
            f'Consider retention offers for this customer.</p></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="result-low-risk"><h2>LOW CHURN RISK</h2>'
            f'<p>{champ} predicts {risk_pct:.1f}% churn probability. '
            f'(Threshold: {champ_thr:.3f}) '
            f'{total - high_cnt}/{total} models indicate the customer '
            f'is likely to stay.</p></div>',
            unsafe_allow_html=True)
    st.caption(
        f"Decision rule: predict churn when probability >= threshold. "
        f"For {champ}: {champ_prob:.3f} vs {champ_thr:.3f}."
    )

    cg, cc = st.columns(2)
    with cg:
        _section("Churn Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_pct,
            number={"suffix": "%",
                    "font": {"size": 48, "color": "#2c3e50"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2,
                         "tickfont": {"size": 12}},
                "bar": {"color": "#2c3e50", "thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "#82E0AA"},
                    {"range": [30, 50], "color": "#F9E79F"},
                    {"range": [50, 70], "color": "#F5B041"},
                    {"range": [70, 100], "color": "#EC7063"},
                ],
                "threshold": {
                    "line": {"color": "#e74c3c", "width": 6},
                    "thickness": 0.85, "value": risk_pct},
            },
            title={"text": f"<b>{champ}</b><br>Churn Probability",
                   "font": {"size": 14, "color": "#7f8c8d"}},
        ))
        fig.update_layout(
            height=320, margin=dict(t=80, b=20, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with cc:
        _section("Model Consensus")
        names_s = sorted(probs, key=lambda x: probs[x], reverse=True)
        probs_s = [probs[n] * 100 for n in names_s]
        cols_s = ["#e74c3c" if probs[n] >= thresholds.get(n, 0.5) else "#27ae60"
                  for n in names_s]

        fig = go.Figure(go.Bar(
            x=probs_s, y=names_s, orientation="h",
            marker=dict(color=cols_s, opacity=0.85),
            text=[f"{p:.1f}%" for p in probs_s], textposition="auto",
            textfont=dict(size=14, color="white", family="Inter")))
        fig.add_vline(
            x=50, line_dash="dash", line_color="#7f8c8d", line_width=2,
            annotation_text="50% Reference",
            annotation_position="top")
        fig.update_layout(
            height=320,
            xaxis=dict(title="Churn Probability (%)", range=[0, 105]),
            margin=dict(t=40, b=50, l=10, r=20),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        verdict = "CHURN" if high_cnt > total / 2 else "RETAIN"
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem;background:#f8f9fa;'
            f'border-radius:8px;margin-top:-0.5rem;">'
            f'<b>{high_cnt}/{total}</b> models predict '
            f'<b>{verdict}</b></div>',
            unsafe_allow_html=True)

    with st.expander("Why This Prediction? (SHAP Explanation)",
                     expanded=False):
        _shap_waterfall(models, inp)

    _section("Customer Profile vs Loyal Customers")
    if loyal:
        _radar_chart(user_data, loyal)

    _section("Detailed Model Predictions")
    rows = []
    for n in MODEL_NAMES:
        if n not in preds:
            continue
        thr = thresholds.get(n, 0.5)
        lbl = "Likely to Churn" if preds[n] == 1 else "Likely to Stay"
        pv = probs[n] * 100
        rows.append({
            "Model": _model_label(n, champ),
            "Prediction": lbl,
            "Churn Probability": f"{pv:.1f}%",
            "Optimal Threshold": f"{thr:.3f}",
            "Rule": f"{probs[n]:.3f} {'>=' if probs[n] >= thr else '<'} {thr:.3f}",
            "Confidence": f"{max(pv, 100 - pv):.1f}%",
        })
    st.dataframe(pd.DataFrame(rows),
                 use_container_width=True, hide_index=True)

    _section("Model Decision Baselines")
    base_rows = []
    for n in MODEL_NAMES:
        if n not in probs:
            continue
        thr = thresholds.get(n, 0.5)
        delta = probs[n] - thr
        base_rows.append({
            "Model": _model_label(n, champ),
            "Default Baseline": "0.500",
            "Tuned Baseline": f"{thr:.3f}",
            "Probability": f"{probs[n]:.3f}",
            "Margin vs Baseline": f"{delta:+.3f}",
        })
    st.dataframe(pd.DataFrame(base_rows), use_container_width=True, hide_index=True)
    st.markdown(
        "**Why not fixed at 50%?** The dataset is imbalanced, so a fixed 0.500 "
        "cutoff can under-detect churners or over-trigger false alarms. "
        "Each model threshold is tuned via 5-fold CV on the training set to maximise F1, "
        "which balances precision and recall for retention decisions."
    )

    with st.expander("View Customer Input Summary"):
        cols = st.columns(3)
        for i, (k, v) in enumerate(user_data.items()):
            with cols[i % 3]:
                st.markdown(f"**{FEATURE_LABELS.get(k, k)}:** {v}")


def _data_browser(df):
    all_cols = list(df.columns)
    selected_cols = st.multiselect(
        "Select columns to display:", all_cols, default=all_cols,
        key="eda_cols")

    if not selected_cols:
        st.warning("Please select at least one column.")
        return

    df_filtered = df[selected_cols].copy()

    st.markdown("**Column Filters:**")
    filter_cols = st.multiselect(
        "Choose columns to filter by:", selected_cols,
        key="eda_filter_cols")

    for fc in filter_cols:
        if df[fc].dtype == "object" or df[fc].nunique() <= 10:
            unique_vals = sorted(df[fc].dropna().unique().tolist())
            chosen = st.multiselect(
                f"Filter `{fc}`:", unique_vals, default=unique_vals,
                key=f"eda_fv_{fc}")
            df_filtered = df_filtered[
                df_filtered[fc].isin(chosen) | df_filtered[fc].isnull()]
        else:
            col_min = float(df[fc].min())
            col_max = float(df[fc].max())
            if col_min < col_max:
                rng = st.slider(
                    f"Filter `{fc}` range:", col_min, col_max,
                    (col_min, col_max), key=f"eda_fr_{fc}")
                df_filtered = df_filtered[
                    (df_filtered[fc] >= rng[0])
                    & (df_filtered[fc] <= rng[1])
                    | df_filtered[fc].isnull()]

    st.markdown(
        '<div style="padding:0.5rem;background:#f8f9fa;border-radius:8px;'
        'margin-bottom:0.5rem;">Showing <b>%s</b> of '
        '<b>%s</b> rows | <b>%d</b> columns '
        'selected</div>' % (
            f"{len(df_filtered):,}", f"{len(df):,}", len(selected_cols)),
        unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)


def page_eda():
    _hero("Data Exploration",
          "Exploratory Data Analysis of the Telco Customer Churn Dataset")

    df = load_dataset()
    if df is None:
        st.error("Dataset not found.")
        return

    _section("Dataset Overview")
    total_cols = df.shape[1]
    excluded_cols = ["customerID", TARGET_COL]
    excluded_present = [c for c in excluded_cols if c in df.columns]
    usable_features = total_cols - len(excluded_present)
    rate = (df[TARGET_COL] == TARGET_POSITIVE).mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(_card("Total Customers", f"{df.shape[0]:,}"),
                     unsafe_allow_html=True)
    with c2:
        st.markdown(_card("Total Columns", total_cols),
                     unsafe_allow_html=True)
    with c3:
        st.markdown(
            _card("Excluded Columns", len(excluded_present), "warning"),
            unsafe_allow_html=True)
    with c4:
        st.markdown(_card("Usable Features", usable_features, "success"),
                     unsafe_allow_html=True)
    with c5:
        st.markdown(_card("Churn Rate", f"{rate:.1f}%", "danger"),
                     unsafe_allow_html=True)

    st.info(
        "📋 **Data Cleaning Note:** Records containing missing values "
        "(e.g. TotalCharges with blank/whitespace entries) have been "
        "**removed from the dataset** during preprocessing to ensure data "
        "quality. This avoids introducing imputed/artificial values that "
        "could bias analysis and model training. The original dataset had "
        "7,043 rows; after removal the cleaned dataset has {:,} rows."
        .format(df.shape[0]))

    with st.expander("Column Details"):
        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = int(df[col].nunique())
            status = ("❌ Excluded" if col in excluded_present
                      else "✅ Feature")
            reason = ""
            if col == "customerID":
                reason = "Unique identifier, not predictive"
            elif col == TARGET_COL:
                reason = "Target variable (label)"
            col_info.append({
                "Column": col, "Type": dtype,
                "Unique": n_unique, "Status": status,
                "Reason": reason,
            })
        st.dataframe(pd.DataFrame(col_info),
                     use_container_width=True, hide_index=True)
        st.markdown(
            '<div style="padding:0.5rem;background:#f8f9fa;border-radius:8px;'
            'margin-top:0.5rem;"><b>Missing Values:</b> 0 (cleaned) | '
            "<b>Excluded:</b> %s</div>" % ", ".join(excluded_present),
            unsafe_allow_html=True)

    _section("Churn Distribution")
    tc = df[TARGET_COL].value_counts()
    fig = go.Figure(go.Pie(
        labels=tc.index, values=tc.values, hole=0.5,
        marker=dict(colors=["#27ae60", "#e74c3c"],
                    line=dict(color="white", width=3)),
        textinfo="label+percent+value", textfont=dict(size=14)))
    fig.update_layout(
        height=350, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{len(df):,}</b><br>Customers",
                          x=0.5, y=0.5, font_size=16, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

    _section("Feature Correlation")
    st.caption(
        "All features encoded numerically. Categorical columns are "
        "label-encoded for correlation analysis.")
    corr_df = df.drop(
        columns=[c for c in ["customerID"] if c in df.columns]).copy()
    for col in corr_df.select_dtypes(include=["object"]).columns:
        corr_df[col] = corr_df[col].astype("category").cat.codes
    corr = corr_df.corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=8)))
    n_corr_cols = len(corr.columns)
    fig_h = max(500, n_corr_cols * 28)
    fig.update_layout(
        height=fig_h, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)))
    st.plotly_chart(fig, use_container_width=True)

    _section("Feature Distributions by Churn")
    sel = st.selectbox("Select numeric feature:",
                       NUMERIC_FEATURES_RAW, key="eda_num")
    col_hist, col_box = st.columns(2)
    with col_hist:
        fig = go.Figure()
        for lab, clr in [(TARGET_NEGATIVE, "#27ae60"),
                         (TARGET_POSITIVE, "#e74c3c")]:
            subset = df[df[TARGET_COL] == lab][sel].dropna()
            fig.add_trace(go.Histogram(
                x=subset, name=lab, marker_color=clr,
                opacity=0.7, nbinsx=40))
        fig.update_layout(
            barmode="overlay", title=f"Distribution of {sel}",
            height=350, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15,
                        xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)
    with col_box:
        fig = go.Figure()
        for lab, clr in [(TARGET_NEGATIVE, "#27ae60"),
                         (TARGET_POSITIVE, "#e74c3c")]:
            fig.add_trace(go.Box(
                y=df[df[TARGET_COL] == lab][sel].dropna(),
                name=lab, marker_color=clr, boxmean="sd"))
        fig.update_layout(
            title=f"Box Plot of {sel}", height=350,
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    _section("Categorical Feature Analysis")
    sel_cat = st.selectbox(
        "Select categorical feature:", ALL_CATEGORICAL_FEATURES,
        key="eda_cat")
    grp = df.groupby(
        [sel_cat, TARGET_COL]).size().reset_index(name="Count")
    fig = px.bar(
        grp, x=sel_cat, y="Count", color=TARGET_COL, barmode="group",
        color_discrete_map={TARGET_NEGATIVE: "#27ae60",
                            TARGET_POSITIVE: "#e74c3c"},
        title=f"{sel_cat} vs Churn")
    fig.update_layout(
        height=420, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(b=100),
        legend=dict(title_text="", orientation="h", y=-0.25,
                    xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data Explorer (Full Dataset with Filters)",
                     expanded=False):
        _data_browser(df)

    with st.expander("Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)


def _kfold_section(models, scaler, enc):
    md = load_metrics()
    kfold_data = md.get("_kfold_details") if md else None

    if kfold_data:
        st.caption(
            "Results from K-Fold threshold optimisation performed "
            "during training (training data only, no test leakage).")

        kf_rows = []
        kfold_results = {}
        for name in MODEL_NAMES:
            if name not in kfold_data:
                continue
            info = kfold_data[name]
            scores = np.array(info["fold_f1"])
            kfold_results[name] = scores
            champ = md.get("_champion", "")
            row = {"Model": _model_label(name, champ)}
            for i, s in enumerate(scores):
                row[f"Fold {i+1}"] = s
            row["Mean"] = info["avg_f1"]
            row["Std"] = info["std_f1"]
            row["Avg Threshold"] = info["avg_threshold"]
            kf_rows.append(row)

        kf_df = pd.DataFrame(kf_rows)
        n_folds = max(
            len(kfold_data[n]["fold_f1"]) for n in kfold_data
        ) if kf_rows else 5
        fold_cols = [f"Fold {i+1}" for i in range(n_folds)]
        fmt_cols = fold_cols + ["Mean", "Std", "Avg Threshold"]
        st.dataframe(
            kf_df.style
            .format({c: "{:.4f}" for c in fmt_cols})
            .highlight_max(subset=["Mean"], color="#d4efdf")
            .highlight_min(subset=["Std"], color="#d4efdf"),
            use_container_width=True, hide_index=True)

        fig_kf = go.Figure()
        for name, scores in kfold_results.items():
            fig_kf.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(scores))],
                y=scores, name=name,
                marker_color=MODEL_COLORS.get(name),
                text=[f"{s:.3f}" for s in scores],
                textposition="outside", textfont=dict(size=9)))
        fig_kf.update_layout(
            barmode="group", height=420,
            title="F1-Score per Fold (5-Fold Stratified CV, Threshold Tuning)",
            yaxis=dict(title="F1-Score", range=[0, 1.1]),
            legend=dict(orientation="h", y=-0.15,
                        xanchor="center", x=0.5),
            margin=dict(t=50, b=70, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_kf, use_container_width=True)

        best_cv = max(kfold_results,
                      key=lambda x: kfold_results[x].mean())
        best_mean = kfold_results[best_cv].mean()
        best_std = kfold_results[best_cv].std()
        st.markdown(
            '<div style="text-align:center;padding:0.8rem;background:#f8f9fa;'
            'border-radius:8px;"><b>Best K-Fold CV Model:</b> %s '
            '(Mean F1 = %.4f &plusmn; %.4f)</div>' % (
                best_cv, best_mean, best_std),
            unsafe_allow_html=True)
    else:
        st.info(
            "K-Fold cross-validation results not found in metrics. "
            "Re-run train.py to generate them.")


def page_model_performance():
    _hero("Model Performance",
          "Evaluation of 5 models (4 base + Ensemble Voting) with RFECV, "
          "SHAP & threshold optimisation")

    md = load_metrics()
    if md is None:
        st.error("Metrics not found. Run train.py first.")
        return

    champ = md.get("_champion", "Random Forest")
    opt_thresholds = md.get("_optimal_thresholds", {})

    _section("Model Scorecard (Default Threshold = 0.5)")
    mnames = ["Accuracy", "Precision", "Recall", "F1-Score",
              "AUC", "Log Loss"]
    rows = []
    for n in MODEL_NAMES:
        if n in md:
            row = {"Model": _model_label(n, champ)}
            for m in mnames:
                row[m] = md[n].get(m, 0)
            rows.append(row)
    sdf = pd.DataFrame(rows)
    st.dataframe(
        sdf.style
        .format({m: "{:.4f}" for m in mnames})
        .highlight_max(subset=["Accuracy", "Precision", "Recall",
                                "F1-Score", "AUC"], color="#d4efdf")
        .highlight_min(subset=["Log Loss"], color="#d4efdf"),
        use_container_width=True, hide_index=True)

    _section("Model Scorecard (Optimal Thresholds)")
    champ_basis = md.get(
        "_champion_basis",
        "highest held-out test F1 at tuned threshold")
    st.markdown(
        f"**Champion Model:** {champ} "
        f"({champ_basis})"
        f"\n\n**Baseline Model:** {BASELINE_MODEL}"
        f"\n\n*Champion is selected from the 4 base models only. "
        f"Ensemble Voting is an extra team collaboration model.*")
    mnames_opt = ["Accuracy_opt", "Precision_opt",
                  "Recall_opt", "F1-Score_opt"]
    rows_opt = []
    for n in MODEL_NAMES:
        if n not in md:
            continue
        thr = opt_thresholds.get(n, 0.5)
        row = {"Model": _model_label(n, champ),
               "Default Baseline": 0.5, "Tuned Baseline": thr}
        for m in mnames_opt:
            row[m.replace("_opt", "")] = md[n].get(m, 0)
        row["KFold_Val_F1"] = md[n].get("Validation_F1_opt", np.nan)
        row["AUC"] = md[n].get("AUC", 0)
        rows_opt.append(row)
    sdf_opt = pd.DataFrame(rows_opt)
    fmt_cols = {c: "{:.4f}" for c in
                ["Default Baseline", "Tuned Baseline", "Accuracy",
                 "Precision", "Recall", "F1-Score", "KFold_Val_F1", "AUC"]}
    st.dataframe(
        sdf_opt.style.format(fmt_cols)
        .highlight_max(subset=["Accuracy", "Precision", "Recall",
                                "F1-Score", "KFold_Val_F1", "AUC"],
                       color="#d4efdf"),
        use_container_width=True, hide_index=True)
    st.caption(
        "Tuned baseline is the K-Fold averaged optimal threshold "
        "per model, not fixed at 0.500."
    )
    st.markdown(
        "**Why thresholds are not 50%:** Churn data is imbalanced and business "
        "costs are asymmetric. A single 0.500 rule is often suboptimal. "
        "Each model's threshold is tuned via 5-fold CV on the training set: "
        "in each fold, the threshold that maximises F1 on the fold-validation "
        "portion is found, and the 5 thresholds are averaged."
    )

    _section("Performance Comparison (Optimal)")
    cmp = ["Accuracy_opt", "Precision_opt", "Recall_opt",
           "F1-Score_opt", "AUC"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    fig = go.Figure()
    for n in MODEL_NAMES:
        if n in md:
            vals = [md[n].get(m, 0) for m in cmp]
            display_name = _model_label(n, champ)
            fig.add_trace(go.Bar(
                x=labels, y=vals, name=display_name,
                marker_color=MODEL_COLORS.get(n),
                text=[f"{v:.3f}" for v in vals],
                textposition="outside", textfont=dict(size=10)))
    fig.update_layout(
        barmode="group", height=450,
        yaxis=dict(range=[0, 1.15]),
        legend=dict(orientation="h", y=-0.12,
                    xanchor="center", x=0.5),
        margin=dict(t=30, b=90, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    _section("K-Fold Cross-Validation (Threshold Optimisation)")
    st.markdown(
        "5-Fold Stratified Cross-Validation for threshold tuning. "
        "In each fold, the full pipeline (StandardScaler → SMOTE → model) "
        "is re-trained on fold-train and its optimal F1-threshold is found on fold-val. "
        "Scaling and SMOTE are applied **inside each fold** to prevent data leakage. "
        "Results are saved during training and displayed here.")
    models = load_models()
    scaler = load_scaler()
    enc = load_encoder_info()
    if models and scaler is not None and enc is not None:
        try:
            _kfold_section(models, scaler, enc)
        except Exception as e:
            st.warning(f"Could not run K-Fold CV: {e}")
    else:
        st.warning("Models not loaded. Run train.py first.")

    _section("Training Visualizations")
    tabs = st.tabs([
        "Confusion Matrices", "ROC Curves", "PR Curves",
        "Feature Importance", "Learning Curves",
        "Threshold Optimisation", "SHAP Summary", "Tuning Evidence",
    ])
    plot_meta = {
        0: ("confusion_matrices.png",
            "Confusion matrices at optimal thresholds. "
            "Red box = missed churners (FN)."),
        1: ("roc_curves.png",
            "ROC curves. Higher AUC = better discrimination."),
        2: ("precision_recall_curves.png",
            "Precision-Recall trade-off. Important for "
            "imbalanced data."),
        3: ("feature_importance.png",
            "Top features for predicting churn (Random Forest)."),
        4: ("learning_curves.png",
            "How model performance changes with more training data."),
        5: ("threshold_optimisation.png",
            "Effect of decision threshold on Precision/Recall/F1."),
    }
    for i, tab in enumerate(tabs[:6]):
        with tab:
            fn, cap = plot_meta[i]
            _display_plot(fn, cap)

    with tabs[6]:
        shap_path = os.path.join(PLOTS_DIR, "shap_summary.png")
        if os.path.exists(shap_path):
            st.image(
                shap_path,
                caption="SHAP Global Feature Importance "
                        "(Random Forest). "
                        "Each dot = one sample. Red = high feature "
                        "value, blue = low.")
        else:
            st.info("SHAP summary plot not found. Ensure shap is "
                    "installed and train.py has been run.")

    with tabs[7]:
        cl, ck = st.columns(2)
        with cl:
            _display_plot("lr_c_vs_accuracy.png",
                          "LR: C vs Accuracy")
        with ck:
            _display_plot("knn_k_vs_error.png",
                          "KNN: K vs Error Rate")
        _display_plot("decision_tree_structure.png",
                      "Decision Tree (top 3 levels)")

    _section("Preprocessing Evidence")
    etabs = st.tabs([
        "Class Distribution", "Correlation",
        "Feature Distributions", "Categorical Churn Rates",
        "Scaling", "Feature Selection (RFE)",
    ])
    emap = {
        0: "class_distribution.png",
        1: "correlation_heatmap.png",
        2: "feature_distributions.png",
        3: "categorical_churn_rates.png",
        4: "scaling_comparison.png",
    }
    for i, tab in enumerate(etabs[:5]):
        with tab:
            _display_plot(emap[i])

    with etabs[5]:
        rfe_path = os.path.join(PLOTS_DIR, "rfe_feature_ranking.png")
        if os.path.exists(rfe_path):
            st.image(
                rfe_path,
                caption="RFECV: Cross-validation F1 scores and "
                        "feature rankings. Green = selected, "
                        "Red = removed.")
        else:
            st.info("RFE feature ranking plot not found.")


def page_batch_prediction():
    _hero("Batch Prediction",
          "Upload a CSV file to predict churn for multiple "
          "customers at once")

    models = load_models()
    scaler = load_scaler()
    enc = load_encoder_info()
    opt_thresholds = load_optimal_thresholds()
    md = load_metrics()

    if not models or scaler is None or enc is None:
        st.error("Models not found. Run `python train.py` first.")
        return

    champ = md.get("_champion", "Random Forest") if md else "Random Forest"
    champ_thr = 0.5
    if opt_thresholds and champ in opt_thresholds:
        champ_thr = opt_thresholds[champ]

    _section("Upload Customer Data")
    st.markdown(
        "Upload a CSV file with the same format as the training data "
        "(`Telco_Customer_Churn.csv`).\n"
        "Required columns: `gender`, `SeniorCitizen`, `Partner`, "
        "`Dependents`, `tenure`,\n"
        "`PhoneService`, `MultipleLines`, `InternetService`, "
        "`OnlineSecurity`, `OnlineBackup`,\n"
        "`DeviceProtection`, `TechSupport`, `StreamingTV`, "
        "`StreamingMovies`, `Contract`,\n"
        "`PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, "
        "`TotalCharges`.\n\n"
        "The `customerID` and `Churn` columns are optional and will be "
        "ignored during prediction.")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"],
                                key="batch_upload")
    if uploaded is None:
        return

    try:
        df_raw = pd.read_csv(uploaded)
        st.success(
            f"Loaded {len(df_raw)} rows x {df_raw.shape[1]} columns")

        with st.expander("Preview Uploaded Data (first 10 rows)",
                         expanded=True):
            st.dataframe(df_raw.head(10), use_container_width=True,
                         hide_index=True)

        if not st.button("Run Batch Prediction", type="primary",
                         use_container_width=True):
            return

        with st.spinner("Processing predictions..."):
            has_id = "customerID" in df_raw.columns
            X_batch, source_rows = prepare_batch(
                df_raw, scaler, enc, return_source_rows=True)
            if len(source_rows) == 0:
                st.error(
                    "No valid rows remain after preprocessing. "
                    "Check `TotalCharges` and required fields.")
                return

            if has_id:
                ids = df_raw.iloc[source_rows]["customerID"].tolist()
            else:
                ids = [int(i) + 1 for i in source_rows]

            dropped_rows = len(df_raw) - len(source_rows)
            if dropped_rows > 0:
                st.warning(
                    f"{dropped_rows} row(s) were excluded during "
                    "preprocessing (e.g. invalid TotalCharges).")

            champ_model = models.get(champ)
            if champ_model is None:
                champ_model = list(models.values())[0]

            probs_champ = champ_model.predict_proba(X_batch)[:, 1]
            preds_champ = (probs_champ >= champ_thr).astype(int)

            results = pd.DataFrame({
                "Customer_ID": ids,
                f"{champ}_Probability (%)": np.round(
                    probs_champ * 100, 2),
                "Prediction": [
                    "Yes" if p == 1 else "No"
                    for p in preds_champ],
                "Risk_Level": pd.cut(
                    probs_champ * 100,
                    bins=[0, 30, 60, 100],
                    labels=["Low", "Medium", "High"],
                    include_lowest=True),
            })

            for name, model in models.items():
                if name != champ:
                    results[f"{name}_Prob (%)"] = np.round(
                        model.predict_proba(X_batch)[:, 1] * 100, 2)

        _section("Prediction Results")

        n_churn = int((preds_champ == 1).sum())
        n_total = len(preds_champ)
        churn_pct = n_churn / n_total * 100
        avg_prob = probs_champ.mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(_card("Total Customers", f"{n_total:,}"),
                         unsafe_allow_html=True)
        with c2:
            st.markdown(
                _card("Predicted Churn", f"{n_churn:,}", "danger"),
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                _card("Churn Rate", f"{churn_pct:.1f}%", "warning"),
                unsafe_allow_html=True)
        with c4:
            st.markdown(
                _card("Avg Probability", f"{avg_prob:.1f}%"),
                unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probs_champ * 100, nbinsx=30,
            marker_color="#3498db", opacity=0.8,
            name="Churn Probability"))
        fig.add_vline(
            x=champ_thr * 100, line_dash="dash",
            line_color="#e74c3c", line_width=2,
            annotation_text=f"Threshold ({champ_thr:.0%})",
            annotation_position="top")
        fig.update_layout(
            title="Churn Probability Distribution",
            xaxis_title="Churn Probability (%)",
            yaxis_title="Number of Customers",
            height=350, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=50, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(results, use_container_width=True,
                     hide_index=True)

        csv_out = results.to_csv(index=False)
        st.download_button(
            label="Download Results CSV", data=csv_out,
            file_name="churn_predictions.csv", mime="text/csv",
            use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Make sure the CSV has the correct column "
                "names and format.")


def page_about():
    _hero("About This Project",
          "BMDS2003 Data Science — Telco Customer Churn "
          "Prediction (v3)")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""

This application uses **5 machine learning models** (4 base + 1 ensemble) to predict whether a telecom
customer will **churn** (cancel their service). Built as part of the BMDS2003 Data Science course,
it follows the **CRISP-DM** methodology with advanced features: **RFECV feature selection**,
**SHAP explainability**, **threshold optimisation**, and **batch prediction**.


Customer churn is one of the biggest challenges for telecom companies. Acquiring a new
customer costs **5-25x more** than retaining an existing one. By predicting which customers
are at risk of leaving, companies can proactively offer retention incentives.


#### 4 Base Models (1 per team member)

| Model | Role | Key Strength |
|-------|------|-------------|
| **Logistic Regression** | Baseline | Fast, interpretable, linear boundary |
| **K-Nearest Neighbors** | Distance-based | Non-parametric, captures local patterns |
| **Decision Tree** | Tree-based (A) | Highly interpretable, visual rules |
| **Random Forest** | Tree-based (B) | Ensemble power, highest accuracy |

#### Extra Model (Team Collaboration)

| Model | Role | Key Strength |
|-------|------|-------------|
| **Ensemble Voting** | Meta-model | CV-F1-weighted soft voting of all 4 base models |


- **RFECV Feature Selection**: Recursive Feature Elimination with CV automatically removes low-signal features for better generalisation
- **RandomizedSearchCV**: Efficient hyperparameter tuning with 80 random iterations across wider parameter distributions (scipy.stats)
- **Ensemble Voting**: Soft voting classifier with CV-F1-weighted probability averaging for more robust predictions
- **SHAP Explainability**: TreeExplainer provides global and per-prediction feature importance ("why did the model predict churn?")
- **Batch Prediction**: Upload a CSV to predict churn for hundreds/thousands of customers with downloadable results
- **Feature Engineering**: 6 new features (AvgMonthlySpend, ChargeRatio, ServiceCount, HasProtectionBundle, HighRiskContract, HasStreaming)
- **K-Fold Threshold Optimisation**: Each model's decision threshold is tuned via 5-fold CV (averaged across folds) to maximise F1-Score
- **SMOTE-in-CV**: Synthetic oversampling applied inside cross-validation folds to prevent data leakage


- **Python 3.10** — Core language
- **scikit-learn + imbalanced-learn** — ML models, SMOTE, RFECV, VotingClassifier
- **SHAP** — Model explainability
- **Streamlit** — Web application framework
- **Plotly** — Interactive visualizations
- **Pandas / NumPy / SciPy** — Data processing & statistical distributions
- **Matplotlib / Seaborn** — Static visualizations
""")

    with c2:
        st.markdown(
            "### Dataset\n\n**Telco Customer Churn**\n"
            "- Records: ~7,043\n"
            "- Original Features: 19\n"
            "- Engineered Features: +6\n"
            "- Target: Churn (Yes/No)\n"
            "- Churn Rate: ~26.5%\n\n"
            "### Quick Stats")
        di = get_device_info()
        gpu_tag = "Available" if di["gpu_available"] else "CPU Only"
        st.markdown(
            f"**Environment:**\n- Python: {di['python_version']}\n"
            f"- GPU: {gpu_tag}")
        md = load_metrics()
        if md:
            ch = md.get("_champion", "N/A")
            st.markdown(
                f"**Best Base Model (Champion):** {ch}\n\n"
                f"**Baseline Model:** {BASELINE_MODEL}\n\n"
                f"*Champion selected from 4 base models only*")
            if ch in md:
                for m in ["Accuracy_opt", "Recall_opt",
                           "F1-Score_opt", "AUC"]:
                    lbl = m.replace("_opt", "")
                    if m in md[ch]:
                        st.metric(lbl, f"{md[ch][m]:.4f}")

    st.markdown(
        '---\n<div style="text-align:center;color:#7f8c8d;'
        'padding:1rem;">'
        '<p>BMDS2003 Data Science Project | Telco Customer Churn '
        'Prediction v3</p>'
        '<p style="font-size:0.8rem;">Disclaimer: This tool is for '
        'educational purposes only.</p></div>',
        unsafe_allow_html=True)


def main():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:1rem 0;">'
            '<h2 style="margin:0;font-size:1.4rem;">Telco Churn<br>'
            'Prediction</h2>'
            '<p style="color:#7f8c8d;font-size:0.8rem;'
            'margin-top:0.3rem;">'
            'BMDS2003 Data Science v3</p></div>',
            unsafe_allow_html=True)
        st.markdown("---")

    t1, t2, t3, t4, t5 = st.tabs([
        "Prediction", "Data Exploration", "Model Performance",
        "Batch Prediction", "About",
    ])
    with t1:
        page_prediction()
    with t2:
        page_eda()
    with t3:
        page_model_performance()
    with t4:
        page_batch_prediction()
    with t5:
        page_about()


if __name__ == "__main__":
    main()
