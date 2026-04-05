# train.py - BMDS2003 Data Science

import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, learning_curve, StratifiedKFold,
    cross_val_score,
)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    roc_curve, precision_recall_curve,
)
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

from config import (
    DATA_PATH, MODELS_DIR, PLOTS_DIR, RANDOM_STATE,
    TARGET_COL, TARGET_POSITIVE, TARGET_NEGATIVE, ID_COL,
    NUMERIC_FEATURES_RAW, NUMERIC_FEATURES,
    ENGINEERED_NUMERIC_FEATURES, ENGINEERED_BINARY_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    PARAM_DISTRIBUTIONS, RAW_C_VALUES,
    RANDOMIZED_N_ITER, MIN_RFE_FEATURES,
    BASE_MODEL_NAMES, BASELINE_MODEL,
    MODEL_NAMES, MODEL_COLORS, MODEL_FILE_KEYS, print_env_info,
)

np.random.seed(RANDOM_STATE)
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

_DPI = 150
_log = []


def knn_plot(X_train, y_train):
    ks = list(range(1, 25))
    errs = []
    cv_knn = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    for k in ks:
        p = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", KNeighborsClassifier(n_neighbors=k)),
        ])
        scores = cross_val_score(p, X_train, y_train, cv=cv_knn,
                                 scoring="accuracy", n_jobs=-1)
        errs.append(1 - scores.mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, errs, "o-", color=MODEL_COLORS["KNN"], lw=2, ms=6)
    bk = np.argmin(errs)
    ax.plot(ks[bk], errs[bk], "s", color="#27ae60", ms=14, zorder=5)
    ax.annotate(
        "Best K=%d\nError=%.4f" % (ks[bk], errs[bk]),
        xy=(ks[bk], errs[bk]), xytext=(20, 20),
        textcoords="offset points", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#27ae60"),
    )
    ax.set_xlabel("K"); ax.set_ylabel("Error Rate")
    ax.set_title("KNN: K vs Error Rate", fontsize=14, fontweight="bold")
    ax.set_xticks(ks)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "knn_k_vs_error.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()


def plot_rf_importance(model, cols):
    fi = pd.Series(model.feature_importances_, index=cols)
    fi.sort_values(inplace=True)
    top = fi.tail(min(20, len(fi)))
    n = len(top)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_arr = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n))
    bars = ax.barh(range(n), top.values, color=cmap_arr,
                edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(top.index, fontsize=10)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {n} Feature Importance (Random Forest)",
                fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                "%.4f" % val, va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()


def check_data(df):
    print("--- Checking data quality ---")
    nrows, ncols = df.shape
    print("rows: %d, cols: %d" % (nrows, ncols))

    n_dup = df.duplicated().sum()
    if n_dup:
        print("WARNING: found %d duplicate rows!" % n_dup)
    if ID_COL in df.columns:
        dup_ids = df[ID_COL].duplicated().sum()
        if dup_ids:
            print("WARNING: %d duplicate customerIDs" % dup_ids)
    else:
        dup_ids = 0
        print("WARNING: ID column '%s' not found; duplicate ID check skipped"
            % ID_COL)
    _log.append({
        "step": "audit",
        "description": "dupes: %d rows, %d IDs" % (n_dup, dup_ids),
    })

    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss):
        print("Missing values found:")
        for col_name, cnt in miss.items():
            print("  %s: %d" % (col_name, cnt))
    else:
        print("No missing values (good)")

    try:
        tc_bad = pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum()
        print("TotalCharges has %d bad entries (whitespace)" % tc_bad)
        _log.append({
            "step": "audit",
            "description": "TotalCharges: %d non-numeric" % tc_bad,
        })
    except KeyError:
        print("TotalCharges column not found, skipping check")

    churn_vc = df[TARGET_COL].value_counts()
    rate = churn_vc.get('Yes', 0) / len(df) * 100
    print("\nChurn rate: %.1f%%" % rate)
    for lab, cnt in churn_vc.items():
        pct = cnt / len(df) * 100
        print("  %s: %d (%.1f%%)" % (lab, cnt, pct))
    _log.append({
        "step": "audit",
        "description": "churn rate=%.1f%%" % rate,
    })

    # outlier check (IQR)
    print("\nChecking outliers...")
    tmp = df.copy()
    tmp["TotalCharges"] = pd.to_numeric(tmp["TotalCharges"], errors="coerce")
    for col in NUMERIC_FEATURES_RAW:
        v = tmp[col].dropna()
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        spread = q3 - q1
        outlier_ct = ((v < q1 - 1.5 * spread) | (v > q3 + 1.5 * spread)).sum()
        if outlier_ct > 0:
            print("  %s: %d outliers (IQR=%.1f)" % (col, outlier_ct, spread))
        _log.append({
            "step": "audit",
            "description": "%s: IQR=%.1f, outliers=%d" % (col, spread, outlier_ct),
            "details": {"Q1": float(q1), "Q3": float(q3),
                        "outliers": int(outlier_ct)},
        })

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c"]
    vc = df[TARGET_COL].value_counts()
    bars = ax.bar(vc.index, vc.values, color=colors,
                edgecolor="white", linewidth=2, width=0.5)
    for b, c in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                f"{c}\n({c / len(df) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Customer Churn Distribution",
                fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    if len(miss) > 0:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.barh(miss.index.astype(str), miss.values,
                color="#3498db", edgecolor="white")
        ax2.set_title("Missing Values Before Cleaning",
                    fontsize=14, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "missing_values.png"),
                    dpi=_DPI, bbox_inches="tight")
        plt.close()

    print("Audit done.")
    return df


def preprocess(df):
    print("\nCleaning + feature engineering")
    df = df.copy()

    df.drop(columns=[ID_COL], inplace=True)
    _log.append({"step": "clean",
                "description": "Dropped '%s'" % ID_COL})

    # drop bad rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_bad = df["TotalCharges"].isna().sum()
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    _log.append({"step": "clean",
                "description": f"TotalCharges: removed {n_bad} bad rows"})

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    _log.append({"step": "clean",
                "description": "SeniorCitizen mapped 0/1 -> No/Yes"})

    print("Building new features...")

    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)  # +1 avoids div/0
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    svc_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = sum(
        (df[c].isin(["Yes", "Fiber optic", "DSL"])).astype(int)
        for c in svc_cols
    )

    df["HasProtectionBundle"] = (
        (df["OnlineSecurity"] == "Yes")
        & (df["OnlineBackup"] == "Yes")
        & (df["DeviceProtection"] == "Yes")
        & (df["TechSupport"] == "Yes")
    ).astype(int)

    df["HighRiskContract"] = (
        (df["Contract"] == "Month-to-month")
        & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    df["HasStreaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    _log.extend([
        {"step": "engineer", "description": "AvgMonthlySpend = TotalCharges/(tenure+1)"},
        {"step": "engineer", "description": "ChargeRatio = Monthly/(Total+1)"},
        {"step": "engineer", "description": "ServiceCount from %d cols" % len(svc_cols)},
        {"step": "engineer", "description": "HasProtectionBundle"},
        {"step": "engineer", "description": "HighRiskContract"},
        {"step": "engineer", "description": "HasStreaming"},
    ])

    churn_bin = (df[TARGET_COL] == TARGET_POSITIVE).astype(int)
    print("Correlations with churn:")
    eng = ENGINEERED_NUMERIC_FEATURES + ENGINEERED_BINARY_FEATURES
    for feat in eng:
        r = df[feat].corr(churn_bin)
        print("  %-25s -> %+.4f" % (feat, r))

    num_all = NUMERIC_FEATURES_RAW + ENGINEERED_NUMERIC_FEATURES + ENGINEERED_BINARY_FEATURES
    corr_df = df[num_all].copy()
    corr_df[TARGET_COL] = churn_bin
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap (incl. Engineered)",
                fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, col in enumerate(NUMERIC_FEATURES_RAW):
        a = axes[idx]
        for lab, clr in [(TARGET_NEGATIVE, "#2ecc71"),
                        (TARGET_POSITIVE, "#e74c3c")]:
            vals = df[df[TARGET_COL] == lab][col].dropna()
            a.hist(vals, bins=30, alpha=0.6, label=lab,
                color=clr, edgecolor="white")
        a.set_title(col, fontsize=12, fontweight="bold")
        a.legend(fontsize=9)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.suptitle("Numeric Features by Churn Status",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_distributions.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    key_cats = ["Contract", "InternetService",
                "PaymentMethod", "PaperlessBilling"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, col in enumerate(key_cats):
        r, c = divmod(idx, 2)
        a = axes[r][c]
        ct = pd.crosstab(df[col], df[TARGET_COL], normalize="index") * 100
        ct.plot(kind="bar", stacked=True,
                color=["#2ecc71", "#e74c3c"], ax=a, edgecolor="white")
        a.set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
        a.set_ylabel("Percentage (%)")
        a.set_xticklabels(a.get_xticklabels(), rotation=30, ha="right")
        a.legend(["Retained", "Churned"], fontsize=9)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "categorical_churn_rates.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    print("Shape after cleaning:", df.shape)
    return df


def encode_split(df):
    print("\nEncoding + splitting...")

    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Target mapping:", label_map)
    _log.append({"step": "encode",
                "description": f"Target encoded: {label_map}"})

    n_cat = len(ALL_CATEGORICAL_FEATURES)
    print("One-hot encoding %d cat features..." % n_cat)
    df = pd.get_dummies(df, columns=ALL_CATEGORICAL_FEATURES,
                        drop_first=True, dtype=int)
    fcols = [c for c in df.columns if c != TARGET_COL]
    print("  -> %d features total" % len(fcols))
    _log.append({
        "step": "encode",
        "description": "OHE %d cats -> %d features" % (n_cat, len(fcols)),
    })

    X = df[fcols].copy()
    y = df[TARGET_COL].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    print("Split: train=%d, test=%d, features=%d" % (
        X_tr.shape[0], X_te.shape[0], X_tr.shape[1]))
    print("  churn rate -> train: %.3f, test: %.3f" % (
        y_tr.mean(), y_te.mean()))
    _log.append({
        "step": "split",
        "description": f"80/20 split: train={X_tr.shape[0]} test={X_te.shape[0]}",
    })

    # Fit scaler for reference plot and app.py deployment;
    # actual scaling happens inside each model pipeline to prevent CV leakage.
    scaler = StandardScaler()
    num_in = [c for c in NUMERIC_FEATURES if c in X.columns]
    scaler.fit(X_tr[num_in])
    _log.append({"step": "scale",
                "description": f"StandardScaler fitted on {len(num_in)} numeric cols "
                               "(applied inside pipeline, not pre-applied)"})

    pre_scale = {}
    for c in NUMERIC_FEATURES_RAW:
        if c in X_tr.columns:
            pre_scale[c] = X_tr[c].values.copy()
    scaled_preview = pd.DataFrame(
        scaler.transform(X_tr[num_in]), columns=num_in, index=X_tr.index)

    ncols = len(NUMERIC_FEATURES_RAW)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 7))
    for i, col in enumerate(NUMERIC_FEATURES_RAW):
        axes[0, i].hist(pre_scale[col], bins=30, color="#3498db",
                        alpha=0.7, edgecolor="white")
        axes[0, i].set_title(col + "\n(Before)", fontsize=10,
                            fontweight="bold")
        axes[0, i].spines["top"].set_visible(False)
        axes[0, i].spines["right"].set_visible(False)

        axes[1, i].hist(scaled_preview[col].values, bins=30, color="#e74c3c",
                        alpha=0.7, edgecolor="white")
        axes[1, i].set_title(col + "\n(After)", fontsize=10,
                            fontweight="bold")
        for sp in ("top", "right"):
            axes[1, i].spines[sp].set_visible(False)
    fig.suptitle("Before vs After StandardScaler (Training Data Only)",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scaling_comparison.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    enc_info = {
        "feature_columns": fcols,
        "numeric_features": NUMERIC_FEATURES,
        "numeric_features_raw": NUMERIC_FEATURES_RAW,
        "engineered_numeric": ENGINEERED_NUMERIC_FEATURES,
        "engineered_binary": ENGINEERED_BINARY_FEATURES,
        "encoded_feature_names": list(X.columns),
        "class_mapping": label_map,
        "label_classes": list(le.classes_),
    }
    joblib.dump(enc_info, os.path.join(MODELS_DIR, "encoder_info.pkl"))

    neg_class = label_map.get(TARGET_NEGATIVE, 0)
    loyal_idx = y_tr.index[y_tr == neg_class]
    loyal_medians = {col: float(df.loc[loyal_idx, col].median())
                    for col in NUMERIC_FEATURES_RAW if col in df.columns}
    joblib.dump(loyal_medians, os.path.join(MODELS_DIR, "healthy_profile.pkl"))

    print("Saved scaler, encoder_info, healthy_profile")
    return X_tr, X_te, y_tr, y_te, fcols, scaler


def rfecv(X_train, X_test, y_train, fcols):
    print("\nRunning RFECV feature selection...")

    base_rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    selector = RFECV(
        estimator=base_rf, step=1,
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1", n_jobs=-1,
        min_features_to_select=MIN_RFE_FEATURES, verbose=0,
    )

    n_feats = X_train.shape[1]
    print("RFECV on %d features (min=%d)..." % (n_feats, MIN_RFE_FEATURES))
    ts = time.time()
    selector.fit(X_train.values, y_train.values)
    wall = time.time() - ts

    mask = selector.support_
    ranks = selector.ranking_
    kept = [f for f, s in zip(fcols, mask) if s]
    dropped = [f for f, s in zip(fcols, mask) if not s]

    print(f"Done in {wall:.1f}s -> {len(kept)}/{len(fcols)} features kept")
    if dropped:
        print("Dropped %d features:" % len(dropped))
        for feat in dropped:
            ri = fcols.index(feat)
            print("  - %s (rank=%d)" % (feat, ranks[ri]))

    _log.append({"step": "rfe",
                "description": f"RFECV: {len(kept)}/{len(fcols)} kept"})
    _log.append({"step": "rfe_removed",
                "description": f"Removed: {dropped}"})

    X_tr_sel = X_train[kept]
    X_te_sel = X_test[kept]

    info = joblib.load(os.path.join(MODELS_DIR, "encoder_info.pkl"))
    info["selected_features"] = kept
    info["removed_features"] = dropped
    info["rfecv_n_features"] = int(selector.n_features_)
    joblib.dump(info, os.path.join(MODELS_DIR, "encoder_info.pkl"))
    print("Saved updated encoder_info")

    fig, (ax_cv, ax_rank) = plt.subplots(1, 2, figsize=(16, 6))

    cv_scores = selector.cv_results_["mean_test_score"]
    x_range = range(MIN_RFE_FEATURES, MIN_RFE_FEATURES + len(cv_scores))
    ax_cv.plot(x_range, cv_scores, "o-", color="#3498db", lw=2, ms=4)
    ax_cv.axvline(selector.n_features_, color="#e74c3c", ls="--", lw=2,
                label="Optimal: %d features" % selector.n_features_)
    ax_cv.set_xlabel("Number of Features", fontsize=11)
    ax_cv.set_ylabel("CV F1 Score", fontsize=11)
    ax_cv.set_title("RFECV: Cross-Validation Score",
                    fontsize=12, fontweight="bold")
    ax_cv.legend(fontsize=10)
    ax_cv.spines["top"].set_visible(False)
    ax_cv.spines["right"].set_visible(False)
    ax_cv.grid(alpha=0.3)

    ranked = sorted(zip(fcols, ranks), key=lambda t: t[1])
    show_n = min(25, len(ranked))
    top_ranked = ranked[:show_n]
    bar_colors = ["#27ae60" if r == 1 else "#e74c3c"
                for _, r in top_ranked]
    ax_rank.barh([f for f, _ in top_ranked],
                [r for _, r in top_ranked],
                color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_rank.set_xlabel("Ranking (1 = Selected)", fontsize=11)
    ax_rank.set_title("Feature Rankings (green=kept, red=removed)",
                    fontsize=12, fontweight="bold")
    ax_rank.spines["top"].set_visible(False)
    ax_rank.spines["right"].set_visible(False)

    fig.suptitle("Recursive Feature Elimination with CV (RFECV)",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rfe_feature_ranking.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    return X_tr_sel, X_te_sel, kept


def train_models(X_train, y_train, fcols):
    print("\n*** Training models ***")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {}
    names = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
    clfs = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1),
    }

    cv_f1 = {}

    for i, name in enumerate(names, 1):
        clf = clfs[name]
        print(f"\n  [{i}/5] {name}...")
        t0 = time.time()
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", clf),
        ])
        search = RandomizedSearchCV(
            pipe, PARAM_DISTRIBUTIONS[name],
            n_iter=RANDOMIZED_N_ITER, cv=cv, scoring="f1",
            n_jobs=-1, verbose=0, random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
        models[name] = search.best_estimator_
        cv_f1[name] = search.best_score_

        bp = {k.replace("model__", ""): v
            for k, v in search.best_params_.items()}
        print("    Best params:", bp)
        print("    CV F1=%.4f (%.1fs)" % (search.best_score_, time.time() - t0))

    print("Generating evidence plots...")

    c_vals = RAW_C_VALUES
    c_accs = []
    cv_ev = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    for c in c_vals:
        p = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(
                C=c, solver="lbfgs", max_iter=3000,
                random_state=RANDOM_STATE, class_weight="balanced")),
        ])
        scores = cross_val_score(p, X_train, y_train, cv=cv_ev,
                                 scoring="accuracy", n_jobs=-1)
        c_accs.append(scores.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(c_vals, c_accs, "o-",
            color=MODEL_COLORS["Logistic Regression"], linewidth=2, markersize=8)
    bi = np.argmax(c_accs)
    ax.plot(c_vals[bi], c_accs[bi], "s",
            color="#e74c3c", markersize=14, zorder=5)
    ax.annotate(f"Best C={c_vals[bi]}\nAcc={c_accs[bi]:.4f}",
                xy=(c_vals[bi], c_accs[bi]),
                xytext=(15, 15), textcoords="offset points",
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Logistic Regression: C vs Accuracy",
                fontsize=14, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lr_c_vs_accuracy.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    knn_plot(X_train, y_train)

    dt = models["Decision Tree"].named_steps["model"]
    fig_dt, ax_dt = plt.subplots(figsize=(20, 10))
    plot_tree(dt, max_depth=3, feature_names=fcols,
            class_names=[TARGET_NEGATIVE, TARGET_POSITIVE],
            filled=True, rounded=True, fontsize=8, ax=ax_dt,
            proportion=True, impurity=True)
    ax_dt.set_title("Decision Tree (Top 3 Levels)",
                    fontsize=16, fontweight="bold")
    fig_dt.tight_layout()
    fig_dt.savefig(os.path.join(PLOTS_DIR, "decision_tree_structure.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    rf = models["Random Forest"].named_steps["model"]
    plot_rf_importance(rf, fcols)

    print("Learning curves...")
    fig_lc, axes_lc = plt.subplots(2, 2, figsize=(14, 10))
    for idx, name in enumerate(names):
        ax = axes_lc.flatten()[idx]
        sizes, train_sc, val_sc = learning_curve(
            models[name], X_train, y_train, cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 8),
            scoring="f1", n_jobs=-1,
        )
        ax.fill_between(sizes,
                        train_sc.mean(1) - train_sc.std(1),
                        train_sc.mean(1) + train_sc.std(1),
                        alpha=0.15, color=MODEL_COLORS[name])
        ax.fill_between(sizes,
                        val_sc.mean(1) - val_sc.std(1),
                        val_sc.mean(1) + val_sc.std(1),
                        alpha=0.15, color="#e74c3c")
        ax.plot(sizes, train_sc.mean(1), "o-",
                color=MODEL_COLORS[name], label="Train", lw=2, ms=4)
        ax.plot(sizes, val_sc.mean(1), "o-",
                color="#e74c3c", label="Validation", lw=2, ms=4)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Size"); ax.set_ylabel("F1")
        ax.legend(loc="best", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig_lc.suptitle("Learning Curves (F1, SMOTE-in-CV)",
                    fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    # ensemble
    print(f"\n  [5/5] Ensemble Voting...")
    t_ens = time.time()
    weights = []
    est_list = []
    abbr = {"Logistic Regression": "lr", "KNN": "knn",
            "Decision Tree": "dt", "Random Forest": "rf"}
    for name in names:
        weights.append(cv_f1[name])
        est_list.append((abbr[name], models[name]))

    print("    CV F1 weights:", dict(
        zip(names, ["%.4f" % w for w in weights])))
    ensemble = VotingClassifier(
        estimators=est_list, voting="soft", weights=weights,
    )
    ensemble.fit(X_train, y_train)
    models["Ensemble Voting"] = ensemble
    print("    Ensemble fitted (%.1fs)" % (time.time() - t_ens))

    print("Models trained.\n")
    return models


def kfold_threshold_search(models, X_train, y_train, n_splits=5):
    print("K-Fold Threshold Search (%d folds)..." % n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_STATE)

    model_names = list(models.keys())
    fold_thresholds = {n: [] for n in model_names}
    fold_val_f1 = {n: [] for n in model_names}

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        print("  Fold %d/%d..." % (fold_i + 1, n_splits))
        X_tr_fold = X_train.iloc[tr_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_tr_fold = y_train.iloc[tr_idx]
        y_val_fold = y_train.iloc[val_idx]

        for name in model_names:
            pipe_clone = clone(models[name])
            pipe_clone.fit(X_tr_fold, y_tr_fold)

            yb = pipe_clone.predict_proba(X_val_fold)[:, 1]
            pr_prec, pr_rec, pr_thr = precision_recall_curve(y_val_fold, yb)

            if len(pr_thr) > 0:
                f1_arr = (2 * pr_prec[:-1] * pr_rec[:-1]
                          / (pr_prec[:-1] + pr_rec[:-1] + 1e-8))
                best_i = int(np.argmax(f1_arr))
                best_t = float(pr_thr[best_i])
                val_f1 = float(f1_arr[best_i])
            else:
                best_t = 0.5
                val_f1 = float(f1_score(
                    y_val_fold, (yb >= 0.5).astype(int), zero_division=0))

            fold_thresholds[name].append(best_t)
            fold_val_f1[name].append(val_f1)

    avg_thr = {}
    avg_f1 = {}
    std_f1 = {}

    print("\n  %-25s  AvgThr  AvgF1   StdF1   Per-fold F1" % "Model")
    print("  " + "-" * 80)
    for name in model_names:
        avg_thr[name] = float(np.mean(fold_thresholds[name]))
        avg_f1[name] = float(np.mean(fold_val_f1[name]))
        std_f1[name] = float(np.std(fold_val_f1[name]))
        fold_str = " ".join(["%.4f" % f for f in fold_val_f1[name]])
        print("  %-25s  %.3f   %.4f  %.4f  [%s]" % (
            name, avg_thr[name], avg_f1[name], std_f1[name], fold_str))

    _log.append({
        "step": "kfold_threshold",
        "description": "K-Fold threshold search (%d folds)" % n_splits,
        "details": {
            n: {"avg_threshold": avg_thr[n], "avg_f1": avg_f1[n],
                "std_f1": std_f1[n]}
            for n in model_names
        },
    })

    return avg_thr, avg_f1, std_f1, fold_val_f1


def evaluate(trained, X_test, y_test, fcols, opt_thr, avg_val_f1):
    print("Evaluating models on held-out test set...")
    results = {}
    ordered = [n for n in MODEL_NAMES if n in trained]
    n_models = len(ordered)

    cm_cols = 3
    cm_rows = (n_models + cm_cols - 1) // cm_cols
    fig_cm, axes_cm = plt.subplots(cm_rows, cm_cols,
                                figsize=(5 * cm_cols, 4.5 * cm_rows))
    cm_flat = axes_cm.flatten() if n_models > 1 else [axes_cm]

    for idx, name in enumerate(ordered):
        pipe = trained[name]
        best_t = opt_thr[name]
        val_f1_opt = avg_val_f1[name]

        yp = pipe.predict(X_test)
        yb = pipe.predict_proba(X_test)[:, 1]

        m = {
            "Accuracy":  float(accuracy_score(y_test, yp)),
            "Precision": float(precision_score(y_test, yp, zero_division=0)),
            "Recall":    float(recall_score(y_test, yp, zero_division=0)),
            "F1-Score":  float(f1_score(y_test, yp, zero_division=0)),
            "AUC":       float(roc_auc_score(y_test, yb)),
            "Log Loss":  float(log_loss(y_test, yb)),
        }

        yp_opt = (yb >= best_t).astype(int)
        m["Accuracy_opt"]  = float(accuracy_score(y_test, yp_opt))
        m["Precision_opt"] = float(precision_score(y_test, yp_opt, zero_division=0))
        m["Recall_opt"]    = float(recall_score(y_test, yp_opt, zero_division=0))
        m["F1-Score_opt"]  = float(f1_score(y_test, yp_opt, zero_division=0))
        m["Optimal_Threshold"] = best_t
        m["Validation_F1_opt"] = val_f1_opt
        results[name] = m

        print("\n  %s (threshold=0.5 / kfold-avg=%.3f):" % (name, best_t))
        print("    K-Fold Avg Validation F1: %.4f" % val_f1_opt)
        print("    Default:  Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f" % (
            m["Accuracy"], m["Precision"], m["Recall"], m["F1-Score"]))
        print("    Optimal:  Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f" % (
            m["Accuracy_opt"], m["Precision_opt"],
            m["Recall_opt"], m["F1-Score_opt"]))
        print(f"    AUC={m['AUC']:.4f} LogLoss={m['Log Loss']:.4f}")

        cm = confusion_matrix(y_test, yp_opt)
        ax = cm_flat[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=[TARGET_NEGATIVE, TARGET_POSITIVE],
                    yticklabels=[TARGET_NEGATIVE, TARGET_POSITIVE],
                    cbar=False,
                    annot_kws={"size": 16, "fontweight": "bold"})
        ax.set_title(
            "%s (thr=%.3f)\nAcc=%.3f Rec=%.3f F1=%.3f" % (
                name, best_t,
                m["Accuracy_opt"], m["Recall_opt"], m["F1-Score_opt"]),
            fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        # highlight FN
        if cm.shape[0] > 1 and cm[1][0] > 0:
            ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False,
                                    edgecolor="red", linewidth=3))

    for j in range(n_models, len(cm_flat)):
        cm_flat[j].set_visible(False)

    fig_cm.suptitle(
        "Confusion Matrices (Optimal Thresholds)\n"
        "(Red box = Missed Churners)",
        fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
    for name in ordered:
        yb = trained[name].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yb)
        auc_val = roc_auc_score(y_test, yb)
        ax_roc.plot(fpr, tpr,
                    label="%s (AUC=%.4f)" % (name, auc_val),
                    color=MODEL_COLORS.get(name, "#333"), lw=2.5)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.spines["top"].set_visible(False)
    ax_roc.spines["right"].set_visible(False)
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    fig_pr, ax_pr = plt.subplots(figsize=(8, 7))
    for name in ordered:
        yb = trained[name].predict_proba(X_test)[:, 1]
        p_c, r_c, _ = precision_recall_curve(y_test, yb)
        ax_pr.plot(r_c, p_c, label=name,
                color=MODEL_COLORS.get(name, "#333"), lw=2.5)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves",
                    fontsize=14, fontweight="bold")
    ax_pr.legend(loc="upper right", fontsize=9)
    for s in ("top", "right"):
        ax_pr.spines[s].set_visible(False)
    ax_pr.grid(alpha=0.3)
    fig_pr.tight_layout()
    fig_pr.savefig(os.path.join(PLOTS_DIR, "precision_recall_curves.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    fig_bar, ax_bar = plt.subplots(figsize=(16, 6))
    bar_metrics = ["Accuracy_opt", "Precision_opt",
                "Recall_opt", "F1-Score_opt", "AUC"]
    bar_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    x_pos = np.arange(len(bar_metrics))
    w = 0.15
    for i, name in enumerate(ordered):
        vals = [results[name][bm] for bm in bar_metrics]
        brs = ax_bar.bar(x_pos + i * w, vals, w, label=name,
                        color=MODEL_COLORS.get(name, "#333"),
                        edgecolor="white", linewidth=0.5)
        for b, v in zip(brs, vals):
            ax_bar.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.005,
                        "%.3f" % v, ha="center", va="bottom",
                        fontsize=7, fontweight="bold")
    ax_bar.set_xticks(x_pos + w * (n_models - 1) / 2)
    ax_bar.set_xticklabels(bar_labels, fontsize=11)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Model Performance (Optimal Thresholds)",
                    fontsize=14, fontweight="bold")
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "metrics_comparison.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()

    thr_cols = 3
    thr_rows = (n_models + thr_cols - 1) // thr_cols
    fig_th, axes_th = plt.subplots(thr_rows, thr_cols,
                                figsize=(5 * thr_cols, 4 * thr_rows))
    th_flat = axes_th.flatten() if n_models > 1 else [axes_th]
    for idx, name in enumerate(ordered):
        ax = th_flat[idx]
        yb = trained[name].predict_proba(X_test)[:, 1]
        thr_range = np.linspace(0.1, 0.9, 50)
        f1s, recs, precs = [], [], []
        for t in thr_range:
            yt = (yb >= t).astype(int)
            f1s.append(f1_score(y_test, yt, zero_division=0))
            recs.append(recall_score(y_test, yt, zero_division=0))
            precs.append(precision_score(y_test, yt, zero_division=0))
        ax.plot(thr_range, f1s, label="F1", color="#3498db", lw=2)
        ax.plot(thr_range, recs, label="Recall", color="#e74c3c", lw=2)
        ax.plot(thr_range, precs, label="Precision", color="#27ae60", lw=2)
        ax.axvline(opt_thr[name], color="#7f8c8d", ls="--", lw=2,
                label="Val-optimal (%.3f)" % opt_thr[name])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for j in range(n_models, len(th_flat)):
        th_flat[j].set_visible(False)
    fig_th.suptitle("Threshold Curves on Test Set (Val-Optimised Threshold)",
                    fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "threshold_optimisation.png"),
                dpi=_DPI, bbox_inches="tight")
    plt.close()
    print("Saved evaluation plots")

    print("Running SHAP...")
    try:
        rf_pipe = trained["Random Forest"]
        rf_model = rf_pipe.named_steps["model"]

        scaler_step = rf_pipe.named_steps.get("scaler")
        if scaler_step is not None:
            X_shap = pd.DataFrame(
                scaler_step.transform(X_test),
                columns=X_test.columns, index=X_test.index)
        else:
            X_shap = X_test

        explainer = shap.TreeExplainer(rf_model)
        sv = explainer.shap_values(X_shap.values)

        if isinstance(sv, list):
            sv_churn = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv_churn = sv[:, :, 1]
        else:
            sv_churn = sv

        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(ev[1])
        else:
            ev = float(ev)
        joblib.dump(ev, os.path.join(MODELS_DIR, "shap_expected_value.pkl"))

        shap.summary_plot(sv_churn, X_shap.values,
                        feature_names=list(X_test.columns),
                        show=False, max_display=20)
        fig_shap = plt.gcf()
        fig_shap.set_size_inches(12, 8)
        fig_shap.tight_layout()
        fig_shap.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"),
                        dpi=_DPI, bbox_inches="tight")
        plt.close("all")
        print("Saved SHAP summary")
    except Exception as exc:
        print("SHAP failed:", exc)

    base_only = {k: v for k, v in results.items() if k in BASE_MODEL_NAMES}
    champion = max(
        base_only,
        key=lambda k: (
            base_only[k]["F1-Score_opt"],
            base_only[k]["Recall_opt"],
            base_only[k]["AUC"],
        ),
    )
    cm = results[champion]
    print("\n--- Base Models Champion (4 individual models) ---")
    print("  Champion: %s (Baseline: %s)" % (champion, BASELINE_MODEL))
    print("  Test-F1=%.4f, Recall=%.4f, AUC=%.4f, KFold-Val-F1=%.4f" % (
        cm["F1-Score_opt"], cm["Recall_opt"],
        cm["AUC"], cm["Validation_F1_opt"]))

    if "Ensemble Voting" in results:
        ens = results["Ensemble Voting"]
        print("\n--- Ensemble Voting (Extra - Team Collaboration) ---")
        print("  Test-F1=%.4f, Recall=%.4f, AUC=%.4f, KFold-Val-F1=%.4f" % (
            ens["F1-Score_opt"], ens["Recall_opt"],
            ens["AUC"], ens["Validation_F1_opt"]))
        delta_f1 = ens["F1-Score_opt"] - cm["F1-Score_opt"]
        print("  vs Champion (%s): F1 %+.4f" % (champion, delta_f1))

    return results, champion, opt_thr


def save_outputs(trained, metrics, best_name, thresholds, df_raw):
    print("\nSaving everything...")
    for name, pipe in trained.items():
        fn = MODEL_FILE_KEYS.get(name)
        if fn:
            joblib.dump(pipe, os.path.join(MODELS_DIR, fn))

    metrics["_champion"] = best_name
    metrics["_optimal_thresholds"] = thresholds
    metrics["_champion_basis"] = (
        "highest held-out test F1 at K-fold averaged threshold")
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("  [SAVED] metrics.json")

    with open(os.path.join(MODELS_DIR, "cleaning_log.json"), "w") as fh:
        json.dump(_log, fh, indent=2, ensure_ascii=False)
    print("  [SAVED] cleaning_log.json (%d steps)" % len(_log))

    joblib.dump(thresholds, os.path.join(MODELS_DIR, "optimal_thresholds.pkl"))
    print("  [SAVED] optimal_thresholds.pkl")
    print("Done saving.")


def main():
    t_start = time.time()
    print("\nTelco Churn - Training Pipeline")
    print("BMDS2003 Data Science\n")
    print_env_info()

    # load
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df_raw = pd.read_csv(DATA_PATH)
    _log.append({"step": "load",
                "description": f"Loaded {df_raw.shape[0]}x{df_raw.shape[1]} from CSV"})
    print("Got %d rows, %d cols" % (df_raw.shape[0], df_raw.shape[1]))

    check_data(df_raw)
    df = preprocess(df_raw)
    X_train, X_test, y_train, y_test, fcols, scaler = encode_split(df)

    X_train, X_test, sel_cols = rfecv(X_train, X_test, y_train, fcols)
    models = train_models(X_train, y_train, sel_cols)

    print("\n*** K-Fold Threshold Optimisation (5-fold) ***")
    opt_thr, avg_val_f1, std_val_f1, fold_details = kfold_threshold_search(
        models, X_train, y_train, n_splits=5)

    metrics, best, thresholds = evaluate(
        models, X_test, y_test, sel_cols, opt_thr, avg_val_f1)

    metrics["_kfold_details"] = {
        name: {
            "fold_f1": [float(f) for f in fold_details.get(name, [])],
            "avg_f1": float(avg_val_f1.get(name, 0)),
            "std_f1": float(std_val_f1.get(name, 0)),
            "avg_threshold": float(opt_thr.get(name, 0.5)),
        }
        for name in models.keys()
    }

    save_outputs(models, metrics, best, thresholds, df_raw)

    total_time = time.time() - t_start
    print("\nDone in %.1fs, best model: %s" % (total_time, best))
    print("Models saved to:", MODELS_DIR)
    print("Plots saved to:", PLOTS_DIR)
    print("%d cleaning steps logged" % len(_log))


if __name__ == "__main__":
    main()
