# config.py - BMDS2003

import os
import sys
from scipy.stats import uniform, randint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Telco_Customer_Churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

RANDOM_STATE = 42

TARGET_COL = "Churn"
TARGET_POSITIVE = "Yes"
TARGET_NEGATIVE = "No"

ID_COL = "customerID"


NUMERIC_FEATURES_RAW = ["tenure", "MonthlyCharges", "TotalCharges"]

ENGINEERED_NUMERIC_FEATURES = [
    "AvgMonthlySpend",
    "ChargeRatio",
    "ServiceCount",
]

NUMERIC_FEATURES = NUMERIC_FEATURES_RAW + ENGINEERED_NUMERIC_FEATURES

BINARY_CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling",
]

MULTI_CATEGORICAL_FEATURES = [
    "MultipleLines",    "InternetService",
    "OnlineSecurity",   "OnlineBackup",
    "DeviceProtection", "TechSupport",
    "StreamingTV",      "StreamingMovies",
    "Contract",         "PaymentMethod",
]

ALL_CATEGORICAL_FEATURES = (
    BINARY_CATEGORICAL_FEATURES + MULTI_CATEGORICAL_FEATURES
)

ENGINEERED_BINARY_FEATURES = [
    "HasProtectionBundle", "HighRiskContract", "HasStreaming",
]


CATEGORY_OPTIONS = {
    "gender":           ["Male", "Female"],
    "SeniorCitizen":    ["No", "Yes"],
    "Partner":          ["Yes", "No"],
    "Dependents":       ["Yes", "No"],
    "PhoneService":     ["Yes", "No"],
    "PaperlessBilling": ["Yes", "No"],
    "MultipleLines":    ["No phone service", "No", "Yes"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["No", "Yes", "No internet service"],
    "OnlineBackup":     ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport":      ["No", "Yes", "No internet service"],
    "StreamingTV":      ["No", "Yes", "No internet service"],
    "StreamingMovies":  ["No", "Yes", "No internet service"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# (min, max, default, step)
NUMERIC_RANGES = {
    "tenure":         (0, 72, 12, 1),
    "MonthlyCharges": (18.0, 120.0, 65.0, 0.5),
    "TotalCharges":   (0.0, 9000.0, 1500.0, 10.0),
}


FEATURE_LABELS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges":  "Monthly Charges ($)",
    "TotalCharges":    "Total Charges ($)",
    "AvgMonthlySpend": "Avg Monthly Spend ($)",
    "ChargeRatio":     "Charge Ratio",
    "ServiceCount":    "Active Services Count",
    "HasProtectionBundle": "Has Protection Bundle",
    "HighRiskContract":    "High-Risk Contract",
    "HasStreaming":        "Has Streaming Services",
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "PhoneService":     "Phone Service",
    "PaperlessBilling": "Paperless Billing",
    "MultipleLines":    "Multiple Lines",
    "InternetService":  "Internet Service",
    "OnlineSecurity":   "Online Security",
    "OnlineBackup":     "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport":      "Tech Support",
    "StreamingTV":      "Streaming TV",
    "StreamingMovies":  "Streaming Movies",
    "Contract":         "Contract Type",
    "PaymentMethod":    "Payment Method",
}


# --- grid search ---

PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "model__solver": ["lbfgs"],
        "model__max_iter": [3000],
        "model__class_weight": ["balanced"],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9, 11, 13, 15, 19, 23],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    },
    "Decision Tree": {
        "model__max_depth":          [3, 5, 7, 10, 15, None],
        "model__min_samples_split":  [2, 5, 10, 20],
        "model__min_samples_leaf":   [1, 2, 4, 8],
        "model__criterion":          ["gini", "entropy"],
        "model__class_weight":       ["balanced"],
    },
    "Random Forest": {
        "model__n_estimators":      [100, 200, 300],
        "model__max_depth":         [5, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__class_weight":      ["balanced"],
    },
}

RAW_C_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

PARAM_DISTRIBUTIONS = {
    "Logistic Regression": {
        "model__C":            uniform(0.001, 20),
        "model__solver":       ["lbfgs"],
        "model__max_iter":     [3000],
        "model__class_weight": ["balanced"],
    },
    "KNN": {
        "model__n_neighbors": randint(3, 30),
        "model__weights": ["uniform", "distance"],
        "model__metric":  ["euclidean", "manhattan", "minkowski"],
    },
    "Decision Tree": {
        "model__max_depth":         [3, 5, 7, 10, 15, 20, None],
        "model__min_samples_split": randint(2, 30),
        "model__min_samples_leaf":  randint(1, 15),
        "model__criterion":         ["gini", "entropy"],
        "model__class_weight":      ["balanced"],
    },
    "Random Forest": {
        "model__n_estimators":      randint(50, 500),
        "model__max_depth":         [5, 10, 15, 20, 25, None],
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf":  randint(1, 10),
        "model__class_weight":      ["balanced"],
    },
}

RANDOMIZED_N_ITER = 80
MIN_RFE_FEATURES = 15


BASE_MODEL_NAMES = [
    "Logistic Regression",
    "KNN",
    "Decision Tree",
    "Random Forest",
]

BASELINE_MODEL = "Logistic Regression"

MODEL_NAMES = BASE_MODEL_NAMES + ["Ensemble Voting"]

MODEL_COLORS = {
    "Logistic Regression": "#636EFA",
    "KNN":                 "#EF553B",
    "Decision Tree":       "#00CC96",
    "Random Forest":       "#AB63FA",
    "Ensemble Voting":     "#FFD700",
}

MODEL_FILE_KEYS = {
    "Logistic Regression": "model_lr.pkl",
    "KNN":                 "model_knn.pkl",
    "Decision Tree":       "model_dt.pkl",
    "Random Forest":       "model_rf.pkl",
    "Ensemble Voting":     "model_voting.pkl",
}


def get_device_info():
    info = {
        "device": "cpu",
        "gpu_available": False,
        "gpu_name": None,
        "cuda_version": None,
        "python_version": "%d.%d.%d" % (
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["device"]       = "cuda"
            info["gpu_available"] = True
            info["gpu_name"]     = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    return info


def print_env_info():
    info = get_device_info()
    rule = "=" * 60
    print(rule)
    print("  Environment Information")
    print(rule)
    print("  Python Version : " + info["python_version"])
    print("  Device         : " + info["device"].upper())
    if info["gpu_available"]:
        print("  GPU Name       : %s" % info["gpu_name"])
        print("  CUDA Version   : %s" % info["cuda_version"])
    else:
        print("  GPU            : Not available (using CPU)")
    print(rule + "\n")
