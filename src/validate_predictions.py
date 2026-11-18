
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_PATH = ROOT / "data/processed/flowcast_features.csv"
HISTORICAL_PATH = ROOT / "data/processed/flowcast_historical_features.csv"

REGRESSION_PATH = ROOT / "models/best_traffic_model.pkl"
CLASSIFIER_PATH = ROOT / "models/best_congestion_classifier.pkl"

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# ==========================================
# 2. MODEL FEATURES
# ==========================================

HISTORY_COLUMNS = [
    "traffic_volume_lag_1",
    "traffic_volume_lag_2",
    "traffic_volume_lag_3",
    "avg_speed_lag_1",
    "avg_speed_lag_2",
    "traffic_rolling_mean_3",
    "traffic_rolling_mean_6",
    "traffic_change"
]

FEATURES = [
    "road_id", "hour", "minute", "day_of_week", "month",
    "is_weekend", "is_morning_rush", "is_evening_rush",
    "temperature", "rainfall", "visibility", "weather_condition",
    "public_holiday", "event_flag", "roadwork_flag"
] + HISTORY_COLUMNS

KEYS = ["road_id", "date", "time"]

print("\n========== FLOWCAST MODEL VALIDATION ==========")

# ==========================================
# 3. LOAD AND MERGE DATA
# ==========================================

original = pd.read_csv(ORIGINAL_PATH)
historical = pd.read_csv(HISTORICAL_PATH)

df = original.merge(
    historical[KEYS + HISTORY_COLUMNS],
    on=KEYS,
    how="inner",
    validate="one_to_one"
)

print("Merged dataset:", df.shape)

# ==========================================
# 4. LOAD SAVED MODELS
# ==========================================

regression_model = joblib.load(REGRESSION_PATH)
classifier_model = joblib.load(CLASSIFIER_PATH)

# ==========================================
# 5. REGRESSION VALIDATION
# Match compare_regression_models.py
# ==========================================

reg_df = df.copy()

reg_df["traffic_volume"] = pd.to_numeric(
    reg_df["traffic_volume"],
    errors="coerce"
)

reg_df["datetime"] = pd.to_datetime(
    reg_df["date"].astype(str) + " " +
    reg_df["time"].astype(str),
    errors="coerce"
)

reg_df = reg_df.dropna(
    subset=["traffic_volume", "datetime"]
).copy()

reg_df = reg_df.sort_values(
    "datetime"
).reset_index(drop=True)

split_index = int(len(reg_df) * 0.80)

reg_test = reg_df.iloc[split_index:].copy()

X_reg = reg_test[FEATURES]
y_reg = reg_test["traffic_volume"]

reg_predictions = regression_model.predict(X_reg)

mae = mean_absolute_error(y_reg, reg_predictions)
rmse = mean_squared_error(y_reg, reg_predictions) ** 0.5
r2 = r2_score(y_reg, reg_predictions)

print("\n========== REGRESSION VALIDATION ==========")
print("Training records:", split_index)
print("Testing records:", len(reg_test))
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

reg_results = reg_test[
    ["road_id", "date", "time", "traffic_volume"]
].copy()

reg_results["predicted_traffic_volume"] = reg_predictions

reg_results.to_csv(
    REPORTS / "regression_validation.csv",
    index=False
)

# ==========================================
# 6. CLASSIFICATION VALIDATION
# Match compare_classifiers.py
# ==========================================

class_df = df.copy()

label_mapping = {
    "free-flow": "Free-flow",
    "free flow": "Free-flow",
    "moderate": "Moderate",
    "heavy": "Heavy",
    "severe": "Severe"
}

class_df["congestion_level"] = (
    class_df["congestion_level"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(label_mapping)
)

class_df = class_df.dropna(
    subset=["congestion_level"]
).copy()

class_df["datetime"] = pd.to_datetime(
    class_df["date"].astype(str) + " " +
    class_df["time"].astype(str),
    errors="coerce"
)

class_df = class_df.dropna(
    subset=["datetime"]
)

class_df = class_df.sort_values(
    "datetime"
).reset_index(drop=True)

split_index = int(len(class_df) * 0.80)

class_test = class_df.iloc[split_index:].copy()

X_class = class_test[FEATURES]
y_class = class_test["congestion_level"]

class_predictions = classifier_model.predict(X_class)

accuracy = accuracy_score(y_class, class_predictions)

weighted_f1 = f1_score(
    y_class,
    class_predictions,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_class,
    class_predictions,
    average="macro",
    zero_division=0
)

print("\n========== CLASSIFICATION VALIDATION ==========")
print("Training records:", split_index)
print("Testing records:", len(class_test))
print(f"Accuracy:    {accuracy:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")

class_results = class_test[
    ["road_id", "date", "time", "congestion_level"]
].copy()

class_results["predicted_congestion"] = class_predictions

class_results.to_csv(
    REPORTS / "classification_validation.csv",
    index=False
)

# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n========== VALIDATION REPORTS SAVED ==========")
print("Regression: reports/regression_validation.csv")
print("Classification: reports/classification_validation.csv")

print("\nFLOWCAST VALIDATION COMPLETED!")
