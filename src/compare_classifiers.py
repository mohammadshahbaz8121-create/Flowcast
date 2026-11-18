
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASETS
# ==========================================

original = pd.read_csv(
    "data/processed/flowcast_features.csv"
)

historical = pd.read_csv(
    "data/processed/flowcast_historical_features.csv"
)

print("Original dataset:", original.shape)
print("Historical dataset:", historical.shape)

# ==========================================
# 2. MERGE HISTORICAL FEATURES
# ==========================================

keys = ["road_id", "date", "time"]

history_columns = [
    "traffic_volume_lag_1",
    "traffic_volume_lag_2",
    "traffic_volume_lag_3",
    "avg_speed_lag_1",
    "avg_speed_lag_2",
    "traffic_rolling_mean_3",
    "traffic_rolling_mean_6",
    "traffic_change"
]

history_data = historical[
    keys + history_columns
].copy()

df = original.merge(
    history_data,
    on=keys,
    how="inner",
    validate="one_to_one"
)

print("\nMerged dataset:", df.shape)

# ==========================================
# 3. PREPARE TARGET
# ==========================================

label_mapping = {
    "free-flow": "Free-flow",
    "free flow": "Free-flow",
    "moderate": "Moderate",
    "heavy": "Heavy",
    "severe": "Severe"
}

df["congestion_level"] = (
    df["congestion_level"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(label_mapping)
)

df = df.dropna(
    subset=["congestion_level"]
).copy()

# Sort chronologically
df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str),
    errors="coerce"
)

df = df.dropna(subset=["datetime"])

df = df.sort_values(
    "datetime"
).reset_index(drop=True)

# ==========================================
# 4. DEFINE FEATURES
# ==========================================

base_features = [
    "road_id",
    "hour",
    "minute",
    "day_of_week",
    "month",
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "temperature",
    "rainfall",
    "visibility",
    "weather_condition",
    "public_holiday",
    "event_flag",
    "roadwork_flag"
]

historical_features = base_features + history_columns

target = "congestion_level"

# Ensure all expected columns exist
required_columns = historical_features + [target]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ==========================================
# 5. CHRONOLOGICAL TRAIN/TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

y_train = train_df[target]
y_test = test_df[target]

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining class distribution:")
print(y_train.value_counts())

# ==========================================
# 6. MODEL TRAINING FUNCTION
# ==========================================

def train_and_evaluate(model_name, features):

    categorical_features = [
        "road_id",
        "weather_condition"
    ]

    numeric_features = [
        col for col in features
        if col not in categorical_features
    ]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        ))
    ])

    preprocessor = ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    X_train = train_df[features]
    X_test = test_df[features]

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test, predictions
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"\n========== {model_name} RESULTS ==========")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=[
            "Free-flow",
            "Moderate",
            "Heavy",
            "Severe"
        ],
        zero_division=0
    )

    print(report)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "Free-flow",
            "Moderate",
            "Heavy",
            "Severe"
        ]
    )

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "name": model_name,
        "model": model,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "predictions": predictions,
        "confusion_matrix": matrix
    }


# ==========================================
# 7. TRAIN ORIGINAL MODEL
# ==========================================

original_result = train_and_evaluate(
    "ORIGINAL RANDOM FOREST",
    base_features
)

# ==========================================
# 8. TRAIN HISTORICAL MODEL
# ==========================================

historical_result = train_and_evaluate(
    "HISTORICAL FEATURE RANDOM FOREST",
    historical_features
)

# ==========================================
# 9. SAVE COMPARISON RESULTS
# ==========================================

comparison = pd.DataFrame([
    {
        "Model": original_result["name"],
        "Accuracy": original_result["accuracy"],
        "Weighted F1": original_result["weighted_f1"],
        "Macro F1": original_result["macro_f1"]
    },
    {
        "Model": historical_result["name"],
        "Accuracy": historical_result["accuracy"],
        "Weighted F1": historical_result["weighted_f1"],
        "Macro F1": historical_result["macro_f1"]
    }
])

comparison.to_csv(
    "reports/historical_model_comparison.csv",
    index=False
)

# Save confusion matrices
labels = [
    "Free-flow",
    "Moderate",
    "Heavy",
    "Severe"
]

for result, filename in [
    (
        original_result,
        "original_confusion_matrix.csv"
    ),
    (
        historical_result,
        "historical_confusion_matrix.csv"
    )
]:

    matrix_df = pd.DataFrame(
        result["confusion_matrix"],
        index=labels,
        columns=labels
    )

    matrix_df.to_csv(
        f"reports/{filename}"
    )

# Save historical model
joblib.dump(
    historical_result["model"],
    "models/historical_congestion_classifier.pkl"
)

# ==========================================
# 10. FINAL COMPARISON
# ==========================================

print("\n========== FINAL MODEL COMPARISON ==========")

print(comparison.to_string(index=False))

print("\n========== PERFORMANCE CHANGE ==========")

print(
    "Accuracy change:",
    round(
        historical_result["accuracy"]
        - original_result["accuracy"],
        4
    )
)

print(
    "Weighted F1 change:",
    round(
        historical_result["weighted_f1"]
        - original_result["weighted_f1"],
        4
    )
)

print(
    "Macro F1 change:",
    round(
        historical_result["macro_f1"]
        - original_result["macro_f1"],
        4
    )
)

print("\nAll comparison reports saved successfully!")
