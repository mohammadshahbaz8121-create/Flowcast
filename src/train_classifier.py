
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/processed/flowcast_features.csv")

print("Dataset loaded:", df.shape)

features = [
    "road_id", "hour", "minute", "day_of_week", "month",
    "is_weekend", "is_morning_rush", "is_evening_rush",
    "temperature", "rainfall", "visibility", "weather_condition",
    "public_holiday", "event_flag", "roadwork_flag"
]

target = "congestion_level"

# ==========================================
# 2. CLEAN TARGET LABELS
# ==========================================

df[target] = (
    df[target]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.replace("_", "-", regex=False)
)

label_map = {
    "free-flow": "Free-flow",
    "free flow": "Free-flow",
    "moderate": "Moderate",
    "heavy": "Heavy",
    "severe": "Severe"
}

df[target] = df[target].map(label_map)
df = df.dropna(subset=[target]).copy()

# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================

df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str),
    errors="coerce"
)

df = df.dropna(subset=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

X = df[features]
y = df[target]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

# ==========================================
# 4. PREPROCESSING
# ==========================================

categorical_features = ["road_id", "weather_condition"]

numeric_features = [
    col for col in features
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ),
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features
        )
    ]
)

# ==========================================
# 5. TEST THREE CONFIGURATIONS
# ==========================================

experiments = {
    "Baseline balanced": {
        "class_weight": "balanced",
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },
    "Balanced subsample": {
        "class_weight": "balanced_subsample",
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },
    "Larger leaf balanced": {
        "class_weight": "balanced",
        "min_samples_leaf": 2,
        "max_features": 0.8
    }
}

results = []
best_score = -1
best_pipeline = None
best_name = None

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

for name, params in experiments.items():

    print(f"\nTraining: {name}")

    classifier = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        **params
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Weighted F1": weighted_f1,
        "Macro F1": macro_f1
    })

    print("Accuracy:", round(accuracy, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Macro F1:", round(macro_f1, 4))

    # Select using macro F1 to give each class equal importance.
    if macro_f1 > best_score:
        best_score = macro_f1
        best_pipeline = pipeline
        best_name = name
        best_predictions = predictions

# ==========================================
# 6. COMPARE RESULTS
# ==========================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1", ascending=False
)

print("\n========== CLASSIFIER COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/classifier_comparison.csv",
    index=False
)

# ==========================================
# 7. BEST MODEL EVALUATION
# ==========================================

print("\n========== BEST CLASSIFIER ==========")
print("Best model:", best_name)

print("\nClassification Report:")
print(classification_report(
    y_test,
    best_predictions,
    labels=["Free-flow", "Moderate", "Heavy", "Severe"],
    zero_division=0
))

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["Free-flow", "Moderate", "Heavy", "Severe"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Free-flow", "Moderate", "Heavy", "Severe"],
    columns=["Free-flow", "Moderate", "Heavy", "Severe"]
)

cm_df.to_csv("reports/confusion_matrix_improved.csv")

# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_pipeline,
    "models/best_congestion_classifier.pkl"
)

print("\nBest classifier saved:")
print("models/best_congestion_classifier.pkl")

print("\nComparison saved:")
print("reports/classifier_comparison.csv")

print("\nImproved confusion matrix saved:")
print("reports/confusion_matrix_improved.csv")

print("\nCLASSIFIER EXPERIMENTS COMPLETED!")
