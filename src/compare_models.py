
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project directory
ROOT = Path(__file__).resolve().parents[1]

# Load dataset
DATA_PATH = ROOT / "data" / "processed" / "flowcast_features.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# Sort data chronologically
df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str),
    errors="coerce"
)

df = df.dropna(subset=["datetime", "traffic_volume"]).copy()
df = df.sort_values("datetime").reset_index(drop=True)

# Select features and target
features = [
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

target = "traffic_volume"

X = df[features].copy()
y = pd.to_numeric(df[target], errors="coerce")

# Remove invalid target rows
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# Feature types
categorical_features = ["road_id", "weather_condition"]
numeric_features = [
    col for col in features if col not in categorical_features
]

# Preprocessing
preprocessor = ColumnTransformer([
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
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numeric_features
    )
])

# Chronological 80/20 split
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Define models
models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=20,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

# Train and evaluate models
results = []
trained_models = {}

for name, algorithm in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print(f"{name} completed!")

# Create comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# Select best model based on lowest RMSE
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\n========== BEST MODEL ==========")
print("Best model:", best_model_name)
print("Lowest RMSE:", round(results_df.iloc[0]["RMSE"], 2))

# Save comparison results
report_path = ROOT / "reports" / "model_comparison.csv"
report_path.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(report_path, index=False)

# Save best model
model_path = ROOT / "models" / "best_traffic_model.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": best_model,
    "features": features,
    "target": target,
    "model_name": best_model_name
}, model_path)

print("\nModel comparison saved:", report_path)
print("Best model saved:", model_path)
print("\nMODEL COMPARISON COMPLETED!")
