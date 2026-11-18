
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project directory
ROOT = Path(__file__).resolve().parents[1]

# Load processed dataset
DATA_PATH = ROOT / "data" / "processed" / "flowcast_features.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# Create datetime for chronological sorting
df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str),
    errors="coerce"
)

df = df.dropna(subset=["datetime", "traffic_volume"]).copy()
df = df.sort_values("datetime").reset_index(drop=True)

# Features
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

# Prepare X and y
X = df[features].copy()
y = pd.to_numeric(df[target], errors="coerce")

# Handle missing feature values
categorical_features = ["road_id", "weather_condition"]
numeric_features = [
    col for col in features if col not in categorical_features
]

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(X[col].median())

# Chronological 80/20 split
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Preprocessing
preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "numeric",
        "passthrough",
        numeric_features
    )
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train model
print("\nTraining FlowCast model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

# Predictions
predictions = model.predict(X_test_processed)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save model
model_output = ROOT / "models" / "traffic_model.pkl"
model_output.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor,
        "features": features
    },
    model_output
)

print("\nModel saved successfully!")
print("Location:", model_output)
