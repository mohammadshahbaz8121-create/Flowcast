
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
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

df = original.merge(
    historical[keys + history_columns],
    on=keys,
    how="inner",
    validate="one_to_one"
)

print("\nMerged dataset:", df.shape)

# ==========================================
# 3. PREPARE TARGET
# ==========================================

df["traffic_volume"] = pd.to_numeric(
    df["traffic_volume"],
    errors="coerce"
)

df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str),
    errors="coerce"
)

df = df.dropna(
    subset=["traffic_volume", "datetime"]
).copy()

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

target = "traffic_volume"

# ==========================================
# 5. CHRONOLOGICAL SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

y_train = train_df[target]
y_test = test_df[target]

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

# ==========================================
# 6. TRAINING FUNCTION
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
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
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
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    print(f"\nTraining {model_name}...")

    model.fit(
        train_df[features],
        y_train
    )

    predictions = model.predict(
        test_df[features]
    )

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    r2 = r2_score(
        y_test, predictions
    )

    print(f"\n========== {model_name} RESULTS ==========")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return {
        "name": model_name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions
    }


# ==========================================
# 7. ORIGINAL REGRESSION MODEL
# ==========================================

original_result = train_and_evaluate(
    "ORIGINAL RANDOM FOREST REGRESSOR",
    base_features
)

# ==========================================
# 8. HISTORICAL REGRESSION MODEL
# ==========================================

historical_result = train_and_evaluate(
    "HISTORICAL FEATURE RANDOM FOREST REGRESSOR",
    historical_features
)

# ==========================================
# 9. SAVE COMPARISON
# ==========================================

comparison = pd.DataFrame([
    {
        "Model": original_result["name"],
        "MAE": original_result["mae"],
        "RMSE": original_result["rmse"],
        "R2 Score": original_result["r2"]
    },
    {
        "Model": historical_result["name"],
        "MAE": historical_result["mae"],
        "RMSE": historical_result["rmse"],
        "R2 Score": historical_result["r2"]
    }
])

comparison.to_csv(
    "reports/historical_regression_comparison.csv",
    index=False
)

# Save historical model separately for evaluation
joblib.dump(
    historical_result["model"],
    "models/historical_traffic_model.pkl"
)

# Save actual and predicted values
predictions_df = pd.DataFrame({
    "actual_traffic_volume": y_test.values,
    "original_prediction": original_result["predictions"],
    "historical_prediction": historical_result["predictions"]
})

predictions_df.to_csv(
    "reports/regression_predictions_comparison.csv",
    index=False
)

# ==========================================
# 10. FINAL COMPARISON
# ==========================================

print("\n========== FINAL REGRESSION COMPARISON ==========")

print(comparison.to_string(index=False))

print("\n========== PERFORMANCE CHANGE ==========")

print(
    "MAE improvement:",
    round(
        original_result["mae"] - historical_result["mae"],
        4
    )
)

print(
    "RMSE improvement:",
    round(
        original_result["rmse"] - historical_result["rmse"],
        4
    )
)

print(
    "R2 improvement:",
    round(
        historical_result["r2"] - original_result["r2"],
        4
    )
)

print("\nRegression comparison completed successfully!")
