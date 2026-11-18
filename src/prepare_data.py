
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Load processed dataset
df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "flowcast_features.csv"
)

# Select useful input features
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
    "roadwork_flag",
]

target = "traffic_volume"

# Check available columns
features = [col for col in features if col in df.columns]

print("\n========== FEATURE SELECTION ==========")

print("\nSelected input features:")
print(features)

print("\nPrediction target:")
print(target)

print("\nInput feature count:", len(features))

print("\nTarget statistics:")
print(df[target].describe())

print("\nMissing values in selected features:")
print(df[features].isnull().sum())

print("\n========== FEATURE SELECTION COMPLETED ==========")
