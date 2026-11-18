
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load processed dataset
df = pd.read_csv(
    ROOT / "data/processed/flowcast_features.csv"
)

# Select input features
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

# Target variable
target = "traffic_volume"

# Check required columns
required_columns = features + [target]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Keep required columns
model_data = df[required_columns].copy()

# Remove rows with missing target
model_data = model_data.dropna(subset=[target])

# Save prepared dataset
output_path = ROOT / "data/processed/model_data.csv"
model_data.to_csv(output_path, index=False)

# Display report
print("\nDATA PREPARATION COMPLETED")
print("=" * 40)
print("Total rows:", len(model_data))
print("Total features:", len(features))
print("Target:", target)

print("\nMissing values:")
print(model_data.isnull().sum())

print("\nDataset preview:")
print(model_data.head())

print("\nSaved to:", output_path)
