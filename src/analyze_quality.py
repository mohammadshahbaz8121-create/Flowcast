
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"

files = {
    "TRAFFIC": "traffic_sensor_log.csv",
    "WEATHER": "weather_observations.csv",
    "CALENDAR": "calendar_events.csv"
}

datasets = {}

for name, filename in files.items():
    datasets[name] = pd.read_csv(DATA_DIR / filename)

print("\n========== FLOWCAST DATA QUALITY REPORT ==========")

for name, df in datasets.items():

    print(f"\n\n========== {name} ==========")

    print("Dataset shape:", df.shape)

    # Missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values")

    # Duplicate records
    print("\nDuplicate records:")
    if name == "TRAFFIC":
        keys = ["road_id", "date", "time"]
    elif name == "WEATHER":
        keys = ["station_id", "date", "time"]
    else:
        keys = ["date"]

    if all(column in df.columns for column in keys):
        print("Duplicate key records:", df.duplicated(subset=keys).sum())
    else:
        print("Expected duplicate-check columns not found.")

    # Traffic-specific checks
    if name == "TRAFFIC":
        print("\nInvalid traffic measurements:")

        checks = {
            "Negative traffic volume": ("traffic_volume", lambda x: x < 0),
            "Speed above 200 km/h": ("avg_speed", lambda x: x > 200),
            "Occupancy above 100%": ("occupancy", lambda x: x > 100)
        }

        for label, (column, condition) in checks.items():
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="coerce")
                print(f"{label}: {condition(values).sum()}")

print("\n========== QUALITY ANALYSIS COMPLETED ==========")
