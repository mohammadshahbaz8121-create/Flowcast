
from pathlib import Path
import pandas as pd

# Find the project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"

datasets = {
    "TRAFFIC DATA": "traffic_sensor_log.csv",
    "WEATHER DATA": "weather_observations.csv",
    "CALENDAR DATA": "calendar_events.csv"
}

print("\n========== FLOWCAST DATA INSPECTION ==========")

for title, filename in datasets.items():

    print(f"\n\n========== {title} ==========")

    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"File not found: {filename}")
        continue

    df = pd.read_csv(file_path)

    print("Successfully loaded!")
    print("Rows and columns:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

print("\n========== INSPECTION COMPLETED ==========")
