
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "interim"

traffic = pd.read_csv(DATA_DIR / "traffic_cleaned.csv")
weather = pd.read_csv(DATA_DIR / "weather_cleaned.csv")

# Display column names
print("\n--- TRAFFIC COLUMNS ---")
print(traffic.columns.tolist())

print("\n--- WEATHER COLUMNS ---")
print(weather.columns.tolist())

# Check station IDs
print("\n--- TRAFFIC STATION IDs ---")
print(traffic["weather_station_id"].head(10).to_list())
print("Data type:", traffic["weather_station_id"].dtype)

print("\n--- WEATHER STATION IDs ---")
print(weather["station_id"].head(10).to_list())
print("Data type:", weather["station_id"].dtype)

# Normalize station IDs
traffic["weather_station_id"] = (
    traffic["weather_station_id"].astype(str).str.strip().str.upper()
)

weather["station_id"] = (
    weather["station_id"].astype(str).str.strip().str.upper()
)

# Normalize dates
traffic["date"] = pd.to_datetime(
    traffic["date"], errors="coerce", dayfirst=True
)

weather["date"] = pd.to_datetime(
    weather["date"], errors="coerce", dayfirst=True
)

# Extract hours
traffic["hour"] = pd.to_datetime(
    traffic["time"].astype(str).str.strip(),
    errors="coerce"
).dt.hour

weather["hour"] = pd.to_datetime(
    weather["time"].astype(str).str.strip(),
    errors="coerce"
).dt.hour

# Compare matching keys
traffic_keys = traffic[
    ["weather_station_id", "date", "hour"]
].drop_duplicates()

weather_keys = weather[
    ["station_id", "date", "hour"]
].drop_duplicates()

traffic_keys = traffic_keys.dropna()
weather_keys = weather_keys.dropna()

traffic_keys = traffic_keys.rename(
    columns={"weather_station_id": "station_id"}
)

matched = traffic_keys.merge(
    weather_keys,
    on=["station_id", "date", "hour"],
    how="inner"
)

print("\n--- JOIN DIAGNOSTICS ---")
print("Unique traffic keys:", len(traffic_keys))
print("Unique weather keys:", len(weather_keys))
print("Matching keys:", len(matched))

print("\nTraffic station IDs:")
print(sorted(traffic["weather_station_id"].unique())[:20])

print("\nWeather station IDs:")
print(sorted(weather["station_id"].unique())[:20])

print("\nTraffic date range:")
print(traffic["date"].min(), "to", traffic["date"].max())

print("\nWeather date range:")
print(weather["date"].min(), "to", weather["date"].max())

print("\nTraffic hours:")
print(sorted(traffic["hour"].dropna().unique()))

print("\nWeather hours:")
print(sorted(weather["hour"].dropna().unique()))
