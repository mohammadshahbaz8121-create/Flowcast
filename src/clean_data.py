
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def clean_traffic():
    print("\nCleaning traffic data...")

    df = pd.read_csv(RAW_DIR / "traffic_sensor_log.csv")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert date column
    df["date"] = pd.to_datetime(
        df["date"], dayfirst=True, errors="coerce"
    )

    # Remove duplicate road/time records
    before = len(df)
    df = df.drop_duplicates(
        subset=["road_id", "date", "time"],
        keep="first"
    )
    print("Duplicate records removed:", before - len(df))

    # Convert measurements to numeric values
    for column in ["traffic_volume", "avg_speed", "occupancy"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Mark invalid measurements as missing
    df.loc[df["traffic_volume"] < 0, "traffic_volume"] = pd.NA
    df.loc[
        (df["avg_speed"] < 0) | (df["avg_speed"] > 200),
        "avg_speed"
    ] = pd.NA
    df.loc[
        (df["occupancy"] < 0) | (df["occupancy"] > 100),
        "occupancy"
    ] = pd.NA

    output = INTERIM_DIR / "traffic_cleaned.csv"
    df.to_csv(output, index=False)

    print("Traffic data saved:", output)
    print("Remaining rows:", len(df))


def clean_weather():
    print("\nCleaning weather data...")

    df = pd.read_csv(RAW_DIR / "weather_observations.csv")

    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(
        df["date"], dayfirst=True, errors="coerce"
    )

    # Standardize weather labels
    df["weather_condition"] = (
        df["weather_condition"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    output = INTERIM_DIR / "weather_cleaned.csv"
    df.to_csv(output, index=False)

    print("Weather data saved:", output)
    print("Remaining rows:", len(df))


def clean_calendar():
    print("\nCleaning calendar data...")

    df = pd.read_csv(RAW_DIR / "calendar_events.csv")

    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(
        df["date"], dayfirst=True, errors="coerce"
    )

    output = INTERIM_DIR / "calendar_cleaned.csv"
    df.to_csv(output, index=False)

    print("Calendar data saved:", output)
    print("Remaining rows:", len(df))


if __name__ == "__main__":
    print("========== FLOWCAST DATA CLEANING ==========")

    clean_traffic()
    clean_weather()
    clean_calendar()

    print("\n========== DATA CLEANING COMPLETED ==========")
