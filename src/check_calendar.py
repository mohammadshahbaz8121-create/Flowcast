
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "interim"

traffic = pd.read_csv(DATA_DIR / "traffic_cleaned.csv")
calendar = pd.read_csv(DATA_DIR / "calendar_cleaned.csv")

# Parse dates using the same method as features.py
traffic["date"] = pd.to_datetime(
    traffic["date"].astype(str).str.strip(),
    format="mixed",
    dayfirst=True,
    errors="coerce"
).dt.normalize()

calendar["date"] = pd.to_datetime(
    calendar["date"].astype(str).str.strip(),
    format="mixed",
    dayfirst=True,
    errors="coerce"
).dt.normalize()

print("\n--- CALENDAR JOIN DIAGNOSTICS ---")

print("\nTraffic date range:")
print(traffic["date"].min(), "to", traffic["date"].max())

print("\nCalendar date range:")
print(calendar["date"].min(), "to", calendar["date"].max())

print("\nUnique traffic dates:", traffic["date"].nunique())
print("Unique calendar dates:", calendar["date"].nunique())

matching_dates = set(traffic["date"].dropna()) & set(calendar["date"].dropna())

print("Matching dates:", len(matching_dates))

print("\nFirst 10 traffic dates:")
print(traffic["date"].drop_duplicates().head(10).to_list())

print("\nFirst 10 calendar dates:")
print(calendar["date"].drop_duplicates().head(10).to_list())

print("\nCalendar columns:")
print(calendar.columns.tolist())
