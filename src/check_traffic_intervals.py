
import pandas as pd

# ==========================================
# 1. LOAD HISTORICAL DATA
# ==========================================

df = pd.read_csv(
    "data/processed/flowcast_historical_features.csv"
)

print("Dataset loaded:", df.shape)

# ==========================================
# 2. CREATE DATETIME
# ==========================================

df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str),
    errors="coerce"
)

df = df.dropna(subset=["datetime"])

df = df.sort_values(
    ["road_id", "datetime"]
).reset_index(drop=True)

# ==========================================
# 3. CALCULATE TIME INTERVALS
# ==========================================

df["interval_minutes"] = (
    df.groupby("road_id")["datetime"]
    .diff()
    .dt.total_seconds() / 60
)

# ==========================================
# 4. INTERVAL DISTRIBUTION
# ==========================================

print("\n========== INTERVAL DISTRIBUTION ==========")

print(
    df["interval_minutes"]
    .value_counts()
    .sort_index()
    .head(20)
)

# ==========================================
# 5. CHECK 30-MINUTE INTERVALS
# ==========================================

valid_intervals = df["interval_minutes"].dropna()

total_intervals = len(valid_intervals)

correct_intervals = (
    valid_intervals == 30
).sum()

percentage = (
    correct_intervals / total_intervals * 100
    if total_intervals > 0 else 0
)

print("\n========== INTERVAL VALIDATION ==========")

print("Total intervals:", total_intervals)
print("30-minute intervals:", correct_intervals)

print(
    "30-minute interval percentage:",
    round(percentage, 2), "%"
)

# ==========================================
# 6. IDENTIFY IRREGULAR INTERVALS
# ==========================================

irregular = df[
    df["interval_minutes"].notna()
    & (df["interval_minutes"] != 30)
]

print("\n========== IRREGULAR INTERVALS ==========")

print("Irregular intervals:", len(irregular))

if not irregular.empty:
    print(
        irregular[
            ["road_id", "datetime", "interval_minutes"]
        ].head(15).to_string(index=False)
    )

# ==========================================
# 7. SAVE RESULTS
# ==========================================

irregular.to_csv(
    "reports/irregular_traffic_intervals.csv",
    index=False
)

print("\nSaved: reports/irregular_traffic_intervals.csv")

print("\nINTERVAL VALIDATION COMPLETED!")
