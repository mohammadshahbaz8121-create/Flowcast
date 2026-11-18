
import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/processed/flowcast_features.csv")

print("Dataset shape:", df.shape)

# ==========================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "traffic_volume",
    "road_capacity",
    "congestion_level"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ==========================================
# 3. CLEAN VALUES
# ==========================================

df["traffic_volume"] = pd.to_numeric(
    df["traffic_volume"], errors="coerce"
)

df["road_capacity"] = pd.to_numeric(
    df["road_capacity"], errors="coerce"
)

df["congestion_level"] = (
    df["congestion_level"]
    .astype("string")
    .str.strip()
)

df = df.dropna(subset=required_columns).copy()

# ==========================================
# 4. CALCULATE CONGESTION RATIO
# ==========================================

# Traffic is recorded every 30 minutes.
# Compare traffic volume with half-hour road capacity.

df = df[df["road_capacity"] > 0].copy()

df["congestion_ratio"] = (
    df["traffic_volume"] / (df["road_capacity"] / 2)
)

# ==========================================
# 5. GENERATE EXPECTED LABELS
# ==========================================

def classify_congestion(ratio):

    if ratio < 0.50:
        return "Free-flow"

    elif ratio < 0.80:
        return "Moderate"

    elif ratio < 1.00:
        return "Heavy"

    else:
        return "Severe"


df["expected_congestion"] = (
    df["congestion_ratio"].apply(classify_congestion)
)

# ==========================================
# 6. COMPARE EXISTING AND EXPECTED LABELS
# ==========================================

print("\n========== EXISTING LABEL DISTRIBUTION ==========")
print(df["congestion_level"].value_counts())

print("\n========== EXPECTED LABEL DISTRIBUTION ==========")
print(df["expected_congestion"].value_counts())

comparison = pd.crosstab(
    df["congestion_level"],
    df["expected_congestion"],
    rownames=["Existing Label"],
    colnames=["Expected Label"]
)

print("\n========== LABEL COMPARISON ==========")
print(comparison)

# ==========================================
# 7. LABEL AGREEMENT
# ==========================================

agreement = (
    df["congestion_level"] == df["expected_congestion"]
).mean()

print("\nLabel Agreement:", round(agreement * 100, 2), "%")

# ==========================================
# 8. RATIO STATISTICS BY EXISTING LABEL
# ==========================================

print("\n========== RATIO STATISTICS ==========")

print(
    df.groupby("congestion_level")["congestion_ratio"]
    .agg(["count", "mean", "min", "max"])
    .round(3)
)

# ==========================================
# 9. SAVE ANALYSIS
# ==========================================

comparison.to_csv("reports/congestion_label_comparison.csv")

df.groupby("congestion_level")["congestion_ratio"].agg(
    ["count", "mean", "min", "max"]
).to_csv("reports/congestion_ratio_statistics.csv")

print("\nAnalysis files saved in reports/")
print("\nCONGESTION ANALYSIS COMPLETED!")
