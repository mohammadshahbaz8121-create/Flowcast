
import pandas as pd

df = pd.read_csv("data/processed/flowcast_features.csv")

print("Original dataset:", df.shape)

df["datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["time"].astype(str),
    errors="coerce"
)

df["traffic_volume"] = pd.to_numeric(
    df["traffic_volume"], errors="coerce"
)

df["avg_speed"] = pd.to_numeric(
    df["avg_speed"], errors="coerce"
)

df = df.sort_values(
    ["road_id", "datetime"]
).reset_index(drop=True)

# Prepare output columns
historical_columns = [
    "traffic_volume_lag_1",
    "traffic_volume_lag_2",
    "traffic_volume_lag_3",
    "avg_speed_lag_1",
    "avg_speed_lag_2",
    "traffic_rolling_mean_3",
    "traffic_rolling_mean_6",
    "traffic_change"
]

for col in historical_columns:
    df[col] = float("nan")

# Process each road independently
for road_id, indices in df.groupby("road_id").groups.items():

    g = df.loc[indices].copy()

    # Gap between current and previous reading
    intervals = g["datetime"].diff().dt.total_seconds() / 60

    # True only when the interval is exactly 30 minutes
    valid_interval = intervals.eq(30)

    # Volume lag features
    for lag in [1, 2, 3]:

        valid_chain = (
            valid_interval.astype(int)
            .rolling(lag, min_periods=lag)
            .sum()
            .eq(lag)
        )

        values = g["traffic_volume"].shift(lag)

        df.loc[indices, f"traffic_volume_lag_{lag}"] = (
            values.where(valid_chain).values
        )

    # Speed lag features
    for lag in [1, 2]:

        valid_chain = (
            valid_interval.astype(int)
            .rolling(lag, min_periods=lag)
            .sum()
            .eq(lag)
        )

        values = g["avg_speed"].shift(lag)

        df.loc[indices, f"avg_speed_lag_{lag}"] = (
            values.where(valid_chain).values
        )

    # Rolling averages using only consecutive readings
    for window in [3, 6]:

        valid_chain = (
            valid_interval.astype(int)
            .rolling(window, min_periods=window)
            .sum()
            .eq(window)
        )

        rolling_mean = (
            g["traffic_volume"]
            .shift(1)
            .rolling(window, min_periods=window)
            .mean()
        )

        df.loc[
            indices, f"traffic_rolling_mean_{window}"
        ] = rolling_mean.where(valid_chain).values

# Change from the previous valid reading
df["traffic_change"] = (
    df["traffic_volume_lag_1"]
    - df["traffic_volume_lag_2"]
)

# Save separate historical dataset
df.to_csv(
    "data/processed/flowcast_historical_features.csv",
    index=False
)

print("\n========== FINAL HISTORICAL FEATURE CHECK ==========")

print(df[historical_columns].isnull().sum())

print("\nFinal dataset shape:", df.shape)

print(
    "\nSaved: data/processed/flowcast_historical_features.csv"
)

print("\nHISTORICAL FEATURE ENGINEERING COMPLETED!")
