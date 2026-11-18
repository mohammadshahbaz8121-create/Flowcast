
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "flowcast_features.csv"
)

# Create output folder
output_dir = BASE_DIR / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Average traffic by hour
hourly_traffic = df.groupby("hour")["traffic_volume"].mean()

plt.figure(figsize=(10, 5))
hourly_traffic.plot(kind="line", marker="o")
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "traffic_by_hour.png")
plt.show()

# 2. Congestion distribution
congestion = df["congestion_level"].value_counts()

plt.figure(figsize=(8, 5))
congestion.plot(kind="bar")
plt.title("Traffic Congestion Distribution")
plt.xlabel("Congestion Level")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(output_dir / "congestion_distribution.png")
plt.show()

# 3. Average traffic by day
daily_traffic = df.groupby("day_name")["traffic_volume"].mean()

day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

daily_traffic = daily_traffic.reindex(day_order)

plt.figure(figsize=(10, 5))
daily_traffic.plot(kind="bar")
plt.title("Average Traffic Volume by Day")
plt.xlabel("Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(output_dir / "traffic_by_day.png")
plt.show()

print("\nVisualizations completed!")
print("Charts saved in:", output_dir)
