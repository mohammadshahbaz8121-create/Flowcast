
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parents[1]

# Load trained model
MODEL_PATH = ROOT / "models" / "traffic_model.pkl"

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
preprocessor = saved_data["preprocessor"]
features = saved_data["features"]

# Load dataset
DATA_PATH = ROOT / "data" / "processed" / "flowcast_features.csv"
df = pd.read_csv(DATA_PATH)

# Prepare features
X = df[features].copy()

categorical_features = ["road_id", "weather_condition"]
numeric_features = [
    col for col in features if col not in categorical_features
]

# Handle missing values in the same way as training
for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(X[col].median())

# Transform features
X_processed = preprocessor.transform(X)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Extract feature importance
importance = model.feature_importances_

# Create importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 features
top_features = importance_df.head(15)

print("\n========== FEATURE IMPORTANCE ==========")
print(top_features.to_string(index=False))

# Save feature importance results
output_csv = ROOT / "reports" / "feature_importance.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)

importance_df.to_csv(output_csv, index=False)

# Create visualization
plt.figure(figsize=(12, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("FlowCast - Top 15 Traffic Prediction Features")

plt.tight_layout()

# Save graph
output_image = ROOT / "reports" / "figures" / "feature_importance.png"
output_image.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_image, dpi=300)
plt.show()

print("\nFeature importance analysis completed!")
print("CSV saved:", output_csv)
print("Graph saved:", output_image)
