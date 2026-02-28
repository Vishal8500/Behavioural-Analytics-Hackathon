import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Data + Model
# -----------------------------
df = pd.read_csv("data/engineered_behaviour_features.csv")

with open("models/burnout_model.pkl", "rb") as f:
    burnout_model = pickle.load(f)

X = df.drop(["student_id", "burnout_label", "dropout_label"], axis=1)

# -----------------------------
# Feature Importance
# -----------------------------
feature_importance = pd.Series(
    burnout_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# Save top drivers
os.makedirs("outputs", exist_ok=True)
feature_importance.to_csv("outputs/feature_importance.csv")

# -----------------------------
# Plot Top 10 Features
# -----------------------------
plt.figure()
feature_importance.head(10).plot(kind="bar")
plt.title("Top Behavioural Risk Drivers")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")

print("Explainability Analysis Saved")