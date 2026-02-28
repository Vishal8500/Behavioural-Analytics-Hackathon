import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/engineered_behaviour_features.csv")

X = df.drop(["student_id", "burnout_label", "dropout_label"], axis=1)

# -----------------------------
# Load Model
# -----------------------------
with open("models/burnout_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Create Explainer
# -----------------------------
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Identify index of HIGH class
# -----------------------------
# Assuming encoded labels: Low=0, Medium=1, High=2
high_class_index = 2

# Extract SHAP values for HIGH class only
high_shap = shap_values[:, :, high_class_index]

# -----------------------------
# 1️⃣ Global Summary Plot (High Risk Class)
# -----------------------------
plt.figure()
shap.plots.beeswarm(high_shap, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary_high_class.png")
plt.close()

# -----------------------------
# 2️⃣ Global Feature Importance (Bar)
# -----------------------------
plt.figure()
shap.plots.bar(high_shap, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_feature_importance_high_class.png")
plt.close()

print("SHAP Global Explainability (High Risk Class) Saved")

# -----------------------------
# 3️⃣ Explain Top High-Risk Student
# -----------------------------
final_df = pd.read_csv("outputs/final_predictions.csv")

top_student_id = final_df.sort_values("risk_score", ascending=False)["student_id"].iloc[0]

student_index = df[df["student_id"] == top_student_id].index[0]

plt.figure()
shap.plots.waterfall(high_shap[student_index], show=False)
plt.tight_layout()
plt.savefig("outputs/shap_top_student_waterfall.png")
plt.close()

print("SHAP Individual Explanation Saved")
print("Top Risk Student ID:", top_student_id)