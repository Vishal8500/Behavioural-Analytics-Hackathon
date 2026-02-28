import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Final Predictions
# -----------------------------
df = pd.read_csv("outputs/final_with_interventions.csv")

# -----------------------------
# Load Original Synthetic Dataset (Correct Filename)
# -----------------------------
raw_df = pd.read_csv("data/synthetic_student_burnout_data_2000.csv")

# Get profile_type per student
profile_map = raw_df[["student_id", "profile_type"]].drop_duplicates()

# Merge profile info
df = df.merge(profile_map, on="student_id", how="left")

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1️⃣ Risk Segmentation
# -----------------------------
def segment(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

df["risk_segment"] = df["risk_score"].apply(segment)

seg_counts = df["risk_segment"].value_counts()

plt.figure()
seg_counts.plot(kind="bar")
plt.title("Risk Segmentation Distribution")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("outputs/risk_segmentation.png")

# -----------------------------
# 2️⃣ Top 10 Highest Risk Students
# -----------------------------
top10 = df.sort_values("risk_score", ascending=False).head(10)
top10.to_csv("outputs/top_10_high_risk_students.csv", index=False)

# -----------------------------
# 3️⃣ Profile-wise Risk Comparison
# -----------------------------
profile_risk = df.groupby("profile_type")["risk_score"].mean()

plt.figure()
profile_risk.plot(kind="bar")
plt.title("Average Risk Score by Behaviour Profile")
plt.ylabel("Average Risk Score")
plt.tight_layout()
plt.savefig("outputs/profile_risk_comparison.png")

print("Advanced Visualizations Saved Successfully")