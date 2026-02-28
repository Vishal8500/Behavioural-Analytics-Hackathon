import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("outputs/final_with_interventions.csv")

os.makedirs("outputs", exist_ok=True)

# Risk distribution
plt.figure()
df["risk_score"].hist(bins=20)
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Number of Students")
plt.savefig("outputs/risk_distribution.png")

# Burnout distribution
plt.figure()
df["burnout_prediction"].value_counts().plot(kind="bar")
plt.title("Burnout Level Distribution")
plt.savefig("outputs/burnout_distribution.png")

print("Visualizations Saved in outputs/")