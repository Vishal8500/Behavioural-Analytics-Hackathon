import pandas as pd

df = pd.read_csv("outputs/final_predictions.csv")

def recommend(row):
    if row["risk_score"] > 75:
        return "Immediate advisor meeting + Counseling referral"
    elif row["risk_score"] > 50:
        return "Academic monitoring + Faculty check-in"
    else:
        return "Normal monitoring"

df["recommended_intervention"] = df.apply(recommend, axis=1)

df.to_csv("outputs/final_with_interventions.csv", index=False)

print("Interventions Added")