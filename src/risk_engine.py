import pickle
import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\Behavioural_Hackathon\data\engineered_behaviour_features.csv")

with open("models/burnout_model.pkl", "rb") as f:
    burnout_model = pickle.load(f)

with open("models/dropout_model.pkl", "rb") as f:
    dropout_model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

X = df.drop(["student_id", "burnout_label", "dropout_label"], axis=1)

burnout_probs = burnout_model.predict_proba(X)
dropout_probs = dropout_model.predict_proba(X)[:, 1]

drift = abs(df["login_drop_pct"]) + abs(df["attendance_drop_pct"]) + abs(df["delay_increase_pct"])

risk_score = (0.4 * burnout_probs.max(axis=1) + 0.4 * dropout_probs + 0.2 * drift) * 100

df["risk_score"] = np.round(risk_score, 2)
df["burnout_prediction"] = le.inverse_transform(burnout_model.predict(X))
df["dropout_probability"] = np.round(dropout_probs, 3)

df.to_csv("outputs/final_predictions.csv", index=False)

print("Risk Scoring Complete")