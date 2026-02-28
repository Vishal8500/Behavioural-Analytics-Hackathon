import pandas as pd
import numpy as np
from scipy.stats import linregress

df = pd.read_csv("synthetic_student_burnout_data_2000.csv")

feature_rows = []

for student_id in df["student_id"].unique():
    
    student = df[df["student_id"] == student_id].sort_values("week")
    
    baseline = student[student["week"] <= 4]
    later = student[student["week"] > 4]
    
    baseline_login = baseline["lms_logins"].mean()
    baseline_att = baseline["attendance"].mean()
    baseline_delay = baseline["submission_delay"].mean()
    baseline_sent = baseline["sentiment_score"].mean()
    
    final_week = student[student["week"] == 16]
    
    login_drop = (final_week["lms_logins"].values[0] - baseline_login) / baseline_login
    att_drop = (final_week["attendance"].values[0] - baseline_att) / baseline_att
    delay_increase = (final_week["submission_delay"].values[0] - baseline_delay) / (baseline_delay + 1)
    sentiment_change = final_week["sentiment_score"].values[0] - baseline_sent
    
    # Slopes
    weeks = student["week"]
    
    login_slope = linregress(weeks, student["lms_logins"]).slope
    att_slope = linregress(weeks, student["attendance"]).slope
    delay_slope = linregress(weeks, student["submission_delay"]).slope
    sent_slope = linregress(weeks, student["sentiment_score"]).slope
    
    # Volatility
    login_std = student["lms_logins"].std()
    sent_std = student["sentiment_score"].std()
    
    burnout_label = student["burnout_label"].iloc[-1]
    dropout_label = student["dropout_label"].iloc[-1]
    
    feature_rows.append([
        student_id,
        login_drop,
        att_drop,
        delay_increase,
        sentiment_change,
        login_slope,
        att_slope,
        delay_slope,
        sent_slope,
        login_std,
        sent_std,
        burnout_label,
        dropout_label
    ])

feature_cols = [
    "student_id",
    "login_drop_pct",
    "attendance_drop_pct",
    "delay_increase_pct",
    "sentiment_change",
    "login_slope",
    "attendance_slope",
    "delay_slope",
    "sentiment_slope",
    "login_volatility",
    "sentiment_volatility",
    "burnout_label",
    "dropout_label"
]

features_df = pd.DataFrame(feature_rows, columns=feature_cols)

features_df.to_csv("engineered_behaviour_features.csv", index=False)

print("Feature Engineering Completed!")
print(features_df.head())