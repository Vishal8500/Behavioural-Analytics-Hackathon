import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

NUM_STUDENTS = 2000
NUM_WEEKS = 16

profiles = {
    "consistent": 0.35,
    "gradual_burnout": 0.30,
    "sudden_disengagement": 0.20,
    "chronic_low": 0.15
}

def assign_profile():
    return np.random.choice(
        list(profiles.keys()),
        p=list(profiles.values())
    )

def generate_student_baseline(profile):
    if profile == "consistent":
        return {
            "logins": np.random.randint(18, 25),
            "attendance": np.random.uniform(85, 100),
            "delay": np.random.uniform(0, 3),
            "sentiment": np.random.uniform(0.2, 0.8)
        }
    elif profile == "gradual_burnout":
        return {
            "logins": np.random.randint(15, 22),
            "attendance": np.random.uniform(80, 95),
            "delay": np.random.uniform(1, 5),
            "sentiment": np.random.uniform(0.1, 0.5)
        }
    elif profile == "sudden_disengagement":
        return {
            "logins": np.random.randint(18, 24),
            "attendance": np.random.uniform(85, 100),
            "delay": np.random.uniform(0, 4),
            "sentiment": np.random.uniform(0.2, 0.6)
        }
    else:  # chronic_low
        return {
            "logins": np.random.randint(5, 12),
            "attendance": np.random.uniform(50, 75),
            "delay": np.random.uniform(5, 15),
            "sentiment": np.random.uniform(-0.5, 0.2)
        }

data = []

for student_id in range(1, NUM_STUDENTS + 1):

    profile = assign_profile()
    baseline = generate_student_baseline(profile)

    dropout_flag = 0

    crash_week = np.random.randint(7, 11) if profile == "sudden_disengagement" else None

    for week in range(1, NUM_WEEKS + 1):

        logins = baseline["logins"]
        attendance = baseline["attendance"]
        delay = baseline["delay"]
        sentiment = baseline["sentiment"]

        # Weeks 1–4 = Stable Baseline
        if week <= 4:
            pass

        else:
            if profile == "gradual_burnout":
                decay_factor = (week - 4) / 12
                logins -= decay_factor * np.random.uniform(4, 10)
                attendance -= decay_factor * np.random.uniform(5, 15)
                delay += decay_factor * np.random.uniform(3, 8)
                sentiment -= decay_factor * np.random.uniform(0.3, 0.7)

            elif profile == "sudden_disengagement":
                if week >= crash_week:
                    logins *= np.random.uniform(0.2, 0.4)
                    attendance *= np.random.uniform(0.3, 0.6)
                    delay += np.random.uniform(10, 20)
                    sentiment -= np.random.uniform(0.5, 1.0)

            elif profile == "chronic_low":
                logins += np.random.normal(0, 2)
                attendance += np.random.normal(0, 5)
                delay += np.random.normal(0, 3)
                sentiment += np.random.normal(0, 0.2)

        # Add realistic noise
        logins = max(0, logins + np.random.normal(0, 2))
        attendance = np.clip(attendance + np.random.normal(0, 3), 0, 100)
        delay = max(0, delay + np.random.normal(0, 2))
        sentiment = np.clip(sentiment + np.random.normal(0, 0.1), -1, 1)

        activity_variance = np.random.uniform(1, 5)

        # Burnout labeling
        if logins < 5 or attendance < 50 or sentiment < -0.5:
            burnout_label = "High"
        elif logins < 12 or attendance < 70:
            burnout_label = "Medium"
        else:
            burnout_label = "Low"

        # Dropout probability logic
        dropout_prob = (
            (1 - attendance / 100) * 0.4 +
            (delay / 20) * 0.3 +
            (max(0, -sentiment)) * 0.3
        )

        if dropout_prob > 0.6:
            dropout_flag = 1

        data.append([
            student_id,
            profile,
            week,
            round(logins, 2),
            round(delay, 2),
            round(attendance, 2),
            round(sentiment, 3),
            round(activity_variance, 2),
            burnout_label,
            dropout_flag
        ])

columns = [
    "student_id",
    "profile_type",
    "week",
    "lms_logins",
    "submission_delay",
    "attendance",
    "sentiment_score",
    "activity_variance",
    "burnout_label",
    "dropout_label"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("synthetic_student_burnout_data_2000.csv", index=False)

print("Dataset Generated Successfully!")
print(df.head())
print("Total Records:", len(df))