import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/engineered_behaviour_features.csv")

le = LabelEncoder()
df["burnout_encoded"] = le.fit_transform(df["burnout_label"])

X = df.drop(["student_id", "burnout_label", "dropout_label", "burnout_encoded"], axis=1)
y = df["burnout_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Load Model
# -----------------------------
with open("models/burnout_model.pkl", "rb") as f:
    burnout_model = pickle.load(f)

y_pred = burnout_model.predict(X_test)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Burnout Classification Confusion Matrix")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png")

# Save classification report
report = classification_report(y_test, y_pred)
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

print("Evaluation Results Saved")