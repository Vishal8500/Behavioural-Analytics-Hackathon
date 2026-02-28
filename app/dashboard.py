import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os

st.set_page_config(layout="wide")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
# -----------------------------
# FIXED PATH HANDLING
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

final_df = pd.read_csv(os.path.join(BASE_DIR, "outputs", "final_with_interventions.csv"))
engineered_df = pd.read_csv(os.path.join(BASE_DIR, "data", "engineered_behaviour_features.csv"))
raw_df = pd.read_csv(os.path.join(BASE_DIR, "data", "synthetic_student_burnout_data_2000.csv"))

profile_map = raw_df[["student_id", "profile_type"]].drop_duplicates()
final_df = final_df.merge(profile_map, on="student_id", how="left")

# Risk segmentation
def segment(score):
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

final_df["risk_segment"] = final_df["risk_score"].apply(segment)

# ---------------------------------------------------
# LOAD MODEL FOR SHAP
# ---------------------------------------------------
with open(r"D:\Behavioural_Hackathon\models\burnout_model.pkl", "rb") as f:
    model = pickle.load(f)

X = engineered_df.drop(["student_id", "burnout_label", "dropout_label"], axis=1)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

high_class_index = 2
high_shap = shap_values[:, :, high_class_index]

# ---------------------------------------------------
# PAGE NAVIGATION
# ---------------------------------------------------
page = st.sidebar.radio("Navigation", ["📊 System Overview", "👤 Individual Student Analysis"])

# ===================================================
# PAGE 1: SYSTEM OVERVIEW
# ===================================================
if page == "📊 System Overview":

    st.title("🎓 Behavioural Early Warning System - Overview")

    # -----------------------------
    # KEY METRICS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(final_df))
    col2.metric("High Risk Students", len(final_df[final_df["risk_segment"] == "High"]))
    col3.metric("Average Risk Score", round(final_df["risk_score"].mean(), 2))

    st.markdown("---")

    # -----------------------------
    # RISK SEGMENTATION PIE
    # -----------------------------
    st.subheader("📊 Risk Segmentation")

    pie = px.pie(final_df, names="risk_segment", title="Risk Distribution")
    st.plotly_chart(pie, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # GLOBAL SHAP IMPORTANCE
    # -----------------------------
    st.subheader("🔬 Global Behavioural Risk Drivers")

    fig = plt.figure()
    shap.plots.bar(high_shap, show=False)
    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------
    # TOP 10 HIGH RISK STUDENTS
    # -----------------------------
    st.subheader("🚨 Top 10 High Risk Students")

    top10 = final_df.sort_values("risk_score", ascending=False).head(10)
    st.dataframe(top10[["student_id", "risk_score", "burnout_prediction", "dropout_probability"]])

    # -----------------------------
    # TREND FOR HIGHEST RISK STUDENT
    # -----------------------------
    st.subheader("📈 Behaviour Trend of Highest Risk Student")

    top_student_id = top10.iloc[0]["student_id"]
    trend_df = raw_df[raw_df["student_id"] == top_student_id]

    fig2 = px.line(
        trend_df,
        x="week",
        y=["attendance", "lms_logins", "sentiment_score"],
        title=f"Behaviour Trends - Student {top_student_id}"
    )
    st.plotly_chart(fig2, use_container_width=True)


# ===================================================
# PAGE 2: INDIVIDUAL STUDENT ANALYSIS
# ===================================================
if page == "👤 Individual Student Analysis":

    st.title("👤 Individual Student Behaviour Analysis")

    student_id = st.sidebar.selectbox(
        "Select Student ID",
        final_df["student_id"].unique()
    )

    student_row = final_df[final_df["student_id"] == student_id].iloc[0]
    student_index = engineered_df[engineered_df["student_id"] == student_id].index[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Score", student_row["risk_score"])
    col2.metric("Burnout Level", student_row["burnout_prediction"])
    col3.metric("Dropout Probability", student_row["dropout_probability"])

    st.markdown("### 🩺 Recommended Intervention")
    st.success(student_row["recommended_intervention"])

    st.markdown("### 📘 Behaviour Profile")
    st.info(student_row["profile_type"])

    st.markdown("---")

    # -----------------------------
    # SHAP WATERFALL
    # -----------------------------
    st.subheader("🔬 SHAP Explanation for This Student")

    fig = plt.figure()
    shap.plots.waterfall(high_shap[student_index], show=False)
    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------
    # MOOD TREND
    # -----------------------------
    st.subheader("😊 Sentiment (Mood) Trend Over Weeks")

    student_trend = raw_df[raw_df["student_id"] == student_id]

    mood_fig = px.line(
        student_trend,
        x="week",
        y="sentiment_score",
        title="Mood Trend"
    )
    st.plotly_chart(mood_fig, use_container_width=True)

    # -----------------------------
    # FULL BEHAVIOURAL TRENDS
    # -----------------------------
    st.subheader("📈 Full Behavioural Trends")

    trend_fig = px.line(
        student_trend,
        x="week",
        y=["attendance", "lms_logins", "submission_delay"],
        title="Weekly Behaviour Trends"
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("Behavioural Early Warning System | Explainable AI Enabled")