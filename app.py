import streamlit as st
import joblib
import pandas as pd
import os

MODEL_PATH = "model/best_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run train.py first.")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Smartphone Addiction Predictor", page_icon="📱", layout="centered")

st.title("📱 Smartphone Addiction Level Predictor")
st.markdown("Predict student smartphone addiction risk levels (**Low**, **Medium**, **High**) using behavioral indicators.")

st.subheader("Student Habits & Parameters")

col1, col2 = st.columns(2)

with col1:
    daily_usage = st.number_input("Daily Usage Hours", 0.0, 24.0, 2.0, step=0.5)
    sleep = st.number_input("Sleep Hours", 0.0, 15.0, 8.0, step=0.5)
    time_social = st.number_input("Time on Social Media (Hours)", 0.0, 15.0, 1.0, step=0.5)
    time_gaming = st.number_input("Time on Gaming (Hours)", 0.0, 15.0, 0.5, step=0.5)
    age = st.number_input("Age", 5, 25, 16)

with col2:
    apps_daily = st.number_input("Apps Used Daily", 1, 50, 5)
    phone_checks = st.number_input("Phone Checks Per Day", 0, 200, 20)
    anxiety = st.number_input("Anxiety Level (0-10)", 0, 10, 3)
    academic = st.number_input("Academic Performance (0-100)", 0, 100, 80)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# Compute engineered features
usage_to_sleep = daily_usage / (sleep + 0.1)
social_plus_gaming = time_social + time_gaming
screen_per_check = daily_usage / (phone_checks + 1.0)

if st.button("Predict Addiction Level", type="primary", use_container_width=True):
    if model is None:
        st.error("Model is not loaded.")
        st.stop()

    input_data = pd.DataFrame([{
        "Daily_Usage_Hours": daily_usage,
        "Sleep_Hours": sleep,
        "Anxiety_Level": anxiety,
        "Academic_Performance": academic,
        "Age": age,
        "Gender": gender,
        "Apps_Used_Daily": apps_daily,
        "Time_on_Social_Media": time_social,
        "Time_on_Gaming": time_gaming,
        "Phone_Checks_Per_Day": phone_checks,
        "Usage_to_Sleep_Ratio": usage_to_sleep,
        "Social_Plus_Gaming": social_plus_gaming,
        "Screen_Per_Check": screen_per_check
    }])

    try:
        prediction = model.predict(input_data)[0]
        
        if prediction == "Low":
            st.success(f"Predicted Risk Level: **{prediction} Risk** 🎉")
        elif prediction == "Medium":
            st.warning(f"Predicted Risk Level: **{prediction} Risk** ⚠️")
        else:
            st.error(f"Predicted Risk Level: **{prediction} Risk** 🚨")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            prob_df = pd.DataFrame({
                "Risk Tier": model.classes_,
                "Confidence": probabilities
            })
            st.subheader("Prediction Confidence Breakdown")
            st.bar_chart(prob_df.set_index("Risk Tier"))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
