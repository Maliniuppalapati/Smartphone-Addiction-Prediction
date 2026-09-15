import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Smartphone Addiction Predictor",
    page_icon="📱",
    layout="centered"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model file: {e}")
        return None

model = load_model()

# Header Section
st.title("📱 Smartphone Addiction Level Predictor")
st.markdown(
    "Predict student smartphone addiction risk levels (**Low**, **Medium**, **High**) "
    "using behavioral habits and psychological metrics."
)

if model is None:
    st.warning("⚠️ **Model file (`best_model.pkl`) not found.**")
    st.info("Please run `python train.py` in your terminal to train and save the model.")
    if st.button("🚀 Train Model Now directly from App"):
        with st.spinner("Training model with SMOTE & Stratified K-Fold CV..."):
            try:
                from train import train_and_save_model
                train_and_save_model()
                st.cache_resource.clear()
                st.success("Model trained successfully! Reloading page...")
                st.rerun()
            except Exception as ex:
                st.error(f"Error training model: {ex}")
    st.stop()

st.subheader("📊 Student Habits & Parameters")

col1, col2 = st.columns(2)

with col1:
    daily_usage = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=4.5, step=0.5)
    sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=15.0, value=7.0, step=0.5)
    time_social = st.number_input("Time on Social Media (Hours)", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
    time_gaming = st.number_input("Time on Gaming (Hours)", min_value=0.0, max_value=15.0, value=1.0, step=0.5)
    age = st.number_input("Age", min_value=5, max_value=25, value=17)

with col2:
    apps_daily = st.number_input("Apps Used Daily", min_value=1, max_value=50, value=12)
    phone_checks = st.number_input("Phone Checks Per Day", min_value=0, max_value=300, value=45)
    anxiety = st.number_input("Anxiety Level (0-10)", min_value=0, max_value=10, value=4)
    academic = st.number_input("Academic Performance (0-100)", min_value=0, max_value=100, value=75)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# Compute engineered features
usage_to_sleep = daily_usage / (sleep + 0.1)
social_plus_gaming = time_social + time_gaming
screen_per_check = daily_usage / (phone_checks + 1.0)

# Input Summary Cards
with st.expander("🔍 View Computed Engineering Features"):
    st.write(f"- **Usage to Sleep Ratio:** `{usage_to_sleep:.2f}`")
    st.write(f"- **Total Social + Gaming Hours:** `{social_plus_gaming:.2f} hrs`")
    st.write(f"- **Screen Time per Phone Check:** `{screen_per_check * 60:.1f} mins/check`")

st.markdown("---")

if st.button("🔮 Predict Addiction Level", type="primary", use_container_width=True):
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
        
        st.subheader("🎯 Prediction Result")
        
        if prediction == "Low":
            st.success(f"Predicted Risk Level: **{prediction} Risk** 🎉")
            st.caption("Healthy usage habits detected. Maintain your current balance!")
        elif prediction == "Medium":
            st.warning(f"Predicted Risk Level: **{prediction} Risk** ⚠️")
            st.caption("Moderate usage. Consider setting screen time limits for social apps & gaming.")
        else:
            st.error(f"Predicted Risk Level: **{prediction} Risk** 🚨")
            st.caption("High risk detected! We recommend digital detox strategies and sleep schedule optimization.")

        # Confidence Chart
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_
            
            # Map probabilities to dictionary
            prob_dict = dict(zip(classes, probabilities))
            
            # Ensure consistent order: Low, Medium, High
            ordered_tiers = ["Low", "Medium", "High"]
            chart_data = pd.DataFrame({
                "Risk Tier": ordered_tiers,
                "Confidence (%)": [float(np.round(prob_dict.get(tier, 0.0) * 100, 2)) for tier in ordered_tiers]
            }).set_index("Risk Tier")

            st.markdown("### 📊 Prediction Confidence Breakdown")
            
            # Bar Chart
            st.bar_chart(chart_data)

            # Detailed metrics view
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Low Risk Confidence", f"{prob_dict.get('Low', 0.0)*100:.1f}%")
            with col_b:
                st.metric("Medium Risk Confidence", f"{prob_dict.get('Medium', 0.0)*100:.1f}%")
            with col_c:
                st.metric("High Risk Confidence", f"{prob_dict.get('High', 0.0)*100:.1f}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
