import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgboost_fraud_model.pkl")
amount_scaler = joblib.load("amount_scaler.pkl")
time_scaler = joblib.load("time_scaler.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
This application predicts whether a credit card transaction is **Fraudulent**
or **Legitimate** using an **XGBoost model** trained on the Kaggle Credit Card Fraud dataset.

**Models Evaluated:**
- Logistic Regression
- Random Forest
- XGBoost ✅
"""
)

st.header("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Amount",
        min_value=0.0,
        value=100.0
    )

with col2:
    time = st.number_input(
        "Time",
        min_value=0.0,
        value=0.0
    )

st.subheader("PCA Features (V1–V28)")

features = {}

cols = st.columns(4)

for i in range(1, 29):
    with cols[(i - 1) % 4]:
        features[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f"
        )

if st.button("Predict"):
    scaled_amount = amount_scaler.transform([[amount]])[0][0]
    scaled_time = time_scaler.transform([[time]])[0][0]

    input_data = []

    for i in range(1, 29):
        input_data.append(features[f"V{i}"])

    input_data.extend([
        scaled_amount,
        scaled_time
    ])

    input_df = pd.DataFrame(
        [input_data],
        columns=[
            "V1", "V2", "V3", "V4", "V5", "V6", "V7",
            "V8", "V9", "V10", "V11", "V12", "V13", "V14",
            "V15", "V16", "V17", "V18", "V19", "V20", "V21",
            "V22", "V23", "V24", "V25", "V26", "V27", "V28",
            "scaled_amount",
            "scaled_time"
        ]
    )

    prediction = model.predict(input_df)[0]
    fraud_probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Fraud Probability",
        f"{fraud_probability:.2%}"
    )

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")
