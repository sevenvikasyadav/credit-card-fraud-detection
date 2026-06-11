import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgboost_fraud_model.pkl")
amount_scaler = joblib.load("amount_scaler.pkl")
demo_df = pd.read_csv("demo_transactions.csv")

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")

st.markdown("""
Detect fraudulent credit card transactions using an XGBoost model trained on the Kaggle Credit Card Fraud dataset.

### Features
- SMOTE for handling class imbalance
- XGBoost classifier
- SHAP explainability
- Interactive hybrid demo
""")

st.info(
    """
For demonstration purposes, anonymized transaction patterns (V1–V28)
are borrowed from representative transactions in the unseen test set,
while the transaction amount can be customized by the user.

In a real banking system, all features would be generated internally
from the complete transaction pipeline.
"""
)

st.header("Enter Transaction Details")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0,
    step=100.0
)

labels = [
    f"Profile {i + 1}"
    for i in range(len(demo_df))
]

selected = st.selectbox(
    "Select Transaction Profile",
    range(len(labels)),
    format_func=lambda x: labels[x]
)

transaction = demo_df.iloc[selected]

if st.checkbox("Show Profile Details (V1–V28)"):
    st.dataframe(
        transaction.drop("Actual_Class").to_frame().T,
        use_container_width=True
    )

if st.button("Predict", use_container_width=True):

    scaled_amount = amount_scaler.transform([[amount]])[0][0]

    input_row = transaction.drop("Actual_Class").copy()

    input_row["scaled_amount"] = scaled_amount

    input_df = pd.DataFrame([input_row])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Fraud Probability",
        f"{probability:.2%}"
    )

    if prediction == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.info(
        f"Selected Profile Actual Label: {actual_label}"
    )

    if prediction == transaction["Actual_Class"]:
        st.success(
            "🎯 The prediction matches the selected profile label."
        )
    else:
        st.warning(
            """
            The prediction differs from the selected profile label.

            This can happen because the transaction amount was modified,
            creating a new synthetic scenario.
            """
        )

st.markdown("---")

st.caption(
    "Built by Vikas Yadav • B.Tech Computer Engineering • Jamia Millia Islamia"
)
