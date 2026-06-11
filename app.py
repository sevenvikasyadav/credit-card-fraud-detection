import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgboost_fraud_model.pkl")
demo_df = pd.read_csv("demo_transactions.csv")

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
Detect fraudulent credit card transactions using an XGBoost model trained on the Kaggle Credit Card Fraud dataset.

### Features
- SMOTE for handling class imbalance
- XGBoost classifier
- SHAP explainability
- Interactive demo using real test transactions
"""
)

st.header("Demo Mode")

st.write(
    "Select a sample transaction from the test set and let the model predict whether it is fraudulent."
)

labels = [f"Transaction {i + 1}" for i in range(len(demo_df))]

selected = st.selectbox(
    "Select a Sample Transaction",
    range(len(labels)),
    format_func=lambda x: labels[x]
)

transaction = demo_df.iloc[selected]

st.subheader("Transaction Preview")

preview = transaction.drop("Actual_Class")
st.dataframe(
    preview.to_frame().T,
    use_container_width=True
)

if st.button("Predict", use_container_width=True):

    input_df = pd.DataFrame(
        [transaction.drop("Actual_Class")]
    )

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

    actual = int(transaction["Actual_Class"])

    st.info(
        f"Actual Label: {'Fraud' if actual == 1 else 'Legitimate'}"
    )

    if prediction == actual:
        st.success("🎯 The model predicted correctly!")
    else:
        st.warning("❌ The model prediction was incorrect.")
