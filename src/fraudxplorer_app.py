
import streamlit as st
import pandas as pd
import joblib

# Page title
st.set_page_config(page_title="FraudXplorer", layout="centered")
st.title("🔍 FraudXplorer - Fraud Detection in Transactions (XGBoost)")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Dataset")
    st.write(df.head())

    # Drop target column if present
    if 'isFraud' in df.columns:
        df = df.drop('isFraud', axis=1)

    # Encode 'type' column if present
    if 'type' in df.columns:
        df['type'] = df['type'].astype('category').cat.codes

    # Normalize 'amount' column if present
    if 'amount' in df.columns:
        df['amount'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()

    # Drop unnecessary columns if present
    drop_cols = ['nameOrig', 'nameDest']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    try:
        # Predict probabilities and apply threshold
        proba = model.predict_proba(df)[:, 1]
        predictions = (proba > 0.3).astype(int)  # You can fine-tune this threshold

        df['Prediction'] = predictions
        df['Prediction'] = df['Prediction'].map({0: 'Not Fraud', 1: 'Fraud'})

        # Filter and show only fraud cases
        fraud_df = df[df['Prediction'] == 'Fraud']

        st.subheader("🚨 Detected Fraud Transactions")
        if not fraud_df.empty:
            st.dataframe(fraud_df)
        else:
            st.info(" No fraud transactions detected.")

        #  Fraud count summary
        fraud_count = len(fraud_df)
        st.success(f" Total Fraud Transactions Detected: {fraud_count}")

    except Exception as e:
        st.error(f" Error during prediction: {e}")
