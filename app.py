
import streamlit as st
import pandas as pd
import joblib
import pickle

# Load model
# The model is directly loaded from the joblib file
loaded_model = joblib.load('churn_pipeline_v1.joblib')

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------- UI HEADER ----------------------
st.title("💜 Customer Churn Prediction Dashboard")
st.markdown("### Your mini-project just got a glow-up ✨")

# ---------------------- Sidebar Charts Example ----------------------
st.sidebar.header("📊 Insights Panel")

# Example: Show sample churn rates (replace with your EDA charts later)
chart_data = pd.DataFrame({
    'Churn': ['Yes', 'No'],
    'Count': [120, 380]
})

st.sidebar.bar_chart(chart_data.set_index('Churn'))

# ---------------------- INPUT SECTION ----------------------
st.subheader("🔍 Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict Churn"):
    # Convert inputs for model
    input_data=  {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    # Create DataFrame from a list containing the input dictionary for a single row
    input_df = pd.DataFrame([input_data])

    # Encode categorical features using the saved encoders
    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    # make a prediction
    prediction = loaded_model.predict(input_df)
    pred_prob = loaded_model.predict_proba(input_df)


    st.write("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")
    st.write(f"Prediction Probability: {pred_prob[0][1]:.2f}")