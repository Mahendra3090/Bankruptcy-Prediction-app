import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

 


#load the saved models &preprocessing objects
def load_pickle(filename):
    path = os.path.join("C:/Users/Asus/Desktop/ML_Project/model/",filename)
    with open(path, "rb") as file:
       return pickle.load(file)

rf_model = load_pickle("rf_model.pkl")
xgb_model = load_pickle("xgb_model.pkl")
scaler = load_pickle("scaler.pkl")
#smote = load_pickle("smote.pkl")
#encoder = load_pickle("encoder.pkl")

#App Title
st.title("💰 Bankruptcy Prediction App")
st.markdown("Enter company risk factors to predict bankruptcy risk.")

# Sidebar for user inputs
st.sidebar.header("Enter Risk Factor Values")

def user_inputs():
    industrial_risk = st.sidebar.selectbox("Industrial Risk", [0, 1])
    management_risk = st.sidebar.selectbox("Management Risk", [0, 1])
    financial_flexibility = st.sidebar.selectbox("Financial Flexibility", [0, 1])
    credibility = st.sidebar.selectbox("Credibility", [0, 1])
    competitiveness = st.sidebar.selectbox("Competitiveness", [0, 1])
    operating_risk = st.sidebar.selectbox("Operating Risk", [0, 1])

    return pd.DataFrame([[industrial_risk, management_risk, financial_flexibility, 
                          credibility, competitiveness, operating_risk]],
                        columns=["industrial_risk", "management_risk", 
                                 "financial_flexibility", "credibility", 
                                 "competitiveness", "operating_risk"])

# Get user input
user_data = user_inputs()

# Display entered data
st.subheader("Entered Data:")
st.write(user_data)

# Model selection
model_choice = st.radio("Select a model for prediction:", ["Random Forest", "XGBoost"])

# Predict on button click
if st.button("Predict Bankruptcy Risk"):
    #Encode input
    #user_data_encoded = encoder.transform(user_data)

    #Scale input data
    user_data_scaled = scaler.transform(user_data)

    #Select the model
    if model_choice == "Random Forest":
        prediction = rf_model.predict(user_data_scaled)[0]
    else:
        prediction = xgb_model.predict(user_data_scaled)[0]

    # Display results
    prediction_text = "❌ At Risk of Bankruptcy" if prediction == 1 else "✅ Not at Risk"
    
    st.subheader("📊 Prediction Result:")
    st.write(f"**{model_choice}:** {prediction_text}")







 
