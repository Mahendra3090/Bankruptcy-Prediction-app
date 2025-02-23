import streamlit as st
import pickle
import numpy as np
import os

# Load trained model
model_path = "random_forest_model.pkl"  # Update with your model path/name 
if not os.path.exists(model_path):
    st.error("Model file not found. Please check the file path.")
    st.stop()

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.write(f"Model loaded successfully: {type(model)}")  # Debugging output
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

 
