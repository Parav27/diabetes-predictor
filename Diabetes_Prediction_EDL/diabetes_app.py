import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
import os
import pickle

# Get absolute path to the .pkl file
model_path = os.path.join(os.path.dirname(__file__), "code", "diabetes_model.pkl")

# Load model
model = pickle.load(open(model_path, "rb"))



st.title("🩺 Diabetes Prediction App")
st.write("Enter the values below to check if a person is likely to have diabetes:")

# Input fields
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 120)

# Predict button
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High chance of Diabetes, CHECKUP RECOMMENDED!!")
    else:
        st.success("✅ Low chance of Diabetes. ALL GOOD!!")
