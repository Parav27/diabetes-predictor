import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Load your trained model (we'll create this file soon)
model = pickle.load(open("code/diabetes_model.pkl", "rb"))


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
