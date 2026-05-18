import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details to predict survival.")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0, 500, 50)

# Encode sex
sex_encoded = 1 if sex == "male" else 0

# Prediction
if st.button("Predict"):
    input_data = np.array([[pclass, sex_encoded, age, fare]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Survived!")
    else:
        st.error("💀 Did not survive")