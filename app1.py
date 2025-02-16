import streamlit as st
import xgboost as xgb
import tensorflow as tf
import numpy as np
import os

# Set correct model paths
maternal_model_path = "app/model.bin"
skin_cancer_model_path = "app/skin_cancer_model.h5"

# Load Maternal Health Model
try:
    maternal_model = xgb.Booster()
    maternal_model.load_model(maternal_model_path)
except Exception as e:
    st.error(f"Error loading maternal health model: {e}")

# Load Skin Cancer Model
try:
    skin_cancer_model = tf.keras.models.load_model(skin_cancer_model_path)
except Exception as e:
    st.error(f"Error loading skin cancer model: {e}")

# Streamlit UI
st.title("AI Diagnostic Platform")

# Maternal Health Risk Prediction
st.header("Maternal Health Risk Prediction")
age = st.number_input("Enter Age", min_value=10, max_value=100, step=1)
weight = st.number_input("Enter Weight (kg)", min_value=30.0, max_value=150.0, step=0.1)

if st.button("Predict Maternal Risk"):
    if maternal_model:
        input_data = np.array([[age, weight]])
        dmatrix = xgb.DMatrix(input_data)
        prediction = maternal_model.predict(dmatrix)[0]
        st.success(f"Predicted Risk Level: {prediction:.2f}")
    else:
        st.error("Maternal health model not loaded.")

# Skin Cancer Detection
st.header("Skin Cancer Detection")
uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, use_container_width=True)
    if st.button("Predict Skin Cancer"):
        if skin_cancer_model:
            img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            prediction = skin_cancer_model.predict(img_array)
            st.success(f"Predicted Skin Cancer Probability: {prediction[0][0]:.2f}")
        else:
            st.error("Skin cancer model not loaded.")


