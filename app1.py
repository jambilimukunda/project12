import tensorflow as tf
import joblib
import streamlit as st
from PIL import Image
import numpy as np

# Load models once
skin_cancer_model_path = r"C:\Users\navee\Downloads\best_model.h5"
model_maternal_health = joblib.load(r"C:\Users\navee\project12\app\model.bin")

try:
    skin_cancer_model = tf.keras.models.load_model(skin_cancer_model_path)
except Exception as e:
    st.error(f"Error loading skin cancer model: {e}")

try:
    model_maternal_health = joblib.load('model.bin')

except Exception as e:
    st.error(f"Error loading maternal health model: {e}")

# Streamlit App
st.title("AI Diagnostic Platform")
st.sidebar.header("Select Diagnosis Method")
option = st.sidebar.selectbox("Choose Diagnosis Type", ["Skin Cancer Detection", "Maternal Health Risk"])

# Skin Cancer Detection
if option == "Skin Cancer Detection":
    st.header("Skin Cancer Detection")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_container_width=True)

        # Preprocess image
        image = image.resize((224, 224))  # Resize for model input
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Predict
        if 'skin_cancer_model' in locals():
            prediction = skin_cancer_model.predict(image)
            st.write(f"Prediction: {prediction}")
        else:
            st.error("Skin cancer model not loaded.")

# Maternal Health Risk Prediction
elif option == "Maternal Health Risk":
    st.header("Maternal Health Risk Prediction")
    
    age = st.number_input("Enter Age", min_value=0, max_value=100)
    weight = st.number_input("Enter Weight (kg)", min_value=0)

    if st.button("Predict Risk"):
        input_data = np.array([[age, weight]])  # Add other required features
        if 'maternal_health_model' in locals():
            risk = maternal_health_model.predict(input_data)
            st.write(f"Maternal Health Risk: {risk[0]}")
        else:
            st.error("Maternal health model not loaded.")

