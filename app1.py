import tensorflow as tf
from sklearn.externals import joblib


# Load the models (change paths to where your models are saved)
model_skin_cancer = tf.keras.models.load_model(r'C:\Users\navee\Downloads\skin.h5')

model_maternal_health = joblib.load('app/model.bin')


import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

# Set title
st.title("AI Diagnostic Platform")

# Sidebar navigation
st.sidebar.header("Select Diagnosis Method")
option = st.sidebar.selectbox("Choose Diagnosis Type", ["Skin Cancer Detection", "Maternal Health Risk"])

# Skin Cancer Detection
if option == "Skin Cancer Detection":
    st.header("Skin Cancer Detection")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Load your pre-trained model here (TensorFlow, etc.)
        model = tf.keras.models.load_model('path/to/skin_cancer_model.h5')
        
        # Preprocess and predict the image
        image = np.array(image.resize((224, 224)))  # Resize for model input
        image = np.expand_dims(image, axis=0) / 255.0  # Normalize
        prediction = model.predict(image)

        # Display prediction results
        st.write(f"Prediction: {prediction}")

# Maternal Health Risk Prediction
elif option == "Maternal Health Risk":
    st.header("Maternal Health Risk Prediction")
    
    # Input fields for maternal health data
    age = st.number_input("Enter Age", min_value=0, max_value=100)
    weight = st.number_input("Enter Weight (kg)", min_value=0)
    
    # Load model for maternal health
    model = joblib.load('path/to/maternal_health_model.pkl')
    
    if st.button("Predict Risk"):
        input_data = np.array([[age, weight]])  # Include other features as needed
        risk = model.predict(input_data)
        
        st.write(f"Maternal Health Risk: {risk[0]}")
