import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
MODEL_PATH = 'models/waste_mobilenet_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

st.title("♻️ Waste Classification App")
st.write("Upload an image to classify whether it’s **Organic** or **Recyclable**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_container_width=True)
    st.write("")

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    class_names = ['Organic', 'Recyclable']

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"### 🧩 Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
