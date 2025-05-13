import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# Load trained model
model = tf.keras.models.load_model("pneumonia_cnn_model.h5")

# Prediction function for Gradio
def predict_xray_gr(image):
    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Preprocess
    img = cv2.resize(image_cv, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]
    predicted_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return f"Predicted: {predicted_class} ({confidence * 100:.2f}%)"

# Gradio UI
iface = gr.Interface(
    fn=predict_xray_gr,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Pneumonia Detection",
    description="Upload a chest X-ray image to detect Pneumonia."
)

iface.launch()
