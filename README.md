# Plant Disease Detection Using Multi-Model Deep Learning Fusion- Team 20

This project introduces a deep learning-based system for identifying plant leaf diseases and recommending sustainable organic treatments. It leverages a fusion of three models—Simple CNN, Deep CNN, and EfficientNetB0—to enhance feature extraction and classification accuracy.

## Features

- Multi-model fusion for improved disease detection  
- Real-time predictions through a web-based interface  
- Organic pesticide suggestions mapped to detected diseases  
- Trained using PlantVillage, Plant Leaves, and PlantDoc datasets  
- Built using TensorFlow and Keras

## How It Works

1. Upload a plant leaf image via the web app  
2. The image is processed by three parallel CNN branches  
3. Extracted features are fused and passed through dense layers  
4. The model predicts the disease and suggests an organic pesticide

## Tech Stack

- Python, TensorFlow, Keras  
- HTML/CSS/JavaScript for frontend  
- JSON for pesticide mapping

## Datasets Used

- **PlantVillage** – clean lab images  
- **Plant Leaves** – varied lighting and conditions  
- **PlantDoc** – real-world, noisy images

## Objective

Deliver an accessible, accurate, and eco-friendly solution to support sustainable agriculture and smart farming practices.
