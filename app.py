# Imports
import joblib
import gradio as gr
import tensorflow as tf
import numpy as np
import os
import logging


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model and check if it was loaded successfully
model_filename = os.getenv('MODEL_FILENAME', 'model_filename.pkl')

try:
    model = joblib.load(model_filename)
    logging.info(f"Model loaded successfully from {model_filename}.")
except Exception as e:
    logging.error(f"Failed to load model from {model_filename}: {e}")
    raise

# Define the labels
labels = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

def classify_image(img_path):
    try:
        # Resize and preprocess the image
        img_resized = tf.image.resize(img_path, (180, 180))
        img = np.expand_dims(img_resized, axis=0)

        # Make a prediction
        prediction = model.predict(img).flatten()
        logging.info(f"Prediction made: {prediction}")

        # Return the top labels with their probabilities
        return {labels[i]: float(prediction[i]) for i in range(len(labels))}
    except Exception as e:
        logging.error(f"Error during image classification: {e}")
        return {label: 0.0 for label in labels}  # Return zero probabilities on error

# Gradio interface setup
try:
    gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        examples=["banana.jpg", "car.jpg"]
    ).launch()
except Exception as e:
    logging.error(f"Failed to launch Gradio interface: {e}")
    raise



