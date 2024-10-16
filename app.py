# Import necessary libraries
import gradio as gr
import tensorflow as tf
import numpy as np
import logging
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from keras.models import load_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the saved model
model = load_model("melanoma_files/model_final.keras")

# Define the labels
LABELS = [
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

# Load calibration data
x_calib = tf.convert_to_tensor(np.load('melanoma_files/tensor_x.npy'))
y_calib = tf.convert_to_tensor(np.load('melanoma_files/tensor_y.npy'))

def classify_image(image):
    """Classify the uploaded image using the pre-trained model."""
    try:
        img_resized = tf.image.resize(image, (180, 180))
        img_preprocessed = np.expand_dims(img_resized, axis=0)

        # Predict the class probabilities
        predictions = model.predict(img_preprocessed).flatten()
        logging.info(f"Prediction made: {predictions}")

        # Return a dictionary with class probabilities
        return {LABELS[i]: float(predictions[i]) for i in range(len(LABELS))}
    except Exception as e:
        logging.error(f"Error during image classification: {e}")
        return {label: 0.0 for label in LABELS}

def compute_nonconformity(model, x_calib, y_calib):
    """Compute nonconformity scores for calibration data."""
    try:
        probs = model.predict(x_calib)
        class_probs = np.array([probs[i, y_calib[i]] for i in range(len(y_calib))])
        logging.info("Nonconformity scores computed successfully.")
        return class_probs
    except Exception as e:
        logging.error(f"Error computing nonconformity scores: {e}")
        return np.array([])

# Calculate nonconformity scores
calib_scores = compute_nonconformity(model, x_calib, y_calib)

def conformal_prediction(model, x_test, calib_scores, conf_level=90):
    """Generate conformal prediction sets based on the given confidence level."""
    try:
        quantile = np.quantile(calib_scores, 1 - (conf_level / 100))
        probs_test = model.predict(x_test)
        prediction_sets = [np.where(probs_test[i] >= quantile)[0] for i in range(len(x_test))]
        return prediction_sets
    except Exception as e:
        logging.error(f"Error during conformal prediction: {e}")
        return []

def lime_explanation(image):
    """Generate a LIME explanation for the uploaded image."""
    try:
        img_resized = tf.image.resize(image, (180, 180))
        img_preprocessed = np.expand_dims(img_resized, axis=0)
        test_image = tf.reshape(img_preprocessed, (180, 180, 3)).numpy().astype('float32') / 255.0
        
        # Get predicted class
        predicted_class = model.predict(np.expand_dims(test_image, axis=0)).argmax()

        # Explain the prediction
        explanation = lime_image.LimeImageExplainer().explain_instance(
            test_image, model.predict, top_labels=5, hide_color=255, num_samples=100
        )

        # Extract image and mask
        temp, mask = explanation.get_image_and_mask(predicted_class, positive_only=True, num_features=10, hide_rest=True)
        return mark_boundaries(temp / 2 + 0.5, mask)
    except Exception as e:
        logging.error(f"Error during LIME explanation: {e}")
        return np.zeros((180, 180, 3))  # Return a blank image on error

def process_image_and_confidence(image, conf_level):
    """Process image and provide class predictions based on confidence level."""
    try:
        # Validate and convert the confidence level input
        conf_level = int(conf_level)
        if not 1 <= conf_level <= 100:
            return "Invalid input. Please enter a number between 1 and 100."

        img_resized = tf.image.resize(image, (180, 180))
        img_preprocessed = np.expand_dims(img_resized, axis=0)

        # Get conformal prediction sets
        confidence_set = conformal_prediction(model, img_preprocessed, calib_scores, conf_level)
        classes = [LABELS[i] for i in confidence_set[0]]

        return classes if classes else "None"
    except ValueError:
        return "Invalid input. Please enter a valid number between 1 and 100."
    except Exception as e:
        logging.error(f"Error during confidence-based prediction: {e}")
        return "An error occurred during processing."

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("## Image Classification with Conformal Prediction and LIME")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload or Select Image", type="pil")
            example_images = [
                "melanoma_files/sample_images/ISIC_0000002.jpg",
                "melanoma_files/sample_images/ISIC_0000004.jpg",
                "melanoma_files/sample_images/ISIC_0000013.jpg"
            ]
            gr.Examples(examples=example_images, inputs=image_input)

        with gr.Column():
            classify_button = gr.Button("Classify")
            label_output = gr.Label(num_top_classes=3, label="Top Classes")
            classify_button.click(classify_image, inputs=image_input, outputs=label_output)

    gr.Markdown("## Conformal Prediction")

    with gr.Row():
        text_input = gr.Textbox(label="Enter a confidence number between 1 and 100", scale=1)
        generate_button = gr.Button("Get Classes", scale=0.5)
        text_output = gr.Textbox(label="Generated Classes", scale=3)

        generate_button.click(process_image_and_confidence, inputs=[image_input, text_input], outputs=text_output)

    gr.Markdown("## LIME Explanation")

    with gr.Row():
        lime_button = gr.Button("Get LIME Output")
        image_output = gr.Image(label="Influential Parts of the Image", type="pil")

        lime_button.click(lime_explanation, inputs=image_input, outputs=image_output)

# Launch the interface
demo.launch()
