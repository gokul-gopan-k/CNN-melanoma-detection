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

################################################
#Image classifier
################################################

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


#Conformal prediction

x= np.load('tensor_x.npy')
y = np.load('tensor_y.npy')

# Convert to a TensorFlow tensor
x_calib = tf.convert_to_tensor(x)
# Convert to a TensorFlow tensor
y_calib = tf.convert_to_tensor(y)

def compute_nonconformity(model, x_calib, y_calib):
    """
    Compute the nonconformity score for each example in the calibration set.
    The nonconformity score is defined as 1 - predicted probability for the true class.
    
    model: trained CNN model
    x_val: validation or calibration set features
    y_val: true labels for the validation or calibration set
    """
    # Get probabilities from the model
    probs = model.predict(x_calib)
 
    # Get the true class probabilities
    calib_scores = 1 - np.array([probs[i, y_calib[i]] for i in range(len(y_calib))])
    
    return calib_scores

# Calculate nonconformity scores
nonconformity_scores = compute_nonconformity(model, x_calib, y_calib)

def conformal_prediction(model, x_test, calib_scores, alpha=0.1):
    """
    Perform conformal prediction and return the set of labels that belong to the 
    confidence set, i.e., labels with p-values greater than alpha.
    
    model: trained CNN model
    x_test: test image for prediction
    calib_scores: nonconformity scores from the calibration set
    alpha: significance level (e.g., 0.1)
    """
    # Get softmax probabilities for the test image
    probs_test = model.predict(x_test)
    #print(probs_test)
    # Calculate p-values for each class
    p_values = []
    for class_idx in range(probs_test.shape[1]):
        nonconformity_test = 1 - probs_test[:, class_idx]
        p_value = np.mean(nonconformity_test >= calib_scores)
        p_values.append(p_value)

    # Return classes with p-values greater than alpha
    return [i for i, p in enumerate(p_values) if p <= (1-float(alpha))]


################################################
# LIME
################################################
from lime import lime_image
import matplotlib.pyplot as plt

# Create a LIME image explainer
explainer = lime_image.LimeImageExplainer()
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def lime(image):
    # Load the image
    #image = load_img(image_path, target_size=target_size)
    # Convert the image to a NumPy array
    #image_array = img_to_array(image)
    # Normalize the image array (optional, depending on your model's requirements)
    # image_array = image_array.astype('float32') / 255.0  # Scale to [0, 1]
    # Convert to tensor
    #image_tensor = tf.convert_to_tensor(image_array)
    # Add a batch dimension (required for models)
    #image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    img_resized = tf.image.resize(image, (180, 180))
    img = np.expand_dims(img_resized, axis=0)


    test_image= tf.reshape(img, (180,180,3))
    test_image = test_image.numpy()
    test_image = test_image.astype('float32') / 255.0 
    predicted_class = model.predict(np.expand_dims(test_image, axis=0)).argmax()
    # Explain the prediction
    explanation = explainer.explain_instance(test_image, 
                                         model.predict, 
                                         top_labels=5, 
                                         hide_color=255, 
                                         num_samples=100)

# Get the image and mask
    temp, mask = explanation.get_image_and_mask(predicted_class, 
                                             positive_only=True, 
                                             num_features=10, 
                                             hide_rest=True)
    return temp

################################################
# Gradio interface setup
################################################

import logging


def process_image_and_float(image, value):
    # Replace this with your logic to process both image and float value
    # Resize and preprocess the image
    # image = load_img(img_path)
    img_resized = tf.image.resize(image, (180, 180))
    img = np.expand_dims(img_resized, axis=0)
    print(img.shape)
    confidence_set = conformal_prediction(model, img, nonconformity_scores, alpha=value)
    print(confidence_set)
    classes =[]
    for i in confidence_set:
        classes.append(labels[i])
    print(classes)
    return classes or "None"



import gradio as gr


def get_tensor_from_image_path(image_path, target_size=(180,180)):
    # Load the image
    image = load_img(image_path, target_size=target_size)
    # Convert the image to a NumPy array
    image_array = img_to_array(image)
    # Normalize the image array (optional, depending on your model's requirements)
    image_array = image_array.astype('float32') / 255.0  # Scale to [0, 1]
    # Convert to tensor
    image_tensor = tf.convert_to_tensor(image_array)
    # Add a batch dimension (required for models)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor

example_images = "/Users/gokulgopank/Documents/deploy/Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000004.jpg"
# Create the interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("## Image Classification with conformal prediction and LIME")
    # Shared image input for all rows
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload or Select Image", type="pil")
            example_images = ["/Users/gokulgopank/Documents/deploy/Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000004.jpg","/Users/gokulgopank/Documents/deploy/Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/ISIC_0000002.jpg"]

        # Add examples to the image input
            gr.Examples(examples=example_images, inputs=image_input)

    # Row 1: Image input to label
        with gr.Column():
            classify_button = gr.Button("Classify")
            label_output = gr.Label(num_top_classes=3, label="Top Classes")
            classify_button.click(classify_image, inputs=image_input, outputs=label_output)
    
    with gr.Row():
        gr.Markdown("## Conformal prediction")
    # Row 2: Image input and text input to text output
    with gr.Row():
        
        text_input = gr.Textbox(label="Enter Text")
        generate_button = gr.Button("Generate Text")
        text_output = gr.Textbox(label="Generated Text")
        generate_button.click(process_image_and_float, inputs=[image_input, text_input], outputs=text_output)
    
    with gr.Row():
        gr.Markdown("## LIME")

    # Row 3: Image input to image output
    with gr.Row():
        transform_button = gr.Button("Transform Image",elem_classes="feedback",scale=0.5)
        image_output = gr.Image(label="Transformed Image", type="pil",scale=1.5)
        transform_button.click(lime, inputs=image_input, outputs=image_output)

# Define the CSS for the button size



# Launch the interface
demo.launch()
