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
        output = {LABELS[i]: float(predictions[i]) for i in range(len(LABELS))}
        label = max(output, key=output.get)
        desp, fat, can, act = analysis(label)
        # Return a dictionary with class probabilities
        return  label, output, desp, fat, can, act
    except Exception as e:
        logging.error(f"Error during image classification: {e}")
        return {label: 0.0 for label in LABELS}

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
        temp, mask = explanation.get_image_and_mask(predicted_class, positive_only=True, num_features=5, hide_rest=True)
        #return features in image
        return temp
    except Exception as e:
        logging.error(f"Error during LIME explanation: {e}")
        return np.zeros((180, 180, 3))  # Return a blank image on error
    

def analysis(disease_id):
    if disease_id == "actinic keratosis":
        desp = "A precancerous, rough, scaly patch on the skin, often caused by sun damage. Common in fair-skinned individuals."
        fat = "Typically non-fatal, but can progress to squamous cell carcinoma (SCC) if untreated."
        can = "Potentially cancerous or precancerous"
        act = "Early treatment is recommended to prevent progression. Options include cryotherapy (freezing), topical treatments (e.g., 5-fluorouracil), or minor surgical removal."
    elif disease_id == "basal cell carcinoma":
        desp = "A common form of skin cancer that originates in the basal cells. Appears as a pearly or waxy bump, often on sun-exposed areas like the face or neck.."
        fat = "Rarely fatal, but it can be locally invasive and cause significant tissue damage if untreated."
        can = "Potentially cancerous or precancerous"
        act = "Surgical removal is typically required. Other options include Mohs surgery, topical treatments, or radiation."
    elif disease_id == "dermatofibroma":
        desp = "A benign, firm, brownish skin growth that often appears on the legs or arms. Not associated with cancer."
        fat = "Not fatal, non-cancerous."
        can = "Non-cancerous (benign)"
        act = "No treatment necessary unless it becomes irritated or bothersome. In some cases, it may be surgically excised."
    elif disease_id == "melanoma":
        desp = "A dangerous and aggressive form of skin cancer that arises from melanocytes (pigment-producing cells). It can appear as a new mole or change in an existing mole."
        fat = "High potential for fatality if not detected and treated early. It is the deadliest form of skin cancer."
        can = "Potentially cancerous or precancerous"
        act = "Immediate consultation with a dermatologist is critical. Treatment often involves surgical excision, and in some cases, immunotherapy or chemotherapy may be necessary."
    elif disease_id == "nevus":
        desp = "A benign growth of melanocytes, often pigmented, commonly appearing as a mole. Most nevi are harmless, but some may develop into melanoma."
        fat = "Usually benign and not fatal, though some can transform into melanoma over time."
        can = "Non-cancerous (benign)"
        act = "Regular monitoring for changes in size, shape, or color. If any suspicious changes occur, a biopsy may be recommended."
    elif disease_id == "pigmented benign keratosis":
        desp = "A benign, darkly pigmented skin growth, often appearing as a flat or slightly elevated patch. Typically associated with sun exposure and aging."
        fat = "Non-cancerous and not fatal."
        can = "Non-cancerous (benign)"
        act = "No treatment required unless it becomes irritated. Removal may be considered for cosmetic reasons."
    elif disease_id == "seborrheic keratosis":
        desp = "A common, non-cancerous skin tumor that appears as a waxy, raised, and often pigmented lesion. It is associated with aging."
        fat = "Not fatal, non-cancerous."
        can = "Non-cancerous (benign)"
        act = "Usually not necessary to treat unless they become itchy or irritated. Removal can be done for cosmetic reasons or if they interfere with daily activities."
    elif disease_id == "squamous cell carcinoma":
        desp = "A type of skin cancer that arises from squamous cells. It often presents as a scaly, red patch or ulcer that doesn't heal. It is more common in people with prolonged sun exposure."
        fat = "Can be fatal if left untreated, especially if it spreads to other parts of the body (metastasizes)."
        can = "Potentially cancerous or precancerous"
        act = "Early surgical excision is typically recommended. In some cases, radiation therapy or chemotherapy may be needed."
    elif disease_id == "vascular lesion":
        desp = "Abnormal growth of blood vessels, which can appear as red, purple, or blue marks on the skin (e.g., spider veins, cherry angiomas, or hemangiomas). Most are benign."
        fat = "Not fatal, typically benign."
        can = "Non-cancerous (benign)"
        act = "Treatment is usually not necessary unless they cause cosmetic concern or other symptoms like bleeding. Laser therapy can be used for cosmetic purposes or to remove larger lesions."
    else:
        print("Error: Invalid disease")
        return
    return desp, fat, can, act


# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("## Melanoma detection from skin images")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload or Select Image", type="pil")
            example_images = [
                "melanoma_files/sample_images/ISIC_0026171.jpg",
                "melanoma_files/sample_images/ISIC_0026349.jpg",
                "melanoma_files/sample_images/ISIC_0000013.jpg"
            ]
            gr.Examples(examples=example_images, inputs=image_input)

        with gr.Column():
            classify_button = gr.Button("Classify")
            label_output = gr.Label(num_top_classes=3, label="Top Classes")
            predicted_label_output = gr.Textbox(label="Predicted Skin disease")
    
    with gr.Row():
        desp = gr.Textbox(label="Disease description")
    with gr.Row():
        fat = gr.Textbox(label="Fatality")
    with gr.Row():
        can = gr.Textbox(label="Cancerous or not")
    with gr.Row():
        act = gr.Textbox(label="Course of action")
    with gr.Row():
        gr.Markdown("<h6 style='text-align: center;'>For any suspicious skin lesion or changes, it’s essential to consult a dermatologist promptly. Regular skin checks and early intervention can significantly reduce risks. </h6>")
    classify_button.click(classify_image, inputs=image_input, outputs=[predicted_label_output, label_output, desp, fat, can, act])
    with gr.Row():
        gr.Markdown("## LIME Explanation")
    with gr.Row():
        lime_button = gr.Button("Get LIME Output")
        image_output = gr.Image(label="Influential Parts of the Image", type="pil")

        lime_button.click(lime_explanation, inputs=image_input, outputs=image_output)

# Launch the interface
demo.launch()
