## How to run app interface
* Creare virtual environment

```python -m venv .venv ```  
```source .venv/bin/activate ```
or
```.venv\Scripts\Activate.ps1```
* Clone the repo
  
```git clone https://github.com/gokul-gopan-k/CNN-melanoma-detection.git```

```cd CNN-melanoma-detection```

* Make the script executable:(run in git bash for windows)
  
```chmod +x script.sh```

* Run the script:
  
```./script.sh```

* Run the app
  
```python app.py```






# CNN_melanoma_detection
# Melanoma-Detection

> To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
 
## Table of Contents
* [General Info](#general-information)
* [Project Pipeline](#project-pipeline)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)


## General Information

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following diseases:

* Actinic keratosis
* Basal cell carcinoma
* Dermatofibroma
* Melanoma
* Nevus
* Pigmented benign keratosis
* Seborrheic keratosis
* Squamous cell carcinoma
* Vascular lesion

### Business Goal:

Build a multiclass classification model using a custom convolutional neural network in TensorFlow. 


## Project Pipeline
- Data Reading/Data Understanding → Defining the path for train and test images 
- Dataset Creation→ Create train & validation dataset from the train directory with a batch size of 32. Image resized to 180*180.
- Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset 
- Model Building & training : 
  - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
  - Choose an appropriate optimiser and loss function for model training
  - Train the model for ~20 epochs
  - Check if there is any evidence of model overfit or underfit.
- Chose an appropriate data augmentation strategy to resolve underfitting/overfitting 
- Model Building & training on the augmented data :
  - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
  - Choose an appropriate optimiser and loss function for model training
  - Train the model for ~20 epochs
  - Write your findings after the model fit, see if the earlier issue is resolved or not?
- Class distribution: Examine the current class distribution in the training dataset 
  - Which class has the least number of samples?
  - Which classes dominate the data in terms of the proportionate number of samples?
- Handling class imbalances: Rectify class imbalances present in the training dataset with Augmentor library.
- Model Building & training on the rectified class imbalance data :
  - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
  - Choose an appropriate optimiser and loss function for model training
  - Train the model for ~30 epochs


## Conclusions
1) Built a CNN based model which can detect melanoma with accuracy of close to 80%.
2) Dataset has too few images from which model can learn to give good accuracy. Hence augmentation stategy was used in training.
3) Dataset is class imbalanced which lead to lower accuracy. Hence Augmentor package was used to increase number of images in each class.


## Technologies Used
- Notebook : version 6.4.8
- pandas : It is used for data cleaning and analysis. 
- numpy : NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, and matrices like to get mean,median etc. 
- seaborn : Seaborn is a library for making statistical graphics in Python.
- matplotlib.pyplot :Matplotlib is a cross-platform, data visualization and graphical plotting library for Python.
- scikit-learn :machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support-vector machine.
- tensorflow :  free and open-source software library for machine learning and artificial intelligence.
- keras :  open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.
- Augmentor :  Python package designed to aid the augmentation and artificial generation of image data for machine learning tasks.

## Acknowledgements
- This project was inspired by Upgrad.
- References: google, stackoverflow, upgrad classes.


## Contact
Created by [https://github.com/gokul-gopan-k] - feel free to contact me!

