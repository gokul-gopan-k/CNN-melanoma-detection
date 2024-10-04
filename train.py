#Library for data manipulation
import pandas as pd
#Library for scientific calculation
import numpy as np

#libraries for data visulalization
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
#library for getting path
import pathlib
# library to interact with os
import os
# import package
import Augmentor


#modules and libraries to build ML model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Defining the path for train and test images
# Path for train data
data_dir_train = pathlib.Path("Skin cancer ISIC The International Skin Imaging Collaboration/Train/")
# Path for test data
data_dir_test = pathlib.Path("Skin cancer ISIC The International Skin Imaging Collaboration/Test/")

# no of images
batch_size = 32
# size to which image to be resized
img_height = 180
img_width = 180

class_names = train_ds.class_names

# setup for better memory utilisation
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# path to initial train data
path_to_training_dataset="Skin cancer ISIC The International Skin Imaging Collaboration/Train/"

for i in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset + i)
    # define augmentation operations
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    #adding 500 samples per class to make sure that none of the classes are sparse.
    p.sample(500)


# path of new original+augmented dataset
data_dir_train="Skin cancer ISIC The International Skin Imaging Collaboration/Train/"
# get tarin dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=123,
  validation_split = 0.2,
  subset = "training",
  image_size=(img_height, img_width),
  batch_size=batch_size)

# get validation dataset.
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=123,
  validation_split = 0.2,
  subset = "validation",
  image_size=(img_height, img_width),
  batch_size=batch_size)

# create custom model
model =Sequential()

# scaling the pixel values to 0-1 range
model.add(layers.Rescaling(scale=1./255,input_shape=(180,180,3)))
# data augmentation layer
model.add(data_augmentation)

# Convolution layer with 16 filters, 3x3 filter and relu activation with Max pooling
model.add(layers.Conv2D(16,(3,3),padding = 'same',activation='relu'))
model.add(layers.MaxPooling2D())

# Convolution layer with 32 filters, 3x3 filter and relu activation with Max pooling
model.add(layers.Conv2D(32,(3,3),padding = 'same',activation='relu'))
model.add(layers.MaxPooling2D())

# Convolution layer with 64 filters, 3x3 filter and relu activation with Max pooling
model.add(layers.Conv2D(64,(3,3),padding = 'same',activation='relu'))
model.add(layers.MaxPooling2D())

#adding a 20% dropout
model.add(layers.Dropout(0.2))

# flatten the output before dense layer
model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu'))
# neurons in last layer is no of classes and softmax due to multi class classification
model.add(layers.Dense(9,activation='softmax'))

## compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# no of epochs
epochs = 30
# fit model on train and validation data
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Assume model is your trained model (e.g., a scikit-learn model)
joblib.dump(model, 'model_filename.pkl')
