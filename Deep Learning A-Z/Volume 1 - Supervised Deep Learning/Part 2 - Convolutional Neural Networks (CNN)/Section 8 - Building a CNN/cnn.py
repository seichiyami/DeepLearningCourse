# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:19:30 2017

@author: Brandon
"""

# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD

IMAGE_INPUT_X = 128
IMAGE_INPUT_Y = 128
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.4
DECAY = LEARNING_RATE / (EPOCHS)

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, kernel_size = (3, 3), input_shape = (IMAGE_INPUT_X, IMAGE_INPUT_Y, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, kernel_size = (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(p = 0.2))
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(p = 0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# sgd = SGD(lr = LEARNING_RATE, momentum = 0.8, decay = DECAY, nesterov = False)
# Compilling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(IMAGE_INPUT_X, IMAGE_INPUT_Y),
        batch_size=BATCH_SIZE,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(IMAGE_INPUT_X, IMAGE_INPUT_Y),
        batch_size=BATCH_SIZE,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_set,
        validation_steps=2000/BATCH_SIZE)

# use model to predict single image
import numpy as np
from keras.preprocessing import image

single_image = image.load_img("dataset/single_prediction/cat_or_dog_2.jpg", target_size = (IMAGE_INPUT_X, IMAGE_INPUT_Y))
single_image = image.img_to_array(single_image)
single_image = np.expand_dims(single_image, axis = 0)
result = classifier.predict(single_image)

training_set.class_indices
if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"





classifier.save("model_best.h5")
del classifier