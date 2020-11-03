import numpy as np
import pandas as pd
import os 
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_io as tfio

from keras.utils import HDF5Matrix
#filter out HDF5Matrix deprecation warning
'''import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)'''

# constants for model
BATCH_SIZE = 32
IMG_HEIGHT, IMG_WIDTH = 96, 96
NUM_CLASSES = 2         # two classes indicating positive or negative tumor
EPOCHS = 10

def load_images():
    '''import images from folders
        x = images
        y = labels 
    :return: HDF5Matrix '''

    


    try:
        x_train = HDF5Matrix('images/train/camelyonpatch_level_2_split_train_x.h5', 'x')
        y_train = HDF5Matrix('images/train/camelyonpatch_level_2_split_train_y.h5', 'y')

        x_valid = HDF5Matrix('images/validate/camelyonpatch_level_2_split_valid_x.h5', 'x')
        y_valid = HDF5Matrix('images/validate/camelyonpatch_level_2_split_valid_y.h5', 'y')

        x_test = HDF5Matrix('images/test/camelyonpatch_level_2_split_test_x.h5', 'x')
        y_test = HDF5Matrix('images/test/camelyonpatch_level_2_split_test_y.h5', 'y')

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    except OSError as e:
        print("[load_image OSError] ", e)


def create_model():
    '''creates CNN model for the algorithm
    :reutrn: model for the algorithm'''

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES)
    ])

    return model

if __name__ == '__main__':

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_images()
    print(len(x_train))

    print(x_train.shape)

    # Zip together samples and corresponding labels
    '''train = tf.data.Dataset.zip((x_train,y_train)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    test = tf.data.Dataset.zip((x_test,y_test)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    valid = tf.data.Dataset.zip((x_valid,y_valid)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
'''
    model = create_model()

    # compile model
    try:
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    except Exception as e:
        print("[Model compilation error]", e)

    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        epochs=EPOCHS,
        shuffle=False
    )





