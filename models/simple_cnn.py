import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

from models.extensions import custom_metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session

set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

from utils import list_all_files
import numpy as np
from skimage.measure import block_reduce
from skimage.io import imread

import pandas as pd

def model():
    model = Sequential()

    # add convlayers and denselayers

    model.add(Conv2D(32, (2, 2), input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    # BatchNormalization(axis=-1)

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # BatchNormalization(axis=-1)
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))

    # BatchNormalization(axis=-1)
    model.add(Conv2D(64, (4, 4)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (4, 4)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))

    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer=Adam(),
                  metrics=['accuracy', custom_metrics.recall_custom, custom_metrics.precision_custom])
    return model