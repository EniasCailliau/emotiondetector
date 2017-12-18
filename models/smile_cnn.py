import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam

from models.extensions.metrics import Metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def build_model():
    model = Sequential()
    model.add(Conv2D(16, (9, 9), input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(),
                  metrics=['accuracy'])

    return model