import tensorflow as tf
from keras.optimizers import Adam

from models.extensions import custom_metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session

set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


def model():
    model = Sequential()

    model.add(Conv2D(68, (2, 2), input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    # BatchNormalization(axis=-1)

    model.add(Conv2D(68, (4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # BatchNormalization(axis=-1)
    model.add(Conv2D(68, (4, 4)))
    model.add(Activation('relu'))

    # BatchNormalization(axis=-1)
    model.add(Conv2D(68, (4, 4)))
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

    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['accuracy', custom_metrics.recall, custom_metrics.precision])

    return model