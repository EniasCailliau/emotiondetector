import tensorflow as tf
from keras.optimizers import Adam

from models.extensions import metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session

set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


def build_model():
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
                  metrics=['accuracy', metrics.calc_recall, metrics.precision])
    return model