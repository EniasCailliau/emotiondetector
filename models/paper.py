import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, AveragePooling2D, Dropout
from keras.optimizers import Adam

from models.extensions.custom_metrics import Metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def build_model():
    model = Sequential()
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    callbacks = [early_stop]

    model.add(Conv2D(32, 5, 5, activation='relu', input_shape=(64, 64, 1)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
