import tensorflow as tf
from keras import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def build_model():
    model = Sequential()
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=0, mode='auto')
    # filepath = "weights-{val_acc:.4f}.h5"
    filepath = "weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    logger = CSVLogger("logs.csv", separator=";", append=False)
    callbacks = [early_stop, checkpoint, logger]

    model.add(Conv2D(32, (7, 7), input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))

    # model.add(Dropout(0.5))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))


    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
