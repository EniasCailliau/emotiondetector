import numpy as np
from keras.models import Sequential
import keras
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle

import tensorflow as tf

import prep

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session

set_session(tf.Session(config=config))


def load_data():
    name = 'data'
    print("Loading current train sets..")
    X_train = pickle.load(open("X_train_" + name + ".pkl", "rb"))
    y_train = pickle.load(open("y_train_" + name + ".pkl", "rb"))
    print("Loaded!")
    return X_train, y_train

    # based on http://cs231n.stanford.edu/reports/2016/pdfs/022_Report.pdf


X, y = prep.load_faces_dataset()
X, y, y_orig, class_weight = prep.prepare_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential()
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=0, mode='auto')
# filepath = "weights-{val_acc:.4f}.h5"
filepath = "weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
logger = CSVLogger("logs.csv", separator=";", append=False)
callbacks = [early_stop, checkpoint, logger]

model.add(Conv2D(32, 7, 7, activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=200,
          verbose=1, validation_data=(X_test, y_test), callbacks=callbacks)

# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print("Achieved test accuracy = {0}% with test loss {1}".format(acc*100, loss))

# model.save("weights.h5")
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()
# store model.json and weights.h5 and use this to

# print the precision recall...
