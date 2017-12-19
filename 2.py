import numpy as np
from keras.models import Sequential
import keras
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle

import tensorflow as tf
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


X, y = load_data()
X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# convert classes to vector
nb_classes = 2
y = np_utils.to_categorical(y, nb_classes).astype(np.float32)

# shuffle all the data
indices = np.arange(len(X))
# np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# prepare weighting for classes since they're unbalanced
class_totals = y.sum(axis=0)
class_weight = class_totals.max() / class_totals

print(X.dtype, X.min(), X.max(), X.shape)
print(y.dtype, y.min(), y.max(), y.shape)

# make your own convnet here...

model = Sequential()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
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
#model.add(Dropout(0.5))
#model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=30, epochs=100, verbose=1, validation_data=(X_test, y_test),
          callbacks=callbacks)

# loss, acc = model.evaluate(X_test, y_test, verbose=0)
# print("Achieved test accuracy = {0}% with test loss {1}".format(acc*100, loss))

model.save("weights.h5")
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()
# store model.json and weights.h5 and use this to

# print the precision recall...
