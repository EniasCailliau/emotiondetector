import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split

import custom_metrics

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

negative_paths = list(list_all_files('faces/dbase/negatives/negatives7/', ['.jpg']))
print ('loaded', len(negative_paths), 'negative examples')
positive_paths = list(list_all_files('faces/dbase/positives/positives7/', ['.jpg']))
print ('loaded', len(positive_paths), 'positive examples')
examples = [(path, 0) for path in negative_paths] + [(path, 1) for path in positive_paths]


def faces_dataset(examples, block_size=1):
    X = []
    y = []
    for path, label in examples:
        img = imread(path, as_grey=True)
        img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
        X.append(img)
        y.append(label)
    return np.asarray(X), np.asarray(y)


X, y = faces_dataset(examples)

X = X.astype(np.float32) / 255.
y = y.astype(np.int32)
print (X.dtype, X.min(), X.max(), X.shape)
print (y.dtype, y.min(), y.max(), y.shape)

from utils import make_mosaic, show_array

show_array(255 * make_mosaic(X[:len(negative_paths)], 8), fmt='jpeg')  # negative at the beginning
show_array(255 * make_mosaic(X[-len(positive_paths):], 8), fmt='jpeg')  # positive at the end

X = np.expand_dims(X, axis=-1)

# convert classes to vector
nb_classes = 2
y = np_utils.to_categorical(y, nb_classes).astype(np.float32)

# shuffle all the data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# prepare weighting for classes since they're unbalanced
class_totals = y.sum(axis=0)
class_weight = class_totals.max() / class_totals

print (X.dtype, X.min(), X.max(), X.shape)
print (y.dtype, y.min(), y.max(), y.shape)

# make your own convnet here...

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

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.10, random_state=42)

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

model.fit_generator(train_generator, steps_per_epoch=int(13165 // 64), epochs=20,
                    validation_data=test_generator, validation_steps=int(10000 // 6 * 64))

score = model.evaluate(X_test, Y_test)
print('\nTest accuracy: ', score[1])

score = model.evaluate(X, y)
print('\nTraining accuracy: ', score[1])

predictions = model.predict_classes(X)

predictions = list(predictions)
actuals = list(y)

sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
print(sub)
sub.to_csv('output_cnn.csv', index=False)




# store model.json and weights.h5 and use this to 

# print the precision recall...
