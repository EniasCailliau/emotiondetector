import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, AveragePooling2D, Dropout
from keras.optimizers import Adam



class Dropout_layer(object):
    def __init__(self, rate=0.25):
        self.rate=rate

    def construct(self):
        return Dropout(rate=self.rate)

    def __str__(self):
        return "Dropout, rate={}".format(self.rate)


class Conv2D_layer(object):
    def __init__(self, filters=32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=None):
        self.filters=filters
        self.kernel_size=kernel_size
        self.activation=activation
        self.input_shape=input_shape

    def construct(self):
        return Conv2D(self.filters, kernel_size=self.kernel_size,
                 activation=self.activation,
                 input_shape=self.input_shape)

    def __str__(self):
        return "Conv2D_layer, filters={}, kernel_size={}, activation={}, input_shape={} ".format(self.filters,self.kernel_size,self.activation,self.input_shape)

class AveragePooling2D_layer(object):
    def __init__(self, pool_size=(2, 2)):
        self.pool_size=pool_size

    def construct(self):
        return AveragePooling2D(pool_size=self.pool_size)

    def __str__(self):
        return "AveragePooling2D_layer, pool_size={}".format(self.pool_size)


class MaxPooling2D_layer(object):
    def __init__(self, pool_size=(2, 2)):
        self.pool_size=pool_size

    def construct(self):
        return MaxPooling2D(pool_size=self.pool_size)

    def __str__(self):
        return "MaxPooling2D_layer, pool_size={}".format(self.pool_size)

class Dense_layer(object):
    def __init__(self, units=2,activation='softmax'):
        self.units=units
        self.activation=activation

    def construct(self):
        return Dense(units=self.units,activation=self.activation)

    def __str__(self):
        return "Dense_layer, units={}, activation={}".format(self.units,self.activation)

class Flatten_layer(object):
    def __init__(self):

    def construct(self):
        return Flatten()

    def __str__(self):
        return "Flatten_layer"

class LayerFactory(object):
    layer_classes = {
        'conv2d': Conv2D_layer,
        'dense': Dense_layer,
        'maxpooling2d': MaxPooling2D_layer,
        'averagepooling2d': AveragePooling2D_layer,
        'dropout': Dropout_layer,
        'flatten': Flatten_layer,

    }

    def new_from_string(self, creation_string):
        splitted = creation_string.split()
        kwargs = FeatureExtractorFactory.extractor_classes[splitted[0]].parse_args(splitted[1:])
        return FeatureExtractorFactory.extractor_classes[splitted[0]](**kwargs)
