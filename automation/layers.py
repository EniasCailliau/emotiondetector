import tensorflow as tf

from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout,AveragePooling2D,BatchNormalization,Activation

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Dropout_layer(object):
    def __init__(self, rate=0.25):
        self.rate=rate

    def construct(self):
        return Dropout(rate=self.rate)

    @classmethod
    def parse_args(cls, encoded):
        kwargs = {}
        for arg in encoded:
            if arg.startswith('rate='):
                par_string = arg[arg.index('=') + 1:]
                kwargs['rate'] = float(par_string)
        return kwargs

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
        if self.input_shape :
            return Conv2D(self.filters, kernel_size=self.kernel_size,
                   activation=self.activation, input_shape=self.input_shape)
        return Conv2D(self.filters, kernel_size=self.kernel_size,
                   activation=self.activation)

    @classmethod
    def parse_args(cls, encoded):
        kwargs = {}
        for arg in encoded:
            if arg.startswith('filters='):
                par_string = arg[arg.index('=') + 1:]
                kwargs['filters'] = int(par_string.strip())
            elif arg.startswith('kernel_size='):
                par_string = arg[arg.index('=') + 1:]
                # drop the braces
                par_array = par_string[1:-1].split(',')
                kwargs['kernel_size'] = (int(par_array[0].strip()),int(par_array[1].strip()))
            elif arg.startswith('input_shape='):
                par_string = arg[arg.index('=') + 1:]
                # drop the braces
                print(arg.index)
                par_array = par_string[1:-1].split(',')
                print(par_array)
                kwargs['input_shape'] = (int(par_array[0].strip()),int(par_array[1].strip()),int(par_array[2].strip()))
        return kwargs

    def __str__(self):
        return "Conv2D_layer, filters={}, kernel_size={}, activation={}, input_shape={} ".format(self.filters,self.kernel_size,self.activation,self.input_shape)

class AveragePooling2D_layer(object):
    def __init__(self, pool_size=(2, 2)):
        self.pool_size=pool_size

    def construct(self):
        return AveragePooling2D(pool_size=self.pool_size)

    @classmethod
    def parse_args(cls, encoded):
        kwargs = {}
        for arg in encoded:
            if arg.startswith('pool_size='):
                par_string = arg[arg.index('=') + 1:]
                par_array = par_string[1:-1].split(',')
                kwargs['pool_size'] = (int(par_array[0].strip()),int(par_array[1].strip()))
        return kwargs
    def __str__(self):
        return "AveragePooling2D_layer, pool_size={}".format(self.pool_size)


class MaxPooling2D_layer(object):
    def __init__(self, pool_size=(2, 2)):
        self.pool_size=pool_size

    def construct(self):
        return MaxPooling2D(pool_size=self.pool_size)

    @classmethod
    def parse_args(cls, encoded):
        kwargs = {}
        for arg in encoded:
            if arg.startswith('pool_size='):
                par_string = arg[arg.index('=') + 1:]
                par_array = par_string[1:-1].split(',')
                kwargs['pool_size'] = (int(par_array[0].strip()),int(par_array[1].strip()))
        return kwargs
    def __str__(self):
        return "MaxPooling2D_layer, pool_size={}".format(self.pool_size)


class Activation_layer(object):
    def __init__(self, activation="relu"):
        self.activation=activation

    def construct(self):
        return Activation(activation=self.activation)

    @classmethod
    def parse_args(cls, encoded):
        kwargs = {}
        for arg in encoded:
            if arg.startswith('activation='):
                par_string = arg[arg.index('=') + 1:]
                kwargs['activation'] = par_string
        return kwargs
    def __str__(self):
        return "Activation_layer, activation={}".format(self.activation)

class Dense_layer(object):
    def __init__(self, units=2,activation='relu'):
        self.units=units
        self.activation=activation

    def construct(self):
        return Dense(units=self.units,activation=self.activation)

    @classmethod
    def parse_args(cls, encoded):
        kwargs = {}
        for arg in encoded:
            if arg.startswith('units='):
                par_string = arg[arg.index('=') + 1:]
                kwargs['units'] = int(par_string.strip())
        return kwargs

    def __str__(self):
        return "Dense_layer, units={}, activation={}".format(self.units,self.activation)

class Flatten_layer(object):

    def construct(self):
        return Flatten()

    @classmethod
    def parse_args(cls, encoded):
        return {}
    def __str__(self):
        return "Flatten_layer"

class BatchNormalization_layer(object):

    def construct(self):
        return BatchNormalization()

    @classmethod
    def parse_args(cls, encoded):
        return {}

    def __str__(self):
        return "BatchNormalization"





class LayerFactory(object):
    layer_classes = {
        'conv2d': Conv2D_layer,
        'dense': Dense_layer,
        'maxpooling2d': MaxPooling2D_layer,
        'averagepooling2d': AveragePooling2D_layer,
        'dropout': Dropout_layer,
        'flatten': Flatten_layer,
        'batchnormalization':BatchNormalization_layer,
        'activation': Activation_layer,

    }

    def new_from_string(self, creation_string):
        splitted = creation_string.split()
        kwargs = LayerFactory.layer_classes[splitted[0]].parse_args(splitted[1:])
        return LayerFactory.layer_classes[splitted[0]](**kwargs)
