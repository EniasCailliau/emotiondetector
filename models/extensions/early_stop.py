import csv

import keras
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score



class Early_stop(keras.callbacks.Callbac):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self,
                 min_delta=0, patience=0, verbose=0, mode='auto',filepath=None):
        super(Early_stop, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.filepath=filepath

        self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.evaluate(logs)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.model.save_weights(self.filepath, overwrite=True)

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def evaluate(self, logs):
        y_pred = self.model.predict(self.validation_data[0])
        y = self.validation_data[1]
        return  self.calculate_roc_auc( y, y_pred)

    @staticmethod
    def calculate_roc_auc( y, y_pred):
        roc_auc = roc_auc_score(y, y_pred) * 100
        return roc_auc


