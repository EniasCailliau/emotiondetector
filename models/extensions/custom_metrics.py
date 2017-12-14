import keras
import numpy as np
from keras import backend as K
from sklearn.metrics import recall_score, precision_score, accuracy_score


class Metrics(keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.recall_list = []
        self.precision_list = []
        self.accuracy_list = []

    def on_epoch_end(self, batch, logs=None):
        # TODO: Add precision recall accuracy for training set (problem because generator is used)
        y_pred_labels = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        y_labels = np.argmax(self.validation_data[1], axis=1)

        recall = recall_score(y_labels, y_pred_labels)
        print("- val_recall:  {}".format(recall))
        precision = precision_score(y_labels, y_pred_labels)
        print("- val_precision:  {}".format(precision))
        accuracy = accuracy_score(y_labels, y_pred_labels)
        print("- accuracy:  {}".format(accuracy))

        self.recall_list.append(recall)
        self.precision_list.append(precision)
        self.accuracy_list.append(accuracy)

        return
