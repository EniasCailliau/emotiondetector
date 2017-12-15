import csv

import keras
import numpy as np
from keras import backend as K
from sklearn.metrics import recall_score, precision_score, accuracy_score


class Metrics(keras.callbacks.Callback):

    def __init__(self, x_train, y_train,task_file):
        super().__init__()
        self.recall_list = []
        self.precision_list = []
        self.accuracy_list = []
        self.x_train = x_train
        self.y_train = y_train
        self.task_file = task_file+".csv"

        fields=["epoch","accuracy","loss","precision","recall","val_accuracy","val_loss","val_precision","val_recall"]
        with open(self.task_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def on_epoch_end(self, batch, logs=None):
        y_pred_labels = np.argmax(self.model.predict(self.x_train), axis=1)
        y_labels = np.argmax(self.y_train, axis=1)
        fields=[]
        fields.append(batch)
        recall = recall_score(y_labels, y_pred_labels)
        print("- Training recall:  {}".format(recall))
        precision = precision_score(y_labels, y_pred_labels)
        print("- Training precision:  {}".format(precision))
        accuracy = accuracy_score(y_labels, y_pred_labels)
        print("- Training accuracy:  {}".format(accuracy))
        fields.append(accuracy)
        fields.append(logs["loss"])
        fields.append(precision)
        fields.append(recall)


        y_pred_labels = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        y_labels = np.argmax(self.validation_data[1], axis=1)

        recall = recall_score(y_labels, y_pred_labels)
        print("- Validation recall:  {}".format(recall))
        precision = precision_score(y_labels, y_pred_labels)
        print("- Validation precision:  {}".format(precision))
        accuracy = accuracy_score(y_labels, y_pred_labels)
        print("- Validation accuracy:  {}".format(accuracy))
        fields.append(accuracy)
        fields.append(logs["val_loss"])
        fields.append(precision)
        fields.append(recall)


        self.recall_list.append(recall)
        self.precision_list.append(precision)
        self.accuracy_list.append(accuracy)
        with open(self.task_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        return
