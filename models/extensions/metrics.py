import csv

import keras
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score


class Metrics(keras.callbacks.Callback):

    def __init__(self, x_train, y_train, task_file):
        super().__init__()
        self.metrics = {
            'train': {
                'recall': [],
                'precision': [],
                'accuracy': [],
                'roc_auc': [],
                'loss': []
            },
            'test': {
                'recall': [],
                'precision': [],
                'accuracy': [],
                'roc_auc': [],
                'loss': []
            },
        }
        self.x_train = x_train
        self.y_train = y_train
        self.task_file = task_file + "metrics.csv"

        fields = ["epoch", "accuracy", "loss", "roc_auc", "precision", "recall", "val_accuracy", "val_loss",
                  "val_roc_auc", "val_precision", "val_recall"]
        with open(self.task_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def on_epoch_end(self, batch, logs=None):
        fields = []
        fields.append(batch)
        self.evaluate_train(logs, fields)
        self.evaluate_test(logs, fields)

        with open(self.task_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def evaluate_test(self, logs, fields):
        y_pred = self.model.predict(self.validation_data[0])
        y_pred_labels = np.argmax(y_pred, axis=1)
        y = self.validation_data[1]
        y_labels = np.argmax(y, axis=1)

        accuracy, loss, precision, recall, roc_auc = self.calculate_metrics(logs["val_loss"], y, y_labels, y_pred,
                                                                            y_pred_labels)

        self.__print_metrics("validation", accuracy, loss, precision, recall, roc_auc)

        self.__add_metrics_to_fields(fields, accuracy, loss, precision, recall, roc_auc)

        self.__store_metric_values('test', accuracy, loss, precision, recall, roc_auc)

    def evaluate_train(self, logs, fields):
        y_pred = self.model.predict(self.x_train)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y = self.y_train
        y_labels = np.argmax(y, axis=1)

        accuracy, loss, precision, recall, roc_auc = self.calculate_metrics(logs["loss"], y, y_labels, y_pred,
                                                                            y_pred_labels)

        self.__print_metrics("train", accuracy, loss, precision, recall, roc_auc)

        self.__add_metrics_to_fields(fields, accuracy, loss, precision, recall, roc_auc)

        self.__store_metric_values('train', accuracy, loss, precision, recall, roc_auc)

    @staticmethod
    def __print_metrics(what, accuracy, loss, precision, recall, roc_auc):
        print("- {} accuracy:  {}".format(what, accuracy))
        print("- {} loss:  {}".format(what, loss))
        print("- {} roc_auc: {}".format(what, roc_auc))
        print("- {} precision:  {}".format(what, precision))
        print("- {} recall:  {}".format(what, recall))

    def __store_metric_values(self, location, accuracy, loss, precision, recall, roc_auc):
        self.metrics[location]['accuracy'].append(accuracy)
        self.metrics[location]['loss'].append(loss)
        self.metrics[location]['roc_auc'].append(roc_auc)
        self.metrics[location]['precision'].append(precision)
        self.metrics[location]['recall'].append(recall)

    @staticmethod
    def __add_metrics_to_fields(fields, accuracy, loss, precision, recall, roc_auc):
        fields.append(accuracy)
        fields.append(loss)
        fields.append(roc_auc)
        fields.append(precision)
        fields.append(recall)

    @staticmethod
    def calculate_metrics(loss, y, y_labels, y_pred, y_pred_labels):
        accuracy = accuracy_score(y_labels, y_pred_labels) * 100
        roc_auc = roc_auc_score(y, y_pred) * 100
        recall = recall_score(y_labels, y_pred_labels) * 100
        precision = precision_score(y_labels, y_pred_labels) * 100
        return accuracy, loss, precision, recall, roc_auc
