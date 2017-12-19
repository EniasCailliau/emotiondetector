import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import pandas as pd

from models.extensions.early_stop import Early_stop
from models.extensions.metrics import Metrics


class Trainer():
    """
        Responsible for managing data
    """

    def __init__(self, X, y, augment):
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(X, y, test_size=0.10,
                                                                                            random_state=42)
        self.batch_size = 64
        if (augment):
            print("The trainer is going to augment the data at runtime")
            train_gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.08, shear_range=0.3,
                                           height_shift_range=0.08, zoom_range=0.08)
        else:
            print("The trainer is NOT going to augment the data at runtime")
            train_gen = ImageDataGenerator()
        self.train_generator = train_gen.flow(self.X_train, self.Y_train, batch_size=self.batch_size)
        print(self.X_train.shape)
        print(self.X_validation.shape)
        print(self.Y_train.shape)
        print(self.Y_validation.shape)

    def train(self, model, task_file):
        metrics = Metrics(self.X_train, self.Y_train, task_file)

        # early_stop_val = EarlyStopping(monitor='val_acc', min_delta=0, patience=25, verbose=1, mode='auto')
        # early_stop = EarlyStopping(monitor='acc', min_delta=0, patience=25, verbose=1, mode='auto')

        filepath = task_file + "weights-{val_acc:.5f}.h5"
        early_stop = Early_stop(filepath=filepath)



        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')


        callbacks = [ early_stop, checkpoint, metrics]

        model.fit_generator(self.train_generator, epochs=500, verbose=1,
                            validation_data=[self.X_validation, self.Y_validation], callbacks=callbacks)

        return model, metrics

    def __evaluate(self, model, X, y, type):
        score = model.evaluate(X, y)
        print("Results for {} set".format(type))
        print("--- {}".format(score))
        print("metric names:")
        print(model.metrics_names)
        return score

    def evaluate(self, model):
        self.__evaluate(model, self.X_train, self.Y_train, "training")
        self.__evaluate(model, self.X_validation, self.Y_validation, "validation")

    def predict(self, model):
        # TODO: this needs to be reviewed
        return model.predict_classes(self.X_train)

    def export(self, model, log_folder):
        model_json = model.to_json()
        with open(os.path.join(log_folder, "model.json"), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.path.join(log_folder, 'final_model_weights.h5'))
        print("Saved last model to disk")
