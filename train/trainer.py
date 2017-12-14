from keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import pandas as pd

from models.extensions.custom_metrics import Metrics


class Trainer():
    """
        Responsible for managing data
    """
    def __init__(self, X, y, y_orig):
        self.X, self.y = X, y
        self.X_train, self.X_validation, self.Y_train, self.Y_validation, self.y_train_orig, self.y_val_orig = train_test_split(X, y, y_orig, test_size=0.10,
                                                                                            random_state=42)

    def train(self, model):
        """

        :param model: Model that is trained using
        :param X: validation + train
        :param y: validation + train
        :return: trained model
        """
        gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                                 height_shift_range=0.08, zoom_range=0.08)


        # TODO: this should be removed and build our own dataset
        test_gen = ImageDataGenerator()
        train_generator = gen.flow(self.X_train, self.Y_train, batch_size=64)

        # test.run generator not used because of custom emtrics
        test_generator = test_gen.flow(self.X_validation, self.Y_validation, batch_size=64)

        steps_per_epoch = int(self.X_train.shape[0] / 64)
        validation_steps=int(self.X_validation.shape[0] / 64)
        print("steps per epoch")
        print(steps_per_epoch)
        print("validation steps")
        print(validation_steps)

        metrics = Metrics()

        model.fit_generator(train_generator, epochs=20, verbose=1,
                            validation_data=[self.X_validation, self.Y_validation], callbacks=[metrics])

        return model

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
        predictions = model.predict_classes(self.X)

        predictions = list(predictions)
        actuals = list(self.y)

        sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
        print(sub)
        sub.to_csv('output_cnn.csv', index=False)


    def export(self, model):
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights('my_model_weights.h5')
        print("Saved model to disk")
