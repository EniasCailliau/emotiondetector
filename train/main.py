from train import prep
from train import trainer
from models import emotion_recognition_cnn
if __name__ == "__main__":
    X,y = prep.unpickle_faces_dataset()
    X,y, y_orig, class_weight = prep.prepare_data(X, y)
    model = emotion_recognition_cnn.build_model()
    trainer = trainer.Trainer(X, y, y_orig)
    model = trainer.train(model)
    trainer.evaluate(model)
    trainer.predict(model)
    trainer.export(model)