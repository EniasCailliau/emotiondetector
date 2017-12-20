from train import prep
from train import trainer
from models import emotion_recognition_cnn


if __name__ == "__main__":
    X, y = prep.load_faces_dataset()
    X, y, class_weight = prep.prepare_data(X, y)
    model = emotion_recognition_cnn.build_model()
    trainer = trainer.Trainer(X, y, augment=False)
    model, metrics = trainer.train(model, "dump/")
    print(metrics.metrics)