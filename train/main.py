
import prep
import trainer
from models import complex_cnn, simple_cnn, smile_cnn, emotion_recognition_cnn, paper

if __name__ == "__main__":
    X,y = prep.load_faces_dataset()
    X,y, y_orig, class_weight = prep.prepare_data(X,y)
    model = emotion_recognition_cnn.build_model()
    trainer = trainer.Trainer(X,y, y_orig)
    model = trainer.train(model)
    trainer.evaluate(model)
    trainer.predict(model)
    trainer.export(model)