import keras
import time
from keras.models import model_from_json
import numpy as np
from utils import show_array
from utils import crop_and_resize
import cv2
from train import prep

def print_indicator(data, model, class_names, bar_width=50):
    probabilities = model.predict(np.array([data]))[0]
    left_side = '-' * int(probabilities[1] * bar_width)
    right_side = '-' * int(probabilities[0] * bar_width)
    print(class_names[0], left_side + '##??##' + right_side, class_names[1])

def print_probas(data, model):
    probabilities = model.predict(np.array([data]))[0]
    print("Smile: ",probabilities[1] )
    print("Neutral: ",probabilities[0] )

# load in the keras model trained in training.py

model = model_from_json(open('model.json').read())
model.load_weights('weights-0.98861.h5')


# X = np.load('X.npy')

X,examples=prep.load_eval_faces_dataset()

#X = X.astype(np.float) / 255.
class_names = ['Neutral', 'Smiling']
print(X.shape)
for i in range(X.shape[0]):
    print(examples[i])
    print_indicator(X[i], model, class_names)
    print_probas(X[i], model,)

#show_array(255 * X[-100])
#print_indicator(X[-100], model, class_names)


# video_capture = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("face-ll.jpg", gray)
#     data = crop_and_resize(gray, 64, zoom=1.)#was.6
#     if data.shape==(0, 0):
#         print("not doing anything")
#         continue
#
#     data = data.astype(np.float) / 255.
#     a = data.reshape(64, 64, 1)
#     print_indicator(a, model, class_names)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


