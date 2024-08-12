# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import pandas as pd
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+\
                                        'haarcascade_frontalface_default.xml')

model = model_from_json(open('new_model.json', 'r').read())
model.load_weights('new_model.h5')

cap = cv2.VideoCapture(0)

while True:
    sec, image = cap.read()
    #print(sec, image)
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade_db.detectMultiScale(converted_image)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0))

        face_gray = converted_image[y:y+w, x:x+h]
        face_gray = cv2.resize(face_gray, (48, 48))
        print(face_gray)
        image_pixels = np.array(face_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels = image_pixels.reshape(image_pixels.shape[0], 48, 48, 1)

        predictions = model.predict(image_pixels)
        print(predictions)
        max_index = np.argmax(predictions[0])
        emotion_detection = ('angry', 'disgust', 'fear', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotion_detection[max_index]

        cv2.putText(image, emotion_prediction, (int(x), int(y)),\
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        resize_image = cv2.resize(image, (1000, 600))
        cv2.imshow('Emotion by Yurin', resize_image)
    if cv2.waitKey(10) == ord('b'):
        break

cap.release()
cv2.destroyAllWindows()
print('sucsess')