"""..."""

import os
import pickle
import time
import cv2
import imutils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'cascades/lbpcascade_frontalface.xml'))
labels = {}
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("data/trainer.yml")

with open("data/labels.pickle", 'rb') as f:
    labels = {v:k for k, v in pickle.load(f).items()}

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/home/emerson/Downloads/xx.mp4")

while True:

    check, frame = cap.read()
    frame = imutils.resize(frame, width=550)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 50, 150), 2)
        region_of_interest = gray_frame[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(region_of_interest)
        if(confidence >= 99.50 and confidence <= 100):
            name = labels[id_]
            label = "{}: {:.2f}%".format(name, confidence)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
            print('label: ' + label)
            time.sleep(5)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(20)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
