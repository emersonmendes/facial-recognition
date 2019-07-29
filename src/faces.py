import numpy as np
import cv2
import pickle
import os
import imutils
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'cascades/lbpcascade_frontalface.xml'))
labels = {}
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("data/trainer.yml") 
color = (255,255,255)

with open("data/labels.pickle", 'rb') as f: 
    labels = {v:k for k,v in pickle.load(f).items()}

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/home/emerson/Downloads/xx.mp4")

while(True):
    # capture frame by frame
    ret, original_frame = cap.read()
    original_frame  = imutils.resize(original_frame, width=550)

    gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)   

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    found = False
    for(x, y, w, h) in faces:
        region_of_interest = gray_frame[y:y + h, x:x + w]
        id_, confidence = recognizer.predict(region_of_interest)
        if(confidence >= 99 and confidence <= 100):
            label = ""
            name = labels[id_]
            label = "{}: {:.2f}%".format(name, confidence)
            cv2.putText(original_frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA )
            print('label: ' + label)
            found = True
    
        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (255,50,150), 2)

    cv2.imshow('frame', original_frame)
    
    if found:
        time.sleep(3)          
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
