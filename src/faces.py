import numpy as np
import cv2
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'cascades/data/haarcascade_frontalface_default.xml'))
labels = {}
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml") 

with open("labels.pickle", 'rb') as f: 
    labels = {v:k for k,v in pickle.load(f).items()}

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/home/emerson/Downloads/xx.mp4")

while(True):
    # capture frame by frame
    ret, original_frame = cap.read()
    gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
    
    for(x, y, w, h) in faces:
        cord_y = y + h
        cord_x = x + w
        region_of_interest = gray_frame[y:cord_y, x:cord_x]

        id_, confidence = recognizer.predict(region_of_interest)
        if(confidence >= 45 and confidence <= 85):
            name = labels[id_]
            color = (255,255,255)
            label = "{}: {:.2f}%".format(name, confidence)
            cv2.putText(original_frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA )

        # draw rectangle
        cv2.rectangle(original_frame, (x, y), (cord_x, cord_y), (255,50,150), 2)

    # display result frame
    cv2.imshow('frame', original_frame)          
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
