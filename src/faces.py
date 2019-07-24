import numpy as np
import cv2 as cv
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv.CascadeClassifier(os.path.join(BASE_DIR, 'cascades/data/lbpcascade_frontalface.xml'))
labels = {}
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

def process_frames(cap):
    # capture frame by frame
    ret, original_frame = cap.read()
    gray_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        draw_data(x, y, w, h, gray_frame, original_frame)
    # display result frame
    cv.imshow('frame', original_frame)

def draw_data(y, x, h, w, gray_frame, original_frame):
    
    cord_y = y + h
    cord_x = x + w
    region_of_interest = gray_frame[y:cord_y, x:cord_x]

    id_, confidence = recognizer.predict(region_of_interest)
    if(confidence > 60):
        name = labels[id_]
        color = (255,255,255)
        label = "{}: {:.2f}%".format(name, confidence)
        cv.putText(original_frame, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA )

    # draw rectangle
    cv.rectangle(original_frame, (x, y), (cord_x, cord_y), (255,50,150), 2)

def init():
    cap = cv.VideoCapture(0)
    #cap = cv.VideoCapture("/home/emerson/Downloads/video.mp4")

    while(True):
        process_frames(cap)        
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows() 

def load_labels():
    labels = {}
    with open("labels.pickle", 'rb') as f: 
        labels = {v:k for k,v in pickle.load(f).items()}
    return labels

labels = load_labels()
init()