import numpy as np
import cv2 as cv
import pickle

face_cascade = cv.CascadeClassifier('/home/emerson/Downloads/facial-recognition/src/cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/home/emerson/Downloads/facial-recognition/src/cascades/data/haarcascade_eye.xml')

recognizer = cv.face.LBPHFaceRecognizer_create()

labels = {"person_name":1}

# get labels  byid
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f) 
    labels = {v:k for k,v in og_labels.items()}

recognizer.read("trainer.yml")

cap = cv.VideoCapture(0)

while(True):
    
    # capture frame by frame
    ret, frame = cap.read()

    # Convert image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for(x, y, w, h) in faces:

        end_cordinate_y = y + h
        end_cordinate_x = x + w
        region_of_interest_gray = gray[y:end_cordinate_y, x:end_cordinate_x]

        # xxxx
        id_, confidence = recognizer.predict(region_of_interest_gray)
        if(confidence > 45):
            print(id_)
            print(labels[id_])
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv.putText(frame, name, (x, y), font, 1, color, stroke, cv.LINE_AA )

        # write last captured image 
        #cv.imwrite("my_img.png", region_of_interest_gray)

        # draw rectangle
        color = (255,0,0)
        stroke = 2
        cv.rectangle(frame, (x, y), (end_cordinate_x, end_cordinate_y), color, stroke)

    # display result frame
    cv.imshow('frame', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows() 