import cv2 as cv
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR= os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

y_labels = []
x_train = []

current_id = 0
label_ids = {}

face_cascade = cv.CascadeClassifier('/home/emerson/Downloads/facial-recognition/src/cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/home/emerson/Downloads/facial-recognition/src/cascades/data/haarcascade_eye.xml')

recognizer = cv.face.LBPHFaceRecognizer_create()

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)
            

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L") # convert to gray

            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            image_array = np.array(final_image, "uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            cv.imwrite("my_img.png", image_array)

            # get region_of_interest
            for(x, y, w, h) in faces:
                end_cordinate_y = y + h
                end_cordinate_x = x + w
                region_of_interest = image_array[y:end_cordinate_y, x:end_cordinate_x]
                x_train.append(region_of_interest)
                y_labels.append(id_)

# Save labels id
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f) 

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")