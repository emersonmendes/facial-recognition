"""..."""

import os
import pickle
import cv2
import numpy as np
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")
frontalface = os.path.join(base_dir, 'cascades/lbpcascade_frontalface.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

y_labels = []
x_train = []

label_ids = {}

def init():

    current_id = 0
    face_cascade = cv2.CascadeClassifier(frontalface)

    for root, _, files in os.walk(image_dir):
        for file in files:
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            image = Image.open(path).convert("L")
            image = image.resize((550, 550), Image.ANTIALIAS)

            image_array = np.array(image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

            # get region_of_interest
            for(x_cord, y_cord, width, height) in faces:
                region_of_interest = image_array[y_cord:y_cord + height, x_cord:x_cord + width]
                x_train.append(region_of_interest)
                y_labels.append(id_)

    # Save labels id

    recognizer.train(x_train, np.array(y_labels))
    save_data()

def save_data():
    recognizer.save("data/trainer.yml")
    with open("data/labels.pickle", 'wb') as file:
        pickle.dump(label_ids, file)
    print('Successfully :)')

init()
