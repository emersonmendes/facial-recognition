"""..."""

import os
import pickle
import cv2
import numpy as np
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")

y_labels = []
x_train = []

current_id = 0
label_ids = {}

face_cascade = cv2.CascadeClassifier(os.path.join(base_dir, 'cascades/lbpcascade_frontalface.xml'))

recognizer = cv2.face.LBPHFaceRecognizer_create()

for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(root).replace(" ", "-").lower()
 
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1
        id_ = label_ids[label]

        # convert to gray
        image = Image.open(path).convert("L")

        image = image.resize((550, 550), Image.ANTIALIAS)

        image_array = np.array(image, "uint8")
        faces = face_cascade.detectMultiScale(image_array)
        #faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)

        # get region_of_interest
        for(x, y, w, h) in faces:
            region_of_interest = image_array[y:y + h, x:x + w]
            x_train.append(region_of_interest)
            y_labels.append(id_)

# Save labels id
with open("data/labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))

recognizer.save("data/trainer.yml")

print('Successfully :)')