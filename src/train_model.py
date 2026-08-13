import cv2
import os
import numpy as np

dataset_path = "../dataset"

faces = []
labels = []

label_map = {}
current_label = 0

for student_folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, student_folder)

    if not os.path.isdir(folder_path):
        continue

    label_map[current_label] = student_folder

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            faces.append(img)
            labels.append(current_label)

    current_label += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(labels)
)

recognizer.save("../trainer.yml")

print("Model training complete!")
print("trainer.yml file created")
print(label_map)
