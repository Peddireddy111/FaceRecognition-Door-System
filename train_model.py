import cv2
import numpy as np
import os

dataset_path = r"D:\FaceRecognitionPython\dataset"
trainer_path = r"D:\FaceRecognitionPython\trainer.yml"
cascade_path = r"D:\FaceRecognitionPython\FaceRecognitionPython\haarcascade_frontalface_default.xml"

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cascade_path)

faces = []
ids = []

print("📸 Training faces...")

for idx, person in enumerate(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue
    label_id = idx + 1
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        detected = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in detected:
            faces.append(img[y:y+h, x:x+w])
            ids.append(label_id)

if len(faces) == 0:
    print("❌ No faces found. Please check your dataset.")
    exit()

recognizer.train(faces, np.array(ids))
recognizer.save(trainer_path)
print(f"✅ Training completed. Model saved to {trainer_path}")
