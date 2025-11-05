import cv2
import os
import numpy as np
import json

# Base paths
BASE = r"D:\FaceRecognitionPython"
DATASET = os.path.join(BASE, "dataset")
TRAINER = os.path.join(BASE, "trainer.yml")
LABEL_MAP = os.path.join(BASE, "label_map.json")
CASCADE = os.path.join(BASE, "haarcascade_frontalface_default.xml")

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE)

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []
label_map = {}
label_id = 1

print("🔄 Training started...")

# Loop through dataset folders (each folder = person)
for person_name in sorted(os.listdir(DATASET)):
    person_folder = os.path.join(DATASET, person_name)

    # Skip non-folder items or intruder folder
    if not os.path.isdir(person_folder) or person_name.lower() == "intruders":
        continue

    label_map[label_id] = person_name  # Example: 1 → Pavan

    # Read each image
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        # Read in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        ids.append(label_id)

    label_id += 1

# Train the model
if len(faces) == 0:
    print("❌ No face data found. Please run create_dataset first.")
    exit()

recognizer.train(faces, np.array(ids))
recognizer.save(TRAINER)

# Save label map
with open(LABEL_MAP, "w") as f:
    json.dump(label_map, f)

print("✅ Training completed!")
print(f"✅ Model saved as: {TRAINER}")
print(f"✅ Label mapping saved as: {LABEL_MAP}")
print("✅ Label Map:", label_map)
