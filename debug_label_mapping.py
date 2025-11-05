# debug_label_mapping.py
import cv2, json, os, numpy as np

trainer = r"D:\FaceRecognitionPython\trainer.yml"
dataset = r"D:\FaceRecognitionPython\dataset"
cascade = r"D:\FaceRecognitionPython\FaceRecognitionPython\haarcascade_frontalface_default.xml"

if not os.path.exists(trainer):
    print("trainer.yml not found:", trainer)
    raise SystemExit

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read(trainer)
print("Loaded trainer:", trainer)

# Build mapping by retracing training order: sorted folder list -> label ids
folders = sorted([d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))])
print("Folders (sorted):", folders)
print("Now predicting a sample from each folder to confirm label mapping...")

face_cascade = cv2.CascadeClassifier(cascade)
mapping = {}
for idx, person in enumerate(folders):
    person_dir = os.path.join(dataset, person)
    # find first image in folder
    imgs = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not imgs:
        continue
    img_path = os.path.join(person_dir, imgs[0])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Cannot read", img_path); continue
    faces = face_cascade.detectMultiScale(img, 1.2, 5)
    if len(faces)==0:
        print("No face found in", img_path)
        continue
    x,y,w,h = faces[0]
    face_roi = img[y:y+h, x:x+w]
    label,conf = rec.predict(face_roi)
    mapping[label] = person
    print(f"Sample {person} -> predicted label {label}, conf {conf:.1f}")

print("\nFinal mapping (label -> name):")
for k in sorted(mapping.keys()):
    print(k, "->", mapping[k])

# Save mapping to json for use by app
out = os.path.join(os.path.dirname(trainer), "label_map.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print("\nSaved label_map.json at", out)
