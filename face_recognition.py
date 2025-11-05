import cv2
import os
import numpy as np

# Path setup
dataset_path = "dataset"
trainer_path = "trainer.yml"

# Load face detector and trained recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

# Automatically get names from dataset folder
names = sorted(os.listdir(dataset_path))
print("[DEBUG] Registered people:", names)

# Start webcam
cam = cv2.VideoCapture(0)
print("🎥 Starting face recognition. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Cannot access camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        print(f"[DEBUG] Detected ID: {id_} | Confidence: {confidence:.2f}")

        # Map ID to name (ensure no out of range)
        if 0 < id_ <= len(names) and confidence < 80:
            name = names[id_ - 1]  # Adjust index
            label = f"{name} ({confidence:.1f}%)"
        else:
            label = "Unknown"

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
