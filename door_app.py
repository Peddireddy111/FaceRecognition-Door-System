import cv2
import numpy as np
import os
import json
import pyttsx3
from datetime import datetime

# ✅ Paths
BASE_PATH = r"D:\FaceRecognitionPython\FaceRecognitionPython"
TRAINER_PATH = os.path.join(BASE_PATH, "trainer.yml")
LABEL_MAP_PATH = os.path.join(BASE_PATH, "label_map.json")
CASCADE_PATH = os.path.join(BASE_PATH, "haarcascade_frontalface_default.xml")
DOOR_CLOSED = os.path.join(BASE_PATH, "door_closed.png")
DOOR_OPEN = os.path.join(BASE_PATH, "door_open.png")
INTRUDER_PATH = os.path.join(BASE_PATH, "intruders")
LOG_FILE = os.path.join(BASE_PATH, "events.csv")

os.makedirs(INTRUDER_PATH, exist_ok=True)

# ✅ Voice Engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ✅ Logging function
def log_event(name, status, confidence=0):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()},{name},{status},{confidence}\n")

# ✅ Load Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ✅ Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)

# ✅ Load name mapping
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

# ✅ Load door images
door_closed_img = cv2.imread(DOOR_CLOSED)
door_open_img = cv2.imread(DOOR_OPEN)

# ✅ Start Webcam
camera = cv2.VideoCapture(0)
print("✅ Smart Door Lock System Running... Press 'q' to exit.")
speak("Smart Door Lock System Activated")

last_status = ""  # to avoid repeating voice

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Camera not detected!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    door_display = door_closed_img.copy()  # Default: Closed door

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face_roi)

        if confidence < 70:  # ✅ Known Person
            name = label_map.get(str(label), "Unknown")
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            door_display = door_open_img.copy()
            cv2.putText(door_display, f"Door Opened for {name}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if last_status != name:
                speak(f"Access Granted. Welcome {name}")
                log_event(name, "Access Granted", confidence)
                last_status = name

        else:  # ❌ Unknown = Intruder
            cv2.putText(frame, "❌ Intruder!", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            door_display = door_closed_img.copy()

            intruder_file = os.path.join(INTRUDER_PATH, f"intruder_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            cv2.imwrite(intruder_file, frame)

            if last_status != "intruder":
                speak("Warning. Intruder detected. Access denied.")
                log_event("Unknown", "Access Denied", confidence)
                last_status = "intruder"

    frame_resized = cv2.resize(frame, (480, 360))
    door_resized = cv2.resize(door_display, (480, 360))
    combined = np.hstack((frame_resized, door_resized))

    cv2.imshow("🔐 Smart Door Lock System with Voice & Logging", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak("System shutting down")
        break

camera.release()
cv2.destroyAllWindows()
