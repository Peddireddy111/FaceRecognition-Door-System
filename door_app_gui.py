import cv2
import os
import time

# ✅ Paths (Update if needed)
cascade_path = r"D:\FaceRecognitionPython\FaceRecognitionPython\haarcascade_frontalface_default.xml"
trainer_path = r"D:\FaceRecognitionPython\trainer.yml"
door_closed_path = r"D:\FaceRecognitionPython\door_closed.png"
door_open_path = r"D:\FaceRecognitionPython\door_open.png"

# ✅ Load cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# ✅ Load recognizer if exists
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(trainer_path):
    recognizer.read(trainer_path)
else:
    print("⚠️ Trainer file not found, using face detection only.")

# ✅ Load door images
door_closed = cv2.imread(door_closed_path)
door_open = cv2.imread(door_open_path)

if door_closed is None or door_open is None:
    print("❌ Door images not found. Check file paths.")
    exit()

# ✅ Registered names (make sure same as dataset folder names)
names = ["Unknown", "Pavan", "Naveen", "Obul"]

# ✅ Start webcam
cap = cv2.VideoCapture(0)

door_opened = False
last_detected_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not available.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    recognized_name = "Unknown"

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w]) if os.path.exists(trainer_path) else (0, 100)
        
        if conf < 70:
            recognized_name = names[id_] if id_ < len(names) else "Unknown"
            color = (0, 255, 0)
            door_opened = True
            last_detected_time = time.time()
        else:
            recognized_name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # ✅ Door logic
    if door_opened:
        door_display = door_open.copy()
        cv2.putText(door_display, "Door Unlocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        if time.time() - last_detected_time > 5:
            door_opened = False
    else:
        door_display = door_closed.copy()
        cv2.putText(door_display, "Door Locked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    # ✅ Combine video + door view
    frame_resized = cv2.resize(frame, (500, 400))
    door_display_resized = cv2.resize(door_display, (500, 400))
    combined = cv2.hconcat([frame_resized, door_display_resized])

    cv2.imshow("Smart Face Door System", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
