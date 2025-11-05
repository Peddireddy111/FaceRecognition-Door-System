# door_app_secure.py
import cv2, os, time, json, numpy as np
from datetime import datetime

BASE = r"D:\FaceRecognitionPython"
TRAINER = os.path.join(BASE, "trainer.yml")
MAP_FILE = os.path.join(BASE, "label_map.json")
CASCADE = r"D:\FaceRecognitionPython\FaceRecognitionPython\haarcascade_frontalface_default.xml"
DOOR_CLOSED = os.path.join(BASE, "door_closed.png")
DOOR_OPEN = os.path.join(BASE, "door_open.png")
INTRUDER_DIR = os.path.join(BASE, "intruders")
os.makedirs(INTRUDER_DIR, exist_ok=True)

# parameters
CONF_THRESHOLD = 90.0   # try 60-70 depending on camera
VOTE_FRAMES = 5        # number of consecutive frames to confirm identity
OPEN_HOLD = 4          # seconds to keep open

# load images
door_closed = cv2.imread(DOOR_CLOSED)
door_open = cv2.imread(DOOR_OPEN)
if door_closed is None or door_open is None:
    print("Door images missing"); raise SystemExit

# load model and mapping
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read(TRAINER)
if not os.path.exists(MAP_FILE):
    print("label_map.json missing. Please run train_and_map.py"); raise SystemExit

with open(MAP_FILE, "r", encoding="utf-8") as f:
    label_map = json.load(f)   # label (string) -> name

# face detector
face_cascade = cv2.CascadeClassifier(CASCADE)
cap = cv2.VideoCapture(0)

# buffers for votes
votes = []
last_open_time = 0
current_name = None

print("Starting secure door app. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    detected_this_frame = None
    detected_conf = None

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        try:
            label,conf = rec.predict(roi)
        except Exception:
            label,conf = -1,999
        # LBPH returns distance; lower is better. Use conf directly here.
        # Decide name only if conf < threshold
        print(f"Detected label={label}, conf={conf:.2f}")
        if conf < CONF_THRESHOLD and str(label) in label_map:
            name = label_map[str(label)]
            detected_this_frame = name
            detected_conf = conf
            cv2.putText(frame, f"{name} ({conf:.1f})", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            detected_this_frame = "UNKNOWN"
            detected_conf = conf
            cv2.putText(frame, f"UNKNOWN ({conf:.1f})", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # voting: push current detection (or None) into buffer
    if detected_this_frame:
        votes.append(detected_this_frame)
    else:
        votes.append("NO_FACE")

    if len(votes) > VOTE_FRAMES:
        votes.pop(0)

    # decide majority
    majority = max(set(votes), key=votes.count)
    # if majority is a known name -> open
    if majority != "NO_FACE" and majority != "UNKNOWN" and votes.count(majority) >= VOTE_FRAMES//2 + 1:
        if current_name != majority:
            print(f"[INFO] Confirmed {majority} by votes -> opening door")
        current_name = majority
        last_open_time = time.time()
        door_display = door_open.copy()
        cv2.putText(door_display, f"Welcome {current_name}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)
    else:
        # if majority unknown (or not enough votes) keep closed
        door_display = door_closed.copy()
        current_name = None

    # auto-close after hold time
    if current_name and (time.time() - last_open_time > OPEN_HOLD):
        print("[INFO] Auto-closing door")
        current_name = None
        votes = []

    # if unknown detected many times, save snapshot
    if votes.count("UNKNOWN") >= 3:
        fname = os.path.join(INTRUDER_DIR, f"intruder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(fname, frame)
        print("[ALERT] Saved intruder:", fname)
        votes = []  # reset to avoid saving repeatedly

    # show combined
    combined = np.hstack((cv2.resize(frame,(480,360)), cv2.resize(door_display,(480,360))))
    cv2.imshow("Secure Door", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
