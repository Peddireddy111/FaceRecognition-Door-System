# door_app_advanced.py
# Smart Door: face recognition + voice + logging + Telegram notifications (synchronous)

import os
import time
import csv
import threading
import json
from datetime import datetime

import cv2
import numpy as np
import requests  # used to send Telegram HTTP requests
import pyttsx3   # for voice alerts

# ================== PATHS ==================
BASE = r"D:\FaceRecognitionPython\FaceRecognitionPython"

CASCADE_PATH      = os.path.join(BASE, "haarcascade_frontalface_default.xml")
TRAINER_PATH      = os.path.join(BASE, "trainer.yml")
LABEL_MAP_PATH    = os.path.join(BASE, "label_map.json")
DOOR_CLOSED_PATH  = os.path.join(BASE, "door_closed.png")
DOOR_OPEN_PATH    = os.path.join(BASE, "door_open.png")
INTRUDERS_DIR     = os.path.join(BASE, "intruders")
LOG_CSV           = os.path.join(BASE, "events.csv")

os.makedirs(INTRUDERS_DIR, exist_ok=True)

# ================== TELEGRAM CONFIG (YOUR PROVIDED VALUES) ==================
TELEGRAM_TOKEN = "8406735191:AAEUj_Ai1NHIg_F3KHMbqO0Mwsr03ywYEpo"
CHAT_ID = "1142895444"

def send_telegram_alert(message, image_path=None):
    """
    Synchronous Telegram notification using HTTP API (requests).
    Sends a text message, and optionally sends an image after.
    """
    try:
        # send text
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        res = requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=10)
        # optionally send photo
        if image_path and os.path.exists(image_path):
            photo_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            with open(image_path, "rb") as img:
                requests.post(photo_url, data={"chat_id": CHAT_ID}, files={"photo": img}, timeout=20)
        print("📩 Telegram alert attempt done (check Telegram).")
    except Exception as e:
        print("❌ Telegram send error:", e)

# ================== VOICE ==================
def speak_async(text):
    def _run():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("Voice error:", e)
    threading.Thread(target=_run, daemon=True).start()

# ================== LOGGING ==================
def log_event(name, status, conf=None):
    header = ["timestamp", "name", "status", "confidence"]
    exists = os.path.exists(LOG_CSV)
    try:
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(header)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             name, status, f"{conf:.1f}" if isinstance(conf, (int,float)) else ""])
    except Exception as e:
        print("Logging error:", e)

# ================== HELPERS ==================
def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")
    return img

def validate_required_files():
    required = [CASCADE_PATH, TRAINER_PATH, LABEL_MAP_PATH, DOOR_CLOSED_PATH, DOOR_OPEN_PATH]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Missing required files:")
        for m in missing:
            print(" -", m)
        raise SystemExit("Place the missing files in the folder and re-run.")

# ================== MAIN ==================
if __name__ == "__main__":
    print("Starting Smart Door (advanced) with Telegram alerts...")
    validate_required_files()

    # load cascade and model
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise SystemExit("Failed to load Haar Cascade.")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        raise SystemExit("cv2.face not available. Install opencv-contrib-python.") from e
    recognizer.read(TRAINER_PATH)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    # normalize keys to int if they are strings
    try:
        label_map = {int(k): v for k, v in label_map.items()}
    except Exception:
        pass

    # door images
    door_closed = safe_imread(DOOR_CLOSED_PATH)
    door_open   = safe_imread(DOOR_OPEN_PATH)
    door_open   = cv2.resize(door_open, (door_closed.shape[1], door_closed.shape[0]))

    # camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera (index 0).")

    # parameters (tweak as needed)
    CONF_THRESHOLD = 75.0    # lower -> stricter (65-85 typical)
    VOTE_FRAMES     = 5      # must be odd; voting window
    OPEN_HOLD       = 4.0    # seconds to keep door open after confirm
    ANIM_STEPS      = 8      # smoothing steps

    votes = []               # sliding window of tags
    current_name = None
    open_until = 0.0
    last_spoken = None
    anim_step = 0
    last_intruder_save = 0.0

    def blend_door(ratio):
        return cv2.addWeighted(door_open, ratio, door_closed, 1.0 - ratio, 0)

    print("✅ System active. Press 'q' to quit.")
    speak_async("Smart door system activated")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        tag_this_frame = "NO_FACE"
        best_conf = None
        best_name = None

        # annotate detected faces and set tag for this frame
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            try:
                roi = cv2.resize(roi, (200,200))
                label, conf = recognizer.predict(roi)
            except Exception:
                label, conf = -1, 9999.0

            if conf < CONF_THRESHOLD and label in label_map:
                tag_this_frame = label_map[label]
                best_name = tag_this_frame
                best_conf = conf
                color = (0,255,0)
                txt = f"{best_name} ({conf:.1f})"
            else:
                tag_this_frame = "UNKNOWN"
                best_name = None
                best_conf = conf
                color = (0,0,255)
                txt = f"Unknown ({conf:.1f})"

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, txt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # voting
        votes.append(tag_this_frame)
        if len(votes) > VOTE_FRAMES:
            votes.pop(0)
        majority = max(set(votes), key=votes.count) if votes else "NO_FACE"
        maj_count = votes.count(majority)
        now = time.time()

        # decision logic
        if majority not in ("NO_FACE", "UNKNOWN") and maj_count >= (VOTE_FRAMES//2 + 1):
            # confirmed known person
            if current_name != majority:
                current_name = majority
                open_until = now + OPEN_HOLD
                log_event(current_name, "access_granted", best_conf if best_conf else 0.0)
                if last_spoken != current_name:
                    speak_async(f"Access granted. Welcome {current_name}")
                    last_spoken = current_name
            target_open = 1.0
        else:
            # handle unknown bursts
            if majority == "UNKNOWN" and maj_count >= (VOTE_FRAMES//2 + 1):
                # throttle to avoid flooding
                if now - last_intruder_save > 2.0:
                    snap_name = f"intruder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    snap_path = os.path.join(INTRUDERS_DIR, snap_name)
                    cv2.imwrite(snap_path, frame)
                    log_event("Unknown", "access_denied", best_conf if best_conf else 999.0)

                    # send Telegram notification (synchronously)
                    try:
                        send_telegram_alert("🚨 Alert! Unknown person detected at your door.", snap_path)
                    except Exception as e:
                        print("Telegram send failed:", e)

                    # voice
                    if last_spoken != "UNKNOWN":
                        speak_async("Warning. Unknown person detected. Access denied.")
                        last_spoken = "UNKNOWN"

                    last_intruder_save = now
            # auto close logic
            if now > open_until:
                current_name = None
            target_open = 1.0 if current_name else 0.0

        # animate door
        if target_open > 0 and anim_step < ANIM_STEPS:
            anim_step += 1
        elif target_open == 0 and anim_step > 0:
            anim_step -= 1
        open_ratio = anim_step / float(ANIM_STEPS) if ANIM_STEPS > 0 else (1.0 if target_open else 0.0)
        door_view = blend_door(open_ratio)

        if current_name:
            cv2.putText(door_view, f"Door Unlocked - {current_name}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        else:
            cv2.putText(door_view, "Door Locked", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # display combined
        cam_r = cv2.resize(frame, (640,480))
        door_r = cv2.resize(door_view, (640,480))
        combo = np.hstack([cam_r, door_r])
        cv2.imshow("🚪 Smart Door (Advanced)", combo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak_async("System shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")
