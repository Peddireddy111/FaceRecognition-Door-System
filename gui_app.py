import os
import cv2
import json
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog

# ========== PATH CONFIGURATION ==========
BASE_DIR = r"D:\FaceRecognitionPython\FaceRecognitionPython"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer.yml")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")
DOOR_SCRIPT = os.path.join(BASE_DIR, "door_app_advanced.py")
EVENTS_CSV = os.path.join(BASE_DIR, "events.csv")
INTRUDERS_DIR = os.path.join(BASE_DIR, "intruders")

# Ensure folders
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(INTRUDERS_DIR, exist_ok=True)

# ========== MAIN GUI WINDOW ==========
root = tk.Tk()
root.title("Smart Face Recognition Door System")
root.geometry("640x520")

status_var = tk.StringVar(value="Ready")

def set_status(msg):
    status_var.set(msg)
    root.update_idletasks()

# ========== ADD PERSON (ASK NAME → CAPTURE 100 → RETURN) ==========
def action_add_person():
    """Ask name in main thread, capture in a background thread."""
    name = simpledialog.askstring("Add Person", "Enter person's name:")
    if not name:
        set_status("❌ Cancelled (No name entered)")
        return
    threading.Thread(target=capture_faces, args=(name.strip(),), daemon=True).start()

def capture_faces(name):
    person_folder = os.path.join(DATASET_DIR, name)
    os.makedirs(person_folder, exist_ok=True)

    # sanity check
    if not os.path.exists(CASCADE_PATH):
        messagebox.showerror("Missing file", f"Haar Cascade not found:\n{CASCADE_PATH}")
        return

    set_status(f"📸 Capturing images for {name}... (Press 'q' to stop)")
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    count = 0

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            set_status("❌ Camera not detected!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite(os.path.join(person_folder, f"{name}_{count}.jpg"), roi)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"{name}: {count}/100", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Capturing Faces (Press 'q' to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    set_status(f"✅ Done capturing {count} images for {name}")

# ========== TRAIN MODEL (LBPH) ==========
def action_train_model():
    threading.Thread(target=_train_worker, daemon=True).start()

def _train_worker():
    import numpy as np
    if not os.path.exists(CASCADE_PATH):
        messagebox.showerror("Missing file", f"Haar Cascade not found:\n{CASCADE_PATH}")
        return

    faces, labels = [], []
    label_map, current_id = {}, 0
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # collect images
    for person in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir) or person.lower() == "intruders":
            continue
        current_id += 1
        label_map[current_id] = person

        for fn in os.listdir(person_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(person_dir, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Optional: detect face in the cropped img (tolerant)
            rects = face_cascade.detectMultiScale(img, 1.2, 5)
            if len(rects) == 0:
                faces.append(cv2.resize(img, (200,200))); labels.append(current_id)
            else:
                for (x,y,w,h) in rects:
                    faces.append(cv2.resize(img[y:y+h, x:x+w], (200,200)))
                    labels.append(current_id)

    if not faces:
        messagebox.showwarning("Training", "No faces found. Add a person first.")
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        messagebox.showerror("OpenCV", "cv2.face is missing. Install:\n\npip install opencv-contrib-python")
        return

    set_status("🧠 Training… please wait")
    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_PATH)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    set_status("✅ Model trained. Saved trainer.yml + label_map.json")

# ========== START DOOR APP (ADVANCED) ==========
def action_start_door():
    if not os.path.exists(TRAINER_PATH) or not os.path.exists(LABEL_MAP_PATH):
        messagebox.showwarning("Train first", "trainer.yml or label_map.json missing. Train the model first.")
        return
    if not os.path.exists(DOOR_SCRIPT):
        messagebox.showerror("Missing", f"Door script not found:\n{DOOR_SCRIPT}")
        return
    set_status("🚪 Starting Door App… (Close it with 'q')")
    threading.Thread(target=lambda: os.system(f'python "{DOOR_SCRIPT}"'), daemon=True).start()

# ========== VIEW LOGS & INTRUDERS ==========
def action_view_logs():
    if os.path.exists(EVENTS_CSV):
        os.startfile(EVENTS_CSV)
        set_status("Opened events.csv")
    else:
        messagebox.showinfo("Logs", "No events.csv found yet (run Door App first).")

def action_open_intruders():
    os.makedirs(INTRUDERS_DIR, exist_ok=True)
    os.startfile(INTRUDERS_DIR)

# ========== VOICE TEST ==========
def action_voice_test():
    def _speak():
        try:
            import pyttsx3
        except Exception:
            messagebox.showinfo("Voice", "pyttsx3 not installed. Install with:\n\npip install pyttsx3")
            return
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.say("Voice test successful. Smart door system ready.")
            engine.runAndWait()
            set_status("🔊 Voice test played.")
        except Exception as e:
            messagebox.showerror("Voice Error", str(e))
    threading.Thread(target=_speak, daemon=True).start()

# ========== UI LAYOUT ==========
header = tk.Label(root, text="Smart Face Recognition Door — Control Panel",
                  font=("Segoe UI", 16, "bold"))
header.pack(pady=(14, 8))

btn_pad = 6
tk.Button(root, text="➕ Add Person (Ask Name & Capture 100)", width=40,
          command=action_add_person).pack(pady=btn_pad)

tk.Button(root, text="🧠 Train Model", width=40,
          command=action_train_model).pack(pady=btn_pad)

tk.Button(root, text="🚪 Start Door App (Advanced)", width=40,
          command=action_start_door).pack(pady=btn_pad)

tk.Button(root, text="📜 View Access Logs (events.csv)", width=40,
          command=action_view_logs).pack(pady=btn_pad)

tk.Button(root, text="🗂 Open Intruders Folder", width=40,
          command=action_open_intruders).pack(pady=btn_pad)

tk.Button(root, text="🔊 Voice Test", width=40,
          command=action_voice_test).pack(pady=btn_pad)

sep = tk.Frame(root, height=2, bd=1, relief="sunken")
sep.pack(fill="x", padx=12, pady=(10, 8))

status_lbl = tk.Label(root, textvariable=status_var, anchor="w", fg="blue")
status_lbl.pack(fill="x", padx=12, pady=(0, 10))

# ========== START APP ==========
set_status("Ready")
root.mainloop()
