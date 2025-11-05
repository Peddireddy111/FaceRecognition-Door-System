import cv2
import os
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Load trained model and names
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

dataset_path = "dataset"
names = os.listdir(dataset_path)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create window
root = tk.Tk()
root.title("Smart Door Lock System")
root.geometry("800x600")
root.configure(bg="#222")

# Door images
door_closed = Image.open("door_closed.jpg")
door_open = Image.open("door_open.jpg")

door_closed = door_closed.resize((300, 400))
door_open = door_open.resize((300, 400))

door_closed_img = ImageTk.PhotoImage(door_closed)
door_open_img = ImageTk.PhotoImage(door_open)

door_label = Label(root, image=door_closed_img, bg="#222")
door_label.pack(pady=20)

status_label = Label(root, text="Door Locked 🔒", fg="red", bg="#222", font=("Arial", 20))
status_label.pack()

# Start webcam
cap = cv2.VideoCapture(0)

def recognize_face():
    ret, frame = cap.read()
    if not ret:
        root.after(10, recognize_face)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detected = False
    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 70:
            name = names[id_-1] if id_-1 < len(names) else "Unknown"
            detected = True
            status_label.config(text=f"Welcome {name} 😀\nDoor Unlocked 🔓", fg="green")
            door_label.config(image=door_open_img)
        else:
            status_label.config(text="Face Not Recognized ❌\nDoor Locked 🔒", fg="red")
            door_label.config(image=door_closed_img)

    if not detected:
        status_label.config(text="No Face Detected 👀", fg="yellow")
        door_label.config(image=door_closed_img)

    root.after(100, recognize_face)

root.after(100, recognize_face)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
