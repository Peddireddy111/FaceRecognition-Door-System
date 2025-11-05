import cv2
import os

# Create a directory for datasets if not exists
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Ask for person's name
name = input("Enter person's name: ")
person_dir = os.path.join(dataset_dir, name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

# Load the face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)
count = 0

print("📸 Capturing 50 face samples. Look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        file_name = os.path.join(person_dir, f"{name}_{count}.jpg")
        cv2.imwrite(file_name, face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capturing Faces", frame)

    if cv2.waitKey(1) == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Dataset created successfully for {name} with {count} images.")
