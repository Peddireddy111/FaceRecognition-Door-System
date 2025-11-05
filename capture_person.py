import cv2, os

BASE = r"D:\FaceRecognitionPython\FaceRecognitionPython"
CASCADE = os.path.join(BASE, "haarcascade_frontalface_default.xml")
DATASET = os.path.join(BASE, "dataset")

os.makedirs(DATASET, exist_ok=True)

name = input("Enter person name (folder will be dataset\\<name>): ").strip()
save_dir = os.path.join(DATASET, name)
os.makedirs(save_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(CASCADE)
cap = cv2.VideoCapture(0)

count = 0
print("Capturing... look at the camera. Press 'q' to stop early.")
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
        path = os.path.join(save_dir, f"{name}_{count:03d}.jpg")
        cv2.imwrite(path, roi)
        count += 1
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(frame, f"Saved: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.imshow("Capture Dataset", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if count >= 100: break

cap.release()
cv2.destroyAllWindows()
print(f"Done. Saved {count} images in {save_dir}")
