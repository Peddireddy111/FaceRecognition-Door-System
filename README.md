# FaceRecognition-Door-System
Smart Face Recognition Door Lock using OpenCV, Voice Alerts, and Telegram Notifications.
# 🔐 Smart Face Lock System  
**Developed by Pavan (GitHub: Peddireddy111)**

A real-time AI-powered **Face Recognition Door Lock System** using **Python, OpenCV**, and **LBPH Face Recognition**.  
This system detects known faces, opens the virtual door with a welcome message, and if an unknown person (intruder) is detected — it **captures their photo and sends a Telegram alert with the image.**

---

## 📸 Demo (Replace With Your Own Images)

| Door Closed | Door Opened |
|-------------|-------------|
| ![Door Closed](<img width="2880" height="1800" alt="Screenshot (54)" src="https://github.com/user-attachments/assets/c540a9df-ce3c-40e0-833b-4793e32113cc" />
) | ![Door Open](<img width="2880" height="1800" alt="Screenshot (53)" src="https://github.com/user-attachments/assets/77bcc5a1-abf6-4183-9df3-ad6e4e1b4b20" />
) |

📌 Create an `images/` folder in your GitHub and place 2 screenshots with names:
- `door_closed_sample.png`
- `door_open_sample.png`

---

## ✅ Features

✔ Real-time Face Detection & Recognition  
✔ Virtual Door Open/Close with Images  
✔ Voice Alerts – "Welcome Pavan" / "Intruder Detected"  
✔ Unknown Face → Saves Image in `/intruders/` Folder  
✔ Sends Telegram Alert + Intruder Photo to Mobile  
✔ Event Logging into `events.csv`  
✔ Offline Model using OpenCV LBPH Algorithm  

---

## 🛠 Technologies Used

| Component      | Technology       |
|----------------|------------------|
| Programming    | Python 3.x       |
| Face Detection | Haar Cascade     |
| Recognition    | LBPH (OpenCV)    |
| Voice Alerts   | pyttsx3          |
| Alerts         | Telegram Bot API |
| Logging        | CSV File         |
| GUI (optional) | Tkinter          |

---

## 📁 Folder Structure

```
SmartFaceLockSystem/
 ├─ door_app_advanced.py         # Main AI Door System
 ├─ create_dataset.py            # Capture face dataset
 ├─ train_model.py               # Train the face model
 ├─ face_recognition.py          # Simple recognizer (optional)
 ├─ haarcascade_frontalface_default.xml
 ├─ trainer.yml                  # Trained face data
 ├─ label_map.json               # ID-Name map for training
 ├─ requirements.txt             # Libraries to install
 ├─ dataset/                     # Images of known people
 ├─ intruders/                   # Auto-saved intruder images
 ├─ events.csv                   # Door access logs
 └─ images/                      # Screenshots for README (optional)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this Repository
```bash
git clone https://github.com/Peddireddy111/Smart-Face-Lock-System.git
cd Smart-Face-Lock-System
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Create Dataset (Capture Face Images)
```bash
python create_dataset.py
```

### 4️⃣ Train the Model
```bash
python train_model.py
```

### 5️⃣ Run Smart Door System
```bash
python door_app_advanced.py
```

---

## 📲 Telegram Alert Setup (for Intruders)

1. Open Telegram → Search **BotFather**  
2. Type `/newbot` → Follow steps → You get a **Bot Token**  
3. Open **@userinfobot** to get your **Chat ID**  
4. Open `door_app_advanced.py` and update:

```python
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
```

---

## 🚀 Future Enhancements

✅ Add OTP via SMS  
✅ Connect to Arduino / Raspberry Pi (real lock)  
✅ Android App for Live Control  
✅ Store data in Firebase / MySQL  
✅ Attendance / Smart Security Dashboard  

---

## 👨‍💻 Author

**Pavan**  
🔗 GitHub: [Peddireddy111](https://github.com/Peddireddy111)

---

## ⭐ Support  

If this project helped you, please ⭐ star this repository on GitHub 😊  

