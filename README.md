# RoboTechArm 🤖🦾
Vision-assisted robotic arm (Final Year Project). YOLOv8 detects fruits; a Flask web app sends commands; Firebase ties them together.  
**Note:** Hardware & Arduino firmware are not included; this repo focuses on the software pipeline.

## 🔗 Demo
- Video: <ADD-YOUR-YOUTUBE-or-DRIVE-LINK>
- Screenshots: see `/assets`

## 🧩 Components
- `src/vision/yolo_fruit_to_firebase.py` — YOLOv8 detection → Firebase RTDB (`Robo_command`, `servoCommand`, `detectionFeed`).
- `src/webapp/app.py` — Flask app for login/signup, send commands, and view history.
- `templates/` — HTML pages (`index`, `dashboard`, `history`, `login`, `signup`).

## 🛠 Tech
- Python, OpenCV, cvzone, Ultralytics YOLOv8  
- Flask, Firebase (pyrebase + firebase-admin)

## 🔐 Configuration (no secrets in Git)
- Place Firebase Admin key at: `secrets/serviceAccount.json` (DO NOT COMMIT).
- Copy `.env.example` to `.env` and set: