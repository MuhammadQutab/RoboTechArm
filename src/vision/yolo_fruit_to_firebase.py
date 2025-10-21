import time
from ultralytics import YOLO
import cvzone
import cv2
import pyrebase

# Firebase configuration
config = {
    "apiKey": "AIzaSyB4H-5X7ISCY-7Zc2mwAXJgAgWxkw7cizM",
    "authDomain": "hand-project2.firebaseapp.com",
    "databaseURL": "https://hand-project2-default-rtdb.firebaseio.com/",
    "storageBucket": "hand-project2.firebasestorage.app"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(config)
database = firebase.database()

# Initialize Firebase paths
database.child("right").set("no")
database.child("servoCommand").set(0)
database.child("detectionFeed").set([])

# Initialize laptop camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open laptop camera.")
    exit()

# OpenCV window for detection
cv2.namedWindow("Detected Fruits", cv2.WINDOW_AUTOSIZE)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with a YOLOv8 model trained on fruits if available

# Specify fruit classes
fruit_classes = ["apple", "banana", "orange", "grape", "pineapple", "strawberry", "watermelon"]  # Adjust as needed
classnames = model.names  # Get YOLO model's class names

# Main detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (640, 480))

    # Run YOLOv8 model on the frame
    results = model(frame, stream=True)

    # Detection processing
    detected_object = False
    detected_classes = []  # List to store detected fruit classes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0] * 100  # Confidence in percentage
            Class = int(box.cls[0])

            # If the detected object is a fruit
            if classnames[Class] in fruit_classes:
                detected_object = True
                detected_classes.append(classnames[Class])  # Add detected fruit class to the list

                # Draw bounding box and label
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green bounding box
                label = f"{classnames[Class]}: {confidence:.2f}%"  # Properly formatted confidence
                cvzone.putTextRect(frame, label, [x1, y1 - 10], scale=1.5, thickness=2, offset=5)

    # Send detected classes to the detectionFeed field in Firebase
    database.child("detectionFeed").set(detected_classes)

    # Check for matching Robo_command
    robo_command = database.child("Robo_command").get().val()
    if robo_command in detected_classes:
        if robo_command == "banana":
            database.child("servoCommand").set(1)
        elif robo_command == "apple":
            database.child("servoCommand").set(2)
        elif robo_command == "orange":
            database.child("servoCommand").set(3)
        else:
            database.child("servoCommand").set(0)  # Default value if no match
    else:
        database.child("servoCommand").set(0)  # No matching command

    # If no fruit is detected, update Firebase
    if not detected_object:
        database.child("right").set("no")
    else:
        database.child("right").set("yes")

    # Show frame with bounding boxes and labels
    cv2.imshow("Detected Fruits", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close OpenCV window
cap.release()
cv2.destroyAllWindows()