import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox



model = YOLO("yolov8n.pt")


weapon_classes = ["knife", "gun", "pistol", "rifle", "sword"]

# Open webcam
cap = cv2.VideoCapture(0)
weapon_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.6)  # increase conf for better precision

    # Annotate detections on frame
    annotated_frame = results[0].plot()

    # Check if any detected class is in weapon_classes
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label.lower() in weapon_classes:
            weapon_detected = True
            break

    # Show frame in a window
    cv2.imshow("Weapon Detection", annotated_frame)

    if weapon_detected:
        break

   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Show Access Denied if weapon was detected
if weapon_detected:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Access Denied", "Weapon Detected! Access Denied ❌")
