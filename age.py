"""import cv2
import numpy as np

# Model files
model_architecture = 'age_deploy.prototxt'
model_weights = 'age_net.caffemodel'

# Load the model
age_net = cv2.dnn.readNetFromCaffe(model_architecture, model_weights)

# Age groups the model predicts
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.426337, 87.768914, 114.895847),
                                     swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        label = f"Age: {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Age Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""
import cv2

# Age categories provided by the original model
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Map to your custom categories
def map_age_to_group(age_label):
    age_ranges = {
        'Child': ['(0-2)', '(4-6)', '(8-12)', '(15-20)'],
        'Adult': ['(25-32)', '(38-43)', '(48-53)'],
        'Senior': ['(60-100)']
    }
    for group, ranges in age_ranges.items():
        if age_label in ranges:
            return group
    return "Unknown"

# Load the model
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")

# Load image from webcam or file
cap = cv2.VideoCapture(0)  # 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocessing
    blob = cv2.dnn.blobFromImage(
        image=cv2.resize(frame, (227, 227)),
        scalefactor=1.0,
        size=(227, 227),
        mean=(78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_index = age_preds[0].argmax()
    predicted_age = AGE_BUCKETS[age_index]
    age_group = map_age_to_group(predicted_age)

    # Display result
    label = f"Age Group: {age_group} ({predicted_age})"
    cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Age Classification", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
