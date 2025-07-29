import cv2

class PersonDetector:
    def __init__(self):
        print("[INFO] Loading person detection model...")
        proto = cv2.data.haarcascades + "haarcascade_fullbody.xml"
        self.person_cascade = cv2.CascadeClassifier(proto)

    def detect_people(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = self.person_cascade.detectMultiScale(gray, 1.1, 3)

        people_count = 0
        for (x, y, w, h) in bodies:
            people_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame, people_count
