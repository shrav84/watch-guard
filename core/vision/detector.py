import cv2
from ultralytics import YOLO
import mediapipe as mp


class PersonDetector:
    def __init__(self, model_name="yolov8l.pt", use_motion=True, use_face=True,
                 motion_threshold=300, min_area=800, aspect_ratio_range=(0.8, 4.0)):
        print("[INFO] Loading YOLOv8 model...")
        self.model = YOLO(model_name)
        self.person_class_name = "person"

        self.use_motion = use_motion
        self.use_face = use_face
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.aspect_ratio_range = aspect_ratio_range

        if self.use_motion:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        if self.use_face:
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)

    def detect_people(self, frame):
        try:
            results = self.model(frame, verbose=False)[0]
            people_count = 0

            if self.use_motion:
                fg_mask = self.bg_subtractor.apply(frame)

            for box in results.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                label = self.model.names[cls_id]

                if label != self.person_class_name:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                width, height = x2 - x1, y2 - y1
                area = width * height
                aspect_ratio = height / width if width != 0 else 0

                print(f"[DEBUG] Detected: {label} (conf: {conf:.2f}) | AR: {aspect_ratio:.2f}, Area: {area}")

                if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]) or area < self.min_area:
                    print("[FILTER] Skipped: Aspect ratio or area out of range.")
                    continue

                if self.use_motion:
                    motion_region = fg_mask[y1:y2, x1:x2]
                    motion_score = cv2.countNonZero(motion_region)
                    print(f"[DEBUG] Motion score: {motion_score}")
                    if motion_score < self.motion_threshold:
                        print("[FILTER] Skipped: Not enough motion.")
                        continue

                if self.use_face:
                    person_crop = frame[y1:y2, x1:x2]
                    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    results_face = self.face_detection.process(rgb_crop) 

                    if not results_face.detections:
                        print("[FILTER] Skipped: No face detected.")
                        continue 
                
                people_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            print(f"[INFO] People counted: {people_count}")
            return frame, people_count
        
        except Exception as e:
            print(f"[ERROR] Exception during detection: {e}")
            return frame, 0  # Always return a fallback

