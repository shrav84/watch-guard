import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)


MIN_FACE_SIZE = 80
SKIN_RATIO_THRESH = 0.25
EDGE_DENSITY_THRESH = 0.005
OCCLUSION_SCORE_THRESH = 2
OCC_FRAMES_TO_TRIGGER = 6
VIS_FRAMES_TO_CLEAR = 3

occluded_count = 0
visible_count = 0
show_warning = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    h_frame, w_frame = frame.shape[:2]
    if w_frame > 800:
        scale = 800.0 / w_frame
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

    if len(faces) == 0:
        # No face found -> reset counters (or could treat as "not visible")
        occluded_count = 0
        visible_count = 0
        show_warning = False
        cv2.putText(frame, "⚠ Face not clearly visible! Please uncover your face", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)
    else:
       
       
        faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
        x, y, w, h = faces[0]

        # draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 255, 180), 2)

        # small-face check
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            cv2.putText(frame, "Face too small / far", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        face_color = frame[y:y+h, x:x+w]
        face_gray  = gray[y:y+h, x:x+w]

        # 1) Skin-color ratio (YCrCb method)
        if face_color.size == 0:
            skin_ratio = 0.0
        else:
            face_ycrcb = cv2.cvtColor(face_color, cv2.COLOR_BGR2YCR_CB)
            lower = np.array([0, 135, 85], dtype=np.uint8)
            upper = np.array([255, 180, 135], dtype=np.uint8)
            skin_mask = cv2.inRange(face_ycrcb, lower, upper)
            skin_pixels = cv2.countNonZero(skin_mask)
            skin_ratio = float(skin_pixels) / float(max(1, w*h))
        cv2.putText(frame, f"Skin:{skin_ratio:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # 2) Edge density
        edges = cv2.Canny(face_gray, 100, 200)
        edge_density = float(cv2.countNonZero(edges)) / float(max(1, w*h))
        cv2.putText(frame, f"Edges:{edge_density:.3f}", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # 3) Eye detection inside face ROI
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15,15))
        eyes_found = len(eyes)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255,180,180), 1)
        cv2.putText(frame, f"Eyes:{eyes_found}", (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # Occlusion scoring (simple voting)
        score = 0
        reasons = []
        if eyes_found == 0:
            score += 1
            reasons.append("eyes not found")
        if skin_ratio < SKIN_RATIO_THRESH:
            score += 1
            reasons.append("low skin ratio")
        if edge_density < EDGE_DENSITY_THRESH:
            score += 1
            reasons.append("low edge density")

        occluded = (score >= OCCLUSION_SCORE_THRESH)

        # smoothing across frames (avoid flicker)
        if occluded:
            occluded_count += 1
            visible_count = 0
        else:
            visible_count += 1
            occluded_count = 0

        if occluded_count >= OCC_FRAMES_TO_TRIGGER:
            show_warning = True
        if visible_count >= VIS_FRAMES_TO_CLEAR:
            show_warning = False

        # Display status and reasons
        if show_warning:
            reason_text = ", ".join(reasons) if reasons else "occluded"
            cv2.putText(frame, "⚠ FACE COVERED, Please uncover your face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.putText(frame, reason_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.putText(frame, "No Violation,Continue", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)

    cv2.imshow("Face coverage check", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()