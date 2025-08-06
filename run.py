from core.vision.detector import PersonDetector
import cv2
import time

def main():
    print("[INFO] Starting camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not access the camera.")
        return

    detector = PersonDetector()
    print("[INFO] Camera started. Press 'q' in the video window to exit.")

    start_time = time.time()
    # TIMEOUT_SECONDS = 30  

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            frame, people_count = detector.detect_people(frame)

           
            cv2.putText(frame, f"People detected: {people_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            
            cv2.imshow("ATM Detection Feed", frame)

            
            key = cv2.waitKey(1) & 0xFF

           
            if key == ord('q'):
                print("[INFO] 'q' pressed. Exiting...")
                break

            
            # if time.time() - start_time > TIMEOUT_SECONDS:
            #     print(f"[INFO] Auto-exiting after {TIMEOUT_SECONDS} seconds.")
            #     break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Program ended.")

if __name__ == "__main__":
    main()

