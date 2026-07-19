import cv2
from ultralytics import YOLO
import time

# Load Better YOLO Model
model = YOLO("yolov8s.pt")

# Open Webcam
cap = cv2.VideoCapture(0)

# Set Camera Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to access webcam")
        break

    # Detection
    results = model(
        frame,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Count Objects
    num_objects = len(results[0].boxes)

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display Object Count
    cv2.putText(
        annotated_frame,
        f"Objects Detected: {num_objects}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "CodeAlpha Object Detection & Tracking",
        annotated_frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
