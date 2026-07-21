import cv2
from ultralytics import YOLO

# Load YOLOv8 Nano model (downloads automatically the first time)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("AI Personality Mirror", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()