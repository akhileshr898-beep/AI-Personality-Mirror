import cv2
from insightface.app import FaceAnalysis

# Initialize face detector
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 uses CPU

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Face {face.det_score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("AI Personality Mirror - Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:   # Q or ESC
        break

cap.release()
cv2.destroyAllWindows()