import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.putText(
        frame,
        "AI Personality Mirror",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Personality Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()