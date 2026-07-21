import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]

        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        if emotion == "happy":
            cv2.putText(
                frame,
                "😊 SMILING",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )

    except Exception as e:
        print(e)

    cv2.imshow("Smile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
