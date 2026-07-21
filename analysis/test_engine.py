import cv2
from analysis.ai_engine import AIEngine

engine = AIEngine()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = engine.process(frame)

    cv2.imshow("AI Personality Mirror", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()