import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()

webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()

    gaze.refresh(frame)

    frame = gaze.annotated_frame()

    text = ""

    if gaze.is_blinking():
        text = "Blinking"

    elif gaze.is_right():
        text = "Looking Right"

    elif gaze.is_left():
        text = "Looking Left"

    elif gaze.is_center():
        text = "Looking Center"

    cv2.putText(
        frame,
        text,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Eye Contact Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()