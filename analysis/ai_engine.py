import cv2
import os
import time
import numpy as np

from insightface.app import FaceAnalysis
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from analysis.voice_engine import VoiceEngine



class AIEngine:


    def __init__(self):

        # =============================
        # InsightFace
        # =============================

        self.face = FaceAnalysis(
            name="buffalo_l"
        )

        self.face.prepare(
            ctx_id=-1,
            det_size=(640,640)
        )


        # =============================
        # MediaPipe Face Landmarker
        # =============================

        model_path = os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            "models",
            "face_landmarker.task"
        )


        base_options = python.BaseOptions(
            model_asset_path=model_path
        )


        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1
        )


        self.face_landmarker = (
            vision.FaceLandmarker
            .create_from_options(options)
        )



        # =============================
        # YOLO Phone Detection
        # =============================

        yolo_path = os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            "models",
            "yolov8n.pt"
        )


        self.yolo = YOLO(
            yolo_path
        )



        # =============================
        # Voice Engine
        # =============================

        self.voice = VoiceEngine()



    # =================================
    # Face Landmarks
    # =================================

    def get_landmarks(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        result = self.face_landmarker.detect(
            image
        )


        if not result.face_landmarks:
            return None


        return result.face_landmarks[0]



    # =================================
    # Eye Contact
    # =================================

    def calculate_eye_contact(self, frame):

        lm = self.get_landmarks(frame)


        if lm is None:
            return 0


        left = lm[468]
        right = lm[473]


        distance = abs(
            left.x-right.x
        )


        score = int(
            max(
                0,
                min(
                    100,
                    100 -
                    abs(distance-0.05)*800
                )
            )
        )


        return score



    # =================================
    # Smile Detection
    # =================================

    def detect_smile(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        result = self.face_landmarker.detect(
            image
        )


        if not result.face_blendshapes:

            return 0



        score = 0


        for item in result.face_blendshapes[0]:

            if item.category_name=="mouthSmileLeft":

                score += item.score


            if item.category_name=="mouthSmileRight":

                score += item.score



        return int(
            min(
                score*50,
                100
            )
        )



    # =================================
    # Head Pose
    # =================================

    def detect_head_pose(self, frame):

        lm = self.get_landmarks(frame)


        if lm is None:

            return "NO FACE"



        nose = lm[1]

        left = lm[33]

        right = lm[263]


        center = (
            left.x+
            right.x
        )/2



        if nose.x < center-0.03:

            return "LEFT"


        if nose.x > center+0.03:

            return "RIGHT"


        if nose.y > 0.60:

            return "DOWN"


        return "STRAIGHT"



    # =================================
    # Posture
    # =================================

    def detect_posture(self, frame):

        lm = self.get_landmarks(frame)


        if lm is None:

            return "NO FACE",0



        left = lm[234]

        right = lm[454]


        center = (
            left.x+
            right.x
        )/2


        deviation = abs(
            0.5-center
        )


        if deviation <0.03:

            return "GOOD",95


        elif deviation<0.08:

            return "OK",70


        return "BAD",40



    # =================================
    # Phone Detection
    # =================================

    def detect_phone(self, frame):

        results = self.yolo(
            frame,
            verbose=False
        )


        detected=False


        for result in results:

            for box in result.boxes:

                cls=int(
                    box.cls[0]
                )


                conf=float(
                    box.conf[0]
                )


                # Mobile phone class

                if cls==67 and conf>0.45:


                    detected=True


                    x1,y1,x2,y2=map(
                        int,
                        box.xyxy[0]
                    )


                    cv2.rectangle(
                        frame,
                        (x1,y1),
                        (x2,y2),
                        (0,0,255),
                        2
                    )


                    cv2.putText(
                        frame,
                        "PHONE",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2
                    )


        return detected
        # =================================
    # Dashboard Graphics
    # =================================

    def draw_bar(
        self,
        frame,
        name,
        value,
        y,
        color
    ):

        cv2.putText(
            frame,
            f"{name}",
            (25,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230,230,230),
            1
        )


        # background bar

        cv2.rectangle(
            frame,
            (120,y-12),
            (280,y),
            (70,70,70),
            -1
        )


        length=int(
            value*1.6
        )


        cv2.rectangle(
            frame,
            (120,y-12),
            (120+length,y),
            color,
            -1
        )


        cv2.putText(
            frame,
            f"{value}%",
            (285,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255,255,255),
            1
        )



    # =================================
    # Main AI Pipeline
    # =================================

    def process(self, frame):


        start=time.time()



        # -----------------------------
        # Face Detection
        # -----------------------------

        faces=self.face.get(frame)


        face_detected=False



        for face in faces:


            face_detected=True


            x1,y1,x2,y2=map(
                int,
                face.bbox
            )


            # keep face visible

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )



        # -----------------------------
        # AI Features
        # -----------------------------


        eye=self.calculate_eye_contact(
            frame
        )


        smile=self.detect_smile(
            frame
        )


        pose=self.detect_head_pose(
            frame
        )


        posture,posture_score=self.detect_posture(
            frame
        )


        phone=self.detect_phone(
            frame
        )


        # Voice

        voice_data=self.voice.analyze_voice()


        voice_score=voice_data["score"]



        # -----------------------------
        # Final Score
        # -----------------------------


        final_score=int(
            (
                eye+
                smile+
                posture_score+
                voice_score
            )/4
        )


        if phone:

            final_score-=10


        final_score=max(
            0,
            final_score
        )



        # -----------------------------
        # Glass Dashboard
        # -----------------------------


        overlay=frame.copy()



        # small dashboard

        cv2.rectangle(
            overlay,
            (15,15),
            (330,330),
            (20,20,20),
            -1
        )


        frame=cv2.addWeighted(
            overlay,
            0.75,
            frame,
            0.25,
            0
        )



        # Title

        cv2.putText(
            frame,
            "AI PERSONALITY MIRROR",
            (25,45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,255,255),
            2
        )



        # Status

        cv2.putText(
            frame,
            f"FACE : {'ACTIVE' if face_detected else 'NO FACE'}",
            (25,75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0,255,0),
            1
        )



        self.draw_bar(
            frame,
            "Eye",
            eye,
            110,
            (0,255,255)
        )


        self.draw_bar(
            frame,
            "Smile",
            smile,
            145,
            (255,255,0)
        )


        self.draw_bar(
            frame,
            "Posture",
            posture_score,
            180,
            (0,255,0)
        )


        self.draw_bar(
            frame,
            "Voice",
            voice_score,
            215,
            (255,0,255)
        )



        # Text information


        cv2.putText(
            frame,
            f"HEAD : {pose}",
            (25,250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255,255,255),
            1
        )


        cv2.putText(
            frame,
            f"PHONE : {'WARNING' if phone else 'SAFE'}",
            (25,275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0,0,255) if phone else (0,255,0),
            1
        )


        cv2.putText(
            frame,
            f"CONFIDENCE : {final_score}%",
            (25,305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255,255,255),
            2
        )



        # FPS

        fps=int(
            1/(time.time()-start)
        )


        cv2.putText(
            frame,
            f"FPS:{fps}",
            (260,20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255,255,255),
            1
        )



        return frame