import streamlit as st
import cv2
import av

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

from analysis.ai_engine import AIEngine


st.set_page_config(
    page_title="AI Personality Mirror",
    layout="wide"
)


st.title("🤖 AI Personality Mirror")
st.subheader("Real-Time AI Interview Coach")


engine = AIEngine()



class VideoProcessor(VideoProcessorBase):


    def __init__(self):

        self.engine = engine



    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )


        # AI processing

        processed = self.engine.process(
            img
        )


        return av.VideoFrame.from_ndarray(
            processed,
            format="bgr24"
        )



webrtc_streamer(

    key="ai-mirror",

    video_processor_factory=VideoProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False
    }

)