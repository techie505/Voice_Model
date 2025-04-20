import streamlit as st
import requests
import numpy as np
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

st.title("🎙️ Mood Prediction from Voice")

# WebRTC audio recorder
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = []

    def recv(self, frame):
        self.audio_data.append(frame.to_ndarray())
        return frame

webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Save and send audio
if webrtc_ctx.audio_processor:
    if st.button("🎤 Stop & Predict"):
        audio_data = np.concatenate(webrtc_ctx.audio_processor.audio_data, axis=0)
        sf.write("temp_audio.wav", audio_data, 16000)

        with open("temp_audio.wav", "rb") as file:
            files = {"file": file}
            response = requests.post("http://localhost:5000/predict", files=files)

        if response.status_code == 200:
            mood = response.json().get("mood", "Unknown")
            st.success(f"Predicted Mood: {mood} 😊")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
