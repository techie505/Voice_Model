import streamlit as st
import numpy as np
import requests
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

st.title("🎙️ Mood Prediction App")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_array = np.array(frame.to_ndarray())
        self.audio_data.append(audio_array)
        return frame

# WebRTC Recorder
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"video": False, "audio": True},
)

if webrtc_ctx.audio_processor:
    if st.button("🔴 Stop & Analyze"):
        # Convert recorded audio to a single NumPy array
        audio_data = np.concatenate(webrtc_ctx.audio_processor.audio_data, axis=0)

        # Debugging Info
        st.write("🔹 Audio Data Shape:", audio_data.shape if hasattr(audio_data, 'shape') else "No shape")
        st.write("🔹 Audio Data Sample:", audio_data[:10] if isinstance(audio_data, np.ndarray) else "Not a NumPy array")

        # Ensure valid data
        if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]

            # Normalize and convert to int16
            audio_data = (audio_data * 32767).astype(np.int16)

            # Save as WAV
            wav_filename = "temp_audio.wav"
            sf.write(wav_filename, audio_data, 16000, format='WAV', subtype='PCM_16')

            st.success("✅ Audio recorded successfully!")

            # Send file to backend
            with open(wav_filename, "rb") as f:
                files = {"file": f}
                response = requests.post("http://localhost:5000/predict", files=files)

            if response.status_code == 200:
                mood = response.json().get("mood", "Unknown")
                st.write(f"🎭 **Predicted Mood:** {mood}")
            else:
                st.error("⚠️ Error predicting mood.")
        else:
            st.error("⚠️ No valid audio data recorded.")
