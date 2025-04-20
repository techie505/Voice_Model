import tensorflow as tf
import numpy as np
import librosa
import os

# Load trained model
model_path = "models/mood_classifier_updated.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ ERROR: Model file '{model_path}' not found.")

model = tf.keras.models.load_model(model_path)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# ✅ Use correct feature extraction method (same as train.py)
def extract_mel_spectrogram(file_path, max_pad_len=220):
    audio, sr = librosa.load(file_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)  # ✅ Fix shape
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec.shape[1] < max_pad_len:
        pad_width = max_pad_len - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec = mel_spec[:, :max_pad_len]

    return mel_spec

# ✅ Predict mood from recorded voice
def predict_mood(file_path="recordings/live_input.wav"):
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Audio file '{file_path}' not found.")
        return

    features = extract_mel_spectrogram(file_path).reshape(1, 64, 220, 1)  # ✅ Fix shape
    prediction = model.predict(features)
    mood = emotion_labels[np.argmax(prediction)]
    print(f"🎯 Predicted Mood: {mood}")

# Run Prediction
predict_mood()
