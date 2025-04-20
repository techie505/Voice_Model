# Voice_Model
# 🎤 Voice-Based Mood Prediction

A machine learning-powered web application that predicts a user’s mood based on their voice input using real-time audio analysis and deep learning.

 🚀 Features

- 🎧 Voice Recording in-browser
- 📊 Real-time mood prediction (Happy, Sad, Angry, etc.)
- 🧠 Trained model using the CREMA-D dataset
- 🎨 Beautiful UI with React + Tailwind CSS
- 🔊 Audio Preprocessing using Librosa
- 🧪 Sentiment model optimized for accuracy

🗂️ Project Structure

```
Voice_Model/
│
├── backend/
│   ├── predict.py
│   ├── train.py
│   └── record.py
│
├── frontend/
│   ├── app_frontend.py
│   ├── components/
│   └── assets/
│
├── models/
│   └── mood_classifier.h5
│
├── dataset/
│   └── CREMA-D/
│
├── recordings/
├── mood_env/
├── README.md
└── .gitignore
```

## 💠 Installation

```bash
git clone https://github.com/techie505/Voice_Model.git
cd Voice_Model
```

### 🐍 Backend Setup

```bash
cd backend
python -m venv mood_env
source mood_env/Scripts/activate  # or `source mood_env/bin/activate` on Unix
pip install -r requirements.txt
```

### 🌐 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## 📈 Model Training

```bash
cd backend
python train.py
```

## 🔮 Prediction

```bash
python predict.py
```

## 💡 Future Enhancements

- 🎵 Image-to-Music Synthesis (Integrating computer vision + audio synthesis)
- 🗣️ Voice sentiment chatbot (Multimodal AI assistant)
- 📱 Mobile responsive UI

## 🧑‍💻 Author

**Anushka Aryan**  
👩‍💻 Passionate about AI, product engineering, and building mindful technologies.

