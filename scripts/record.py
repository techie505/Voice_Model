import sounddevice as sd
import wavio
import os

# Ensure recordings folder exists
if not os.path.exists("recordings"):
    os.makedirs("recordings")

def record_audio(filename="recordings/live_input.wav", duration=3, sr=16000):
    print("🎤 Recording... Speak now!")
    
    # Record audio
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished

    # Save the file
    wavio.write(filename, audio, sr, sampwidth=2)
    print(f"✅ Recording complete! File saved at: {filename}")

# Start recording
record_audio()

