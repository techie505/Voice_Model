import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from collections import Counter

# ✅ Dataset Path
DATASET_PATH = r"C:\Users\anush\OneDrive\Desktop\voice model\dataset\AudioWAV"
if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: Dataset path '{DATASET_PATH}' does not exist.")
    exit()

# ✅ Emotion Labels
label_map = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
num_classes = len(label_map)

# ✅ Feature Extraction (Mel Spectrograms)
def extract_features(audio, sr, max_pad_len=220):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, fmax=8000)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec.shape[1] < max_pad_len:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max_pad_len - mel_spec.shape[1])), mode='constant')
    else:
        mel_spec = mel_spec[:, :max_pad_len]

    return mel_spec

# ✅ Data Augmentation (Mild Adjustments)
def augment_audio(audio, sr):
    augmented_audio = [audio]
    if np.random.rand() > 0.5:
        augmented_audio.append(librosa.effects.time_stretch(audio, rate=np.random.uniform(0.95, 1.05)))
    return augmented_audio

# ✅ Load Dataset
data, labels = [], []
audio_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".wav")]
print(f"📂 Total audio files found: {len(audio_files)}")

for file in audio_files:
    file_path = os.path.join(DATASET_PATH, file)
    parts = file.split("_")

    if len(parts) < 3:
        print(f"⚠️ Skipping invalid file: {file}")
        continue

    emotion = parts[2][:3]
    if emotion in label_map:
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            augmented_audios = augment_audio(audio, sr)

            for aug_audio in augmented_audios:
                feature = extract_features(aug_audio, sr)
                if feature.shape == (64, 220):  # ✅ Ensure Correct Shape
                    data.append(feature)
                    labels.append(label_map[emotion])
                    print(f"✅ Processed: {file} → Label: {emotion}")
                else:
                    print(f"⚠️ Skipping {file} due to incorrect shape: {feature.shape}")

        except Exception as e:
            print(f"❌ Skipping {file} due to error: {e}")

# ✅ Convert Data to NumPy Arrays
if len(data) == 0:
    print("❌ No valid samples extracted! Check dataset format.")
    exit()

X = np.array(data)[..., np.newaxis]  # ✅ Ensure shape (64, 220, 1)
y = np.array(labels)

# ✅ Balanced Class Weights
class_counts = Counter(y)
total_samples = len(y)
class_weights = {i: max(0.5, total_samples / class_counts[i]) for i in class_counts}
print("📊 Updated Class Weights:", class_weights)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, to_categorical(y, num_classes=num_classes), test_size=0.2, stratify=y, random_state=42
)
print(f"📊 Training samples: {X_train.shape}, Test samples: {X_test.shape}")

# ✅ CNN Model (ResNet-Inspired)
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(64, 220, 1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    return model

# ✅ Compile Model
model = build_model()
model.compile(
    optimizer=AdamW(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks
model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss", mode="min", verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# ✅ Train Model
history = model.fit(
    X_train, y_train,
    epochs=50,  
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[model_checkpoint, lr_scheduler, early_stopping]
)

# ✅ Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"🎯 Test Accuracy: {test_accuracy * 100:.2f}%")

# ✅ Save Model
MODEL_PATH = r"C:\Users\anush\OneDrive\Desktop\voice model\models\mood_classifier_updated.h5"
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")