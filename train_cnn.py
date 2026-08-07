"""
============================================================
Speech Emotion Recognition - CNN Training
Stage 1 : Dataset Loading
Stage 2 : Train / Validation / Test Split
============================================================
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from utils.feature_utils import (
    load_audio,
    extract_log_mel
)

from utils.augmentation import (
    add_noise,
    pitch_shift,
    time_stretch,
    volume_scale,
    time_shift
)

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = r"C:\Users\PMLS\Desktop\SpeechEmotionResearchDemo\dataset\RAVDESS"

# ==========================================================
# HEADER
# ==========================================================

print("=" * 60)
print("Speech Emotion Recognition")
print("=" * 60)

print("\nCurrent Directory:")
print(os.getcwd())
print("\nDataset Exists:", os.path.exists(DATASET_PATH))

# ==========================================================
# LOAD ALL WAV FILES
# ==========================================================

audio_files = glob.glob(
    os.path.join(DATASET_PATH, "Actor_*", "*.wav")
)

print("\nTotal wav files found:", len(audio_files))

# ==========================================================
# EMOTION MAP
# ==========================================================

emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}

# ==========================================================
# CREATE DATAFRAME
# ==========================================================

data = []

for file in audio_files:

    filename = os.path.basename(file)

    emotion_code = filename.split("-")[2]

    emotion = emotion_map[emotion_code]

    actor = os.path.basename(os.path.dirname(file))

    data.append({
        "Path": file,
        "Emotion": emotion,
        "Actor": actor
    })

df = pd.DataFrame(data)

# ==========================================================
# DATASET INFORMATION
# ==========================================================

print("\nEmotion Distribution:\n")
print(df["Emotion"].value_counts())

print("\nTotal Samples :", len(df))
print("Total Actors  :", df["Actor"].nunique())

# ==========================================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

print("\nCreating Train / Validation / Test Split...")

# 70% Train
# 30% Temporary

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["Emotion"]
)

# Remaining 30%
# Split equally -> 15% Validation + 15% Test

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Emotion"]
)

# ==========================================================
# DISPLAY SPLIT INFORMATION
# ==========================================================

print("\n" + "=" * 60)
print("Dataset Split Summary")
print("=" * 60)

print(f"Training Samples   : {len(train_df)}")
print(f"Validation Samples : {len(val_df)}")
print(f"Testing Samples    : {len(test_df)}")

print("\nTraining Distribution\n")
print(train_df["Emotion"].value_counts())

print("\nValidation Distribution\n")
print(val_df["Emotion"].value_counts())

print("\nTesting Distribution\n")
print(test_df["Emotion"].value_counts())

# ==========================================================
# SAVE SPLITS
# ==========================================================

results_folder = "results"

os.makedirs(results_folder, exist_ok=True)

train_df.to_csv(
    os.path.join(results_folder, "train_split.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(results_folder, "val_split.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(results_folder, "test_split.csv"),
    index=False
)

print("\nDataset splits saved successfully.")

print("\nSaved Files:")
print("results/train_split.csv")
print("results/val_split.csv")
print("results/test_split.csv")

print("\nDataset Loaded Successfully!")

print("=" * 60)

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

print("\n" + "=" * 60)
print("Extracting Log-Mel Features")
print("=" * 60)

os.makedirs("saved_features", exist_ok=True)

label_encoder = LabelEncoder()
label_encoder.fit(df["Emotion"])

joblib.dump(label_encoder, "saved_features/label_encoder.pkl")

print("\nLabel Encoder Saved.")

# ----------------------------------------------------------
# PROCESS DATASET
# ----------------------------------------------------------

def process_dataset(dataset, augment=False):

    X = []
    y = []

    for _, row in tqdm(
        dataset.iterrows(),
        total=len(dataset),
        desc="Processing"
    ):

        path = row["Path"]
        emotion = row["Emotion"]

        signal = load_audio(path)

        # ==================================================
        # Original Feature
        # ==================================================

        feature = extract_log_mel(signal)

        X.append(feature)
        y.append(emotion)

        # ==================================================
        # TRAINING AUGMENTATION ONLY
        # ==================================================

        if augment:

            # -----------------------------
            # Noise
            # -----------------------------

            aug = add_noise(signal)

            X.append(extract_log_mel(aug))
            y.append(emotion)

            # -----------------------------
            # Pitch Shift
            # -----------------------------

            aug = pitch_shift(signal)

            X.append(extract_log_mel(aug))
            y.append(emotion)

            # -----------------------------
            # Time Stretch
            # -----------------------------

            aug = time_stretch(signal)

            X.append(extract_log_mel(aug))
            y.append(emotion)

            # -----------------------------
            # Volume Scaling
            # -----------------------------

            aug = volume_scale(signal)

            X.append(extract_log_mel(aug))
            y.append(emotion)

            # -----------------------------
            # Time Shift
            # -----------------------------

            aug = time_shift(signal)

            X.append(extract_log_mel(aug))
            y.append(emotion)

    # ==================================================
    # Convert to NumPy
    # ==================================================

    X = np.array(X, dtype=np.float32)

    X = X[..., np.newaxis]

    y = label_encoder.transform(y)

    return X, y

# ==========================================================
# TRAIN FEATURES
# ==========================================================

print("\nExtracting Training Features...")

X_train, y_train = process_dataset(
    train_df,
    augment=True
)

# ==========================================================
# VALIDATION FEATURES
# ==========================================================

print("\nExtracting Validation Features...")

X_val, y_val = process_dataset(
    val_df,
    augment=False
)

# ==========================================================
# TEST FEATURES
# ==========================================================

print("\nExtracting Testing Features...")

X_test, y_test = process_dataset(
    test_df,
    augment=False
)

# ==========================================================
# SAVE FEATURES
# ==========================================================

np.save("saved_features/X_train.npy", X_train)
np.save("saved_features/y_train.npy", y_train)

np.save("saved_features/X_val.npy", X_val)
np.save("saved_features/y_val.npy", y_val)

np.save("saved_features/X_test.npy", X_test)
np.save("saved_features/y_test.npy", y_test)

print("\nFeatures Saved Successfully!")

print("\nSaved Files")

print("saved_features/X_train.npy")
print("saved_features/y_train.npy")

print("saved_features/X_val.npy")
print("saved_features/y_val.npy")

print("saved_features/X_test.npy")
print("saved_features/y_test.npy")

print("saved_features/label_encoder.pkl")

print("\nFeature Shapes")

print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)

print("\nNumber of Classes :", len(label_encoder.classes_))
print("Class Names :", list(label_encoder.classes_))

print("\nTraining Labels :", y_train.shape)
print("Validation Labels :", y_val.shape)
print("Testing Labels :", y_test.shape)

print("\nPreprocessing Completed Successfully!")

print("=" * 60)