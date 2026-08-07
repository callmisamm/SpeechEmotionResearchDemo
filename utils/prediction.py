"""
============================================================
Prediction Utilities
Speech Emotion Recognition
============================================================
"""

import time
import numpy as np
import joblib

from tensorflow.keras.models import load_model

from utils.feature_utils import (
    load_audio,
    extract_log_mel
)


# ==========================================================
# LOAD MODEL
# ==========================================================

MODEL_PATH = "models/cnn_model.keras"
LABEL_ENCODER_PATH = "saved_features/label_encoder.pkl"

model = None
label_encoder = None


def load_prediction_model():
    """
    Load CNN model only once.
    """

    global model

    if model is None:
        model = load_model(MODEL_PATH)

    return model


def load_label_encoder():
    """
    Load label encoder only once.
    """

    global label_encoder

    if label_encoder is None:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return label_encoder


# ==========================================================
# PREPARE MODEL INPUT
# ==========================================================

def prepare_input(audio_path):
    """
    Converts audio into model input.
    """

    signal = load_audio(audio_path)

    feature = extract_log_mel(signal)

    feature = np.expand_dims(feature, axis=0)
    feature = np.expand_dims(feature, axis=-1)

    return feature


# ==========================================================
# PREDICT EMOTION
# ==========================================================

def predict_emotion(audio_path):
    """
    Predict emotion from a WAV file.

    Returns:
        emotion
        confidence
        probabilities
        class_names
        inference_time
    """

    model = load_prediction_model()

    encoder = load_label_encoder()

    class_names = encoder.classes_

    feature = prepare_input(audio_path)

    start = time.time()

    prediction = model.predict(
        feature,
        verbose=0
    )

    inference_time = time.time() - start

    probabilities = prediction[0]

    predicted_index = np.argmax(probabilities)

    emotion = class_names[predicted_index]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    return {

        "emotion": emotion,

        "confidence": confidence,

        "probabilities": probabilities,

        "class_names": class_names,

        "prediction_index": predicted_index,

        "inference_time": inference_time

    }


# ==========================================================
# TOP K PREDICTIONS
# ==========================================================

def get_top_predictions(result, k=3):
    """
    Returns Top-K predictions.
    """

    probabilities = result["probabilities"]
    class_names = result["class_names"]

    indices = np.argsort(probabilities)[::-1][:k]

    top_predictions = []

    for idx in indices:

        top_predictions.append({

            "emotion": class_names[idx],

            "confidence": float(
                probabilities[idx] * 100
            )

        })

    return top_predictions


# ==========================================================
# EMOJI
# ==========================================================

def emotion_emoji(emotion):

    emoji = {

        "Happy": "😊",

        "Sad": "😢",

        "Angry": "😠",

        "Fear": "😨",

        "Disgust": "🤢",

        "Surprise": "😲",

        "Neutral": "😐",

        "Calm": "😌"

    }

    return emoji.get(emotion, "🎤")