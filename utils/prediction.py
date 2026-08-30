"""
============================================================
Prediction Utilities
Speech Emotion Recognition
CNN Version 2
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
# MODEL CONFIGURATION
# ==========================================================

MODEL_PATH = "models/cnn_model_v2.keras"

LABEL_ENCODER_PATH = (
    "saved_features/label_encoder.pkl"
)


# ==========================================================
# GLOBAL OBJECTS
# ==========================================================

model = None
label_encoder = None


# ==========================================================
# LOAD CNN V2 MODEL
# ==========================================================

def load_prediction_model():
    """
    Load the trained CNN Version 2 model.

    Model:
        cnn_model_v2.keras

    Test Accuracy:
        69.44%
    """

    global model

    if model is None:

        model = load_model(
            MODEL_PATH
        )

    return model


# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

def load_label_encoder():
    """
    Load the label encoder used during training.
    """

    global label_encoder

    if label_encoder is None:

        label_encoder = joblib.load(
            LABEL_ENCODER_PATH
        )

    return label_encoder


# ==========================================================
# PREPARE MODEL INPUT
# ==========================================================

def prepare_input(audio_path):
    """
    Convert WAV audio into the input format
    required by CNN Version 2.

    Pipeline:

        Audio
          ↓
        Load Audio
          ↓
        Log-Mel Spectrogram
          ↓
        Resize 128 × 128
          ↓
        Add Batch Dimension
          ↓
        Add Channel Dimension
          ↓
        CNN V2
    """

    # ------------------------------------------------------
    # Load audio
    # ------------------------------------------------------

    signal = load_audio(
        audio_path
    )

    # ------------------------------------------------------
    # Extract Log-Mel feature
    # ------------------------------------------------------

    feature = extract_log_mel(
        signal
    )

    # ------------------------------------------------------
    # Add batch dimension
    #
    # (128, 128)
    #       ↓
    # (1, 128, 128)
    # ------------------------------------------------------

    feature = np.expand_dims(
        feature,
        axis=0
    )

    # ------------------------------------------------------
    # Add channel dimension
    #
    # (1, 128, 128)
    #       ↓
    # (1, 128, 128, 1)
    # ------------------------------------------------------

    feature = np.expand_dims(
        feature,
        axis=-1
    )

    return feature.astype(
        np.float32
    )


# ==========================================================
# PREDICT EMOTION
# ==========================================================

def predict_emotion(audio_path):
    """
    Predict emotion from a WAV audio file
    using CNN Version 2.

    Returns:
        Dictionary containing:

        emotion
        confidence
        probabilities
        class_names
        prediction_index
        inference_time
    """

    # ------------------------------------------------------
    # Load CNN V2
    # ------------------------------------------------------

    model = load_prediction_model()

    # ------------------------------------------------------
    # Load label encoder
    # ------------------------------------------------------

    encoder = load_label_encoder()

    class_names = encoder.classes_

    # ------------------------------------------------------
    # Prepare input
    # ------------------------------------------------------

    feature = prepare_input(
        audio_path
    )

    # ------------------------------------------------------
    # CNN inference
    # ------------------------------------------------------

    start_time = time.time()

    prediction = model.predict(
        feature,
        verbose=0
    )

    inference_time = (
        time.time() - start_time
    )

    # ------------------------------------------------------
    # Extract probabilities
    # ------------------------------------------------------

    probabilities = prediction[0]

    # ------------------------------------------------------
    # Find highest probability
    # ------------------------------------------------------

    predicted_index = np.argmax(
        probabilities
    )

    # ------------------------------------------------------
    # Convert index → emotion
    # ------------------------------------------------------

    emotion = class_names[
        predicted_index
    ]

    # ------------------------------------------------------
    # Confidence
    # ------------------------------------------------------

    confidence = float(
        probabilities[
            predicted_index
        ] * 100
    )

    # ------------------------------------------------------
    # Return prediction
    # ------------------------------------------------------

    return {

        "emotion": emotion,

        "confidence": confidence,

        "probabilities": probabilities,

        "class_names": class_names,

        "prediction_index": int(
            predicted_index
        ),

        "inference_time": inference_time

    }


# ==========================================================
# TOP-K PREDICTIONS
# ==========================================================

def get_top_predictions(
    result,
    k=3
):
    """
    Return the Top-K emotion predictions.
    """

    probabilities = (
        result["probabilities"]
    )

    class_names = (
        result["class_names"]
    )

    # ------------------------------------------------------
    # Sort probabilities descending
    # ------------------------------------------------------

    indices = np.argsort(
        probabilities
    )[::-1][:k]

    top_predictions = []

    # ------------------------------------------------------
    # Create result list
    # ------------------------------------------------------

    for idx in indices:

        top_predictions.append({

            "emotion":
                class_names[idx],

            "confidence":
                float(
                    probabilities[idx] * 100
                )

        })

    return top_predictions


# ==========================================================
# EMOTION EMOJI
# ==========================================================

def emotion_emoji(
    emotion
):
    """
    Return an emoji corresponding
    to the predicted emotion.
    """

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

    return emoji.get(
        emotion,
        "🎤"
    )


# ==========================================================
# CHUNK-LEVEL EMOTION PREDICTION
# ==========================================================

def predict_emotion_by_chunks(
    audio_path
):
    """
    Predict emotion for every complete
    3-second audio chunk.

    Each chunk is independently processed
    by CNN Version 2.

    Returns:
        List containing prediction information
        for every complete chunk.
    """

    from utils.feature_utils import (
        split_audio_into_chunks
    )

    # ------------------------------------------------------
    # Load CNN V2
    # ------------------------------------------------------

    model = load_prediction_model()

    # ------------------------------------------------------
    # Load encoder
    # ------------------------------------------------------

    encoder = load_label_encoder()

    class_names = encoder.classes_

    # ------------------------------------------------------
    # Split audio into 3-second chunks
    # ------------------------------------------------------

    chunks, sample_rate = (
        split_audio_into_chunks(
            audio_path
        )
    )

    results = []

    # ------------------------------------------------------
    # Process every chunk
    # ------------------------------------------------------

    for i, chunk in enumerate(
        chunks
    ):

        # --------------------------------------------------
        # Extract Log-Mel
        # --------------------------------------------------

        feature = extract_log_mel(
            chunk
        )

        # --------------------------------------------------
        # Prepare CNN input
        #
        # (128, 128)
        #       ↓
        # (1, 128, 128, 1)
        # --------------------------------------------------

        model_input = np.expand_dims(
            feature,
            axis=0
        )

        model_input = np.expand_dims(
            model_input,
            axis=-1
        )

        model_input = model_input.astype(
            np.float32
        )

        # --------------------------------------------------
        # CNN V2 inference
        # --------------------------------------------------

        start_time = time.time()

        prediction = model.predict(
            model_input,
            verbose=0
        )

        inference_time = (
            time.time() - start_time
        )

        # --------------------------------------------------
        # Probabilities
        # --------------------------------------------------

        probabilities = prediction[0]

        # --------------------------------------------------
        # Predicted class
        # --------------------------------------------------

        predicted_index = np.argmax(
            probabilities
        )

        emotion = class_names[
            predicted_index
        ]

        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        confidence = float(
            probabilities[
                predicted_index
            ] * 100
        )

        # --------------------------------------------------
        # Chunk timing
        # --------------------------------------------------

        start_time_audio = (
            i * 3
        )

        end_time_audio = (
            start_time_audio + 3
        )

        # --------------------------------------------------
        # Save result
        # --------------------------------------------------

        results.append({

            "chunk":
                i + 1,

            "start_time":
                start_time_audio,

            "end_time":
                end_time_audio,

            "emotion":
                emotion,

            "confidence":
                confidence,

            "probabilities":
                probabilities,

            "inference_time":
                inference_time

        })

    return results


# ==========================================================
# CHUNK EMOTION STATISTICS
# ==========================================================

def calculate_chunk_statistics(
    chunk_results
):
    """
    Calculate emotion statistics from
    chunk-level CNN V2 predictions.

    Returns:

        emotion_counts
        emotion_percentages
        dominant_emotion
        dominant_percentage
        average_confidence
    """

    # ------------------------------------------------------
    # Handle empty results
    # ------------------------------------------------------

    if not chunk_results:

        return {

            "emotion_counts": {},

            "emotion_percentages": {},

            "dominant_emotion": None,

            "dominant_percentage": 0.0,

            "average_confidence": 0.0

        }

    # ------------------------------------------------------
    # Extract predicted emotions
    # ------------------------------------------------------

    emotions = [

        result["emotion"]

        for result in chunk_results

    ]

    # ------------------------------------------------------
    # Count emotions
    # ------------------------------------------------------

    unique_emotions, counts = (
        np.unique(
            emotions,
            return_counts=True
        )
    )

    emotion_counts = {

        emotion: int(count)

        for emotion, count
        in zip(
            unique_emotions,
            counts
        )

    }

    # ------------------------------------------------------
    # Calculate percentages
    # ------------------------------------------------------

    total_chunks = len(
        chunk_results
    )

    emotion_percentages = {

        emotion:
            (count / total_chunks) * 100

        for emotion, count
        in emotion_counts.items()

    }

    # ------------------------------------------------------
    # Dominant emotion
    # ------------------------------------------------------

    dominant_emotion = max(

        emotion_counts,

        key=emotion_counts.get

    )

    dominant_percentage = (
        emotion_percentages[
            dominant_emotion
        ]
    )

    # ------------------------------------------------------
    # Average confidence
    # ------------------------------------------------------

    average_confidence = np.mean([

        result["confidence"]

        for result in chunk_results

    ])

    # ------------------------------------------------------
    # Return statistics
    # ------------------------------------------------------

    return {

        "emotion_counts":
            emotion_counts,

        "emotion_percentages":
            emotion_percentages,

        "dominant_emotion":
            dominant_emotion,

        "dominant_percentage":
            float(
                dominant_percentage
            ),

        "average_confidence":
            float(
                average_confidence
            )

    }