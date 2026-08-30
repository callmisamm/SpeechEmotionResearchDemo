
"""
======================================================================
Speech Emotion Recognition
Streamlit Thesis Demonstrator
CNN Version 2
======================================================================

Model:
    CNN Version 2

Dataset:
    RAVDESS

Feature:
    Log-Mel Spectrogram

Input:
    128 x 128 x 1

Classes:
    Angry, Calm, Disgust, Fear, Happy, Neutral, Sad, Surprise

Test Accuracy:
    69.44%

======================================================================
"""

# ====================================================================
# IMPORTS
# ====================================================================

import os
import tempfile

import cv2
import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from tensorflow.keras.models import load_model


# ====================================================================
# PAGE CONFIGURATION
# ====================================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ====================================================================
# CONSTANTS
# ====================================================================

SAMPLE_RATE = 22050

DURATION = 3

SAMPLES = SAMPLE_RATE * DURATION

MODEL_PATH = "models/cnn_model_v2.keras"

LABEL_PATH = "saved_features/label_encoder.pkl"

TEST_ACCURACY = 69.44

NUM_CLASSES = 8

INPUT_SIZE = (128, 128, 1)


# ====================================================================
# EMOTION ICONS
# ====================================================================

EMOTION_ICONS = {

    "Happy": "😊",

    "Sad": "😢",

    "Angry": "😠",

    "Fear": "😨",

    "Disgust": "🤢",

    "Surprise": "😲",

    "Neutral": "😐",

    "Calm": "😌"

}


# ====================================================================
# LOAD CNN MODEL
# ====================================================================

@st.cache_resource
def load_cnn_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"CNN model not found:\n{MODEL_PATH}"
        )

    model = load_model(MODEL_PATH)

    return model


# ====================================================================
# LOAD LABEL ENCODER
# ====================================================================

@st.cache_resource
def load_encoder():

    if not os.path.exists(LABEL_PATH):

        raise FileNotFoundError(
            f"Label encoder not found:\n{LABEL_PATH}"
        )

    encoder = joblib.load(LABEL_PATH)

    return encoder


# ====================================================================
# LOAD MODEL AND LABEL ENCODER
# ====================================================================

try:

    model = load_cnn_model()

    label_encoder = load_encoder()

    class_names = label_encoder.classes_

except Exception as e:

    st.error(
        "Unable to load the trained CNN model or label encoder."
    )

    st.exception(e)

    st.stop()


# ====================================================================
# SESSION STATE
# ====================================================================

if "prediction_done" not in st.session_state:

    st.session_state.prediction_done = False


if "predicted_emotion" not in st.session_state:

    st.session_state.predicted_emotion = None


if "predicted_confidence" not in st.session_state:

    st.session_state.predicted_confidence = None


if "prediction_vector" not in st.session_state:

    st.session_state.prediction_vector = None


if "spectrogram" not in st.session_state:

    st.session_state.spectrogram = None


if "audio_duration" not in st.session_state:

    st.session_state.audio_duration = None


# ====================================================================
# AUDIO LOADER
# ====================================================================

def load_audio(audio_path):

    """
    Load audio at 22.05 kHz and standardize it to exactly 3 seconds.

    If audio is longer than 3 seconds:
        First 3 seconds are retained.

    If audio is shorter than 3 seconds:
        Zero padding is applied.
    """

    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    original_duration = len(signal) / SAMPLE_RATE

    # ---------------------------------------------------------------
    # TRIM
    # ---------------------------------------------------------------

    if len(signal) > SAMPLES:

        signal = signal[:SAMPLES]

    # ---------------------------------------------------------------
    # PAD
    # ---------------------------------------------------------------

    elif len(signal) < SAMPLES:

        padding = SAMPLES - len(signal)

        signal = np.pad(
            signal,
            (0, padding),
            mode="constant"
        )

    return signal, original_duration


# ====================================================================
# FEATURE EXTRACTION
# ====================================================================

def extract_log_mel(signal):

    """
    Extract a 128 x 128 Log-Mel Spectrogram.

    Parameters:
        signal : 3-second audio signal

    Returns:
        normalized 128 x 128 Log-Mel feature
    """

    # ---------------------------------------------------------------
    # MEL SPECTROGRAM
    # ---------------------------------------------------------------

    mel = librosa.feature.melspectrogram(

        y=signal,

        sr=SAMPLE_RATE,

        n_fft=2048,

        hop_length=512,

        win_length=2048,

        n_mels=128,

        fmin=20,

        fmax=8000,

        power=2.0
    )

    # ---------------------------------------------------------------
    # CONVERT TO DECIBELS
    # ---------------------------------------------------------------

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # ---------------------------------------------------------------
    # RESIZE TO 128 x 128
    # ---------------------------------------------------------------

    mel_db = cv2.resize(
        mel_db,
        (128, 128),
        interpolation=cv2.INTER_LINEAR
    )

    # ---------------------------------------------------------------
    # MIN-MAX NORMALIZATION
    # ---------------------------------------------------------------

    min_value = mel_db.min()

    max_value = mel_db.max()

    mel_db = (
        mel_db - min_value
    ) / (
        max_value - min_value + 1e-8
    )

    return mel_db.astype(np.float32)


# ====================================================================
# DISPLAY SPECTROGRAM
# ====================================================================

def plot_spectrogram(feature):

    """
    Generate Log-Mel Spectrogram visualization.
    """

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    img = ax.imshow(

        feature,

        origin="lower",

        aspect="auto",

        cmap="viridis"
    )

    ax.set_title(
        "128 × 128 Normalized Log-Mel Spectrogram"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Mel Frequency Bands"
    )

    plt.colorbar(
        img,
        ax=ax,
        label="Normalized Magnitude"
    )

    plt.tight_layout()

    return fig


# ====================================================================
# PREDICTION FUNCTION
# ====================================================================

def predict_emotion(audio_path):

    """
    Complete inference pipeline:

    Audio
        ↓
    22.05 kHz loading
        ↓
    3-second standardization
        ↓
    Mel Spectrogram
        ↓
    Log-Mel conversion
        ↓
    128 × 128 resize
        ↓
    Min-Max normalization
        ↓
    CNN
        ↓
    8-class Softmax prediction
    """

    # ---------------------------------------------------------------
    # LOAD AUDIO
    # ---------------------------------------------------------------

    signal, original_duration = load_audio(
        audio_path
    )

    # ---------------------------------------------------------------
    # FEATURE EXTRACTION
    # ---------------------------------------------------------------

    feature = extract_log_mel(
        signal
    )

    # ---------------------------------------------------------------
    # ADD BATCH + CHANNEL DIMENSIONS
    # ---------------------------------------------------------------

    image = feature[
        np.newaxis,
        ...,
        np.newaxis
    ]

    # Expected shape:
    # (1, 128, 128, 1)

    # ---------------------------------------------------------------
    # CNN PREDICTION
    # ---------------------------------------------------------------

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    # ---------------------------------------------------------------
    # PREDICTED CLASS
    # ---------------------------------------------------------------

    predicted_index = int(
        np.argmax(prediction)
    )

    emotion = class_names[
        predicted_index
    ]

    confidence = float(
        prediction[predicted_index] * 100
    )

    return (

        emotion,

        confidence,

        prediction,

        feature,

        original_duration

    )


# ====================================================================
# PAGE HEADER
# ====================================================================

st.title(
    "🎤 Speech Emotion Recognition"
)

st.markdown(
    """
### CNN-Based Speech Emotion Recognition

This research demonstrator predicts **human emotions from speech**
using a **Convolutional Neural Network (CNN)** trained on the
**RAVDESS emotional speech dataset**.

The system converts speech into a **128 × 128 Log-Mel Spectrogram**
and uses the trained CNN to classify the speech into one of
**8 emotional categories**.
"""
)

st.divider()


# ====================================================================
# SIDEBAR
# ====================================================================

st.sidebar.title(
    "🔬 Model Information"
)

st.sidebar.markdown(
    "### CNN Version 2"
)

st.sidebar.success(
    "Convolutional Neural Network"
)

st.sidebar.markdown("---")

st.sidebar.write(
    "**Dataset:** RAVDESS"
)

st.sidebar.write(
    "**Audio Samples:** 1440"
)

st.sidebar.write(
    "**Classes:** 8"
)

st.sidebar.write(
    "**Feature:** Log-Mel Spectrogram"
)

st.sidebar.write(
    "**Input Size:** 128 × 128 × 1"
)

st.sidebar.write(
    "**Sampling Rate:** 22,050 Hz"
)

st.sidebar.write(
    "**Input Duration:** 3 seconds"
)

st.sidebar.write(
    "**Test Accuracy:** 69.44%"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### Emotion Classes"
)

for emotion in class_names:

    icon = EMOTION_ICONS.get(
        emotion,
        "🎤"
    )

    st.sidebar.write(
        f"{icon} {emotion}"
    )

st.sidebar.markdown("---")

st.sidebar.write(
    "**Architecture:**"
)

st.sidebar.write(
    "• 3 Convolution Blocks"
)

st.sidebar.write(
    "• Batch Normalization"
)

st.sidebar.write(
    "• Dropout"
)

st.sidebar.write(
    "• Global Average Pooling"
)

st.sidebar.write(
    "• Dense Layer"
)

st.sidebar.write(
    "• Softmax Output"
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Developed for"
)

st.sidebar.write(
    "**MPhil Thesis Demonstrator**"
)


# ====================================================================
# UPLOAD SECTION
# ====================================================================

st.header(
    "📂 Upload Speech Audio"
)

uploaded_file = st.file_uploader(

    "Choose a WAV audio file",

    type=["wav"],

    help="Upload a speech recording in WAV format."
)


# ====================================================================
# PROCESS UPLOADED AUDIO
# ====================================================================

if uploaded_file is not None:

    st.success(
        "Audio uploaded successfully."
    )

    # ---------------------------------------------------------------
    # SAVE TEMPORARY FILE
    # ---------------------------------------------------------------

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".wav"

        ) as temp_audio:

            temp_audio.write(
                uploaded_file.getvalue()
            )

            temp_path = temp_audio.name

        # -----------------------------------------------------------
        # AUDIO PLAYER
        # -----------------------------------------------------------

        st.subheader(
            "▶ Uploaded Audio"
        )

        st.audio(
            temp_path
        )

        # -----------------------------------------------------------
        # AUDIO INFORMATION
        # -----------------------------------------------------------

        try:

            audio_info, audio_sr = librosa.load(
                temp_path,
                sr=None,
                mono=True
            )

            duration = len(audio_info) / audio_sr

            info1, info2, info3 = st.columns(3)

            with info1:

                st.metric(
                    "Original Duration",
                    f"{duration:.2f} sec"
                )

            with info2:

                st.metric(
                    "Sampling Rate",
                    f"{audio_sr:,} Hz"
                )

            with info3:

                st.metric(
                    "Processing Duration",
                    "3.00 sec"
                )

        except Exception:

            pass

        # -----------------------------------------------------------
        # PREDICTION BUTTON
        # -----------------------------------------------------------

        predict_button = st.button(

            "🎯 Predict Emotion",

            type="primary",

            use_container_width=True
        )

        # -----------------------------------------------------------
        # RUN PREDICTION
        # -----------------------------------------------------------

        if predict_button:

            try:

                with st.spinner(
                    "Analyzing speech and extracting emotional features..."
                ):

                    (
                        emotion,
                        confidence,
                        probabilities,
                        feature,
                        original_duration
                    ) = predict_emotion(
                        temp_path
                    )

                # ---------------------------------------------------
                # STORE RESULTS
                # ---------------------------------------------------

                st.session_state.prediction_done = True

                st.session_state.predicted_emotion = emotion

                st.session_state.predicted_confidence = confidence

                st.session_state.prediction_vector = probabilities

                st.session_state.spectrogram = feature

                st.session_state.audio_duration = original_duration

                st.success(
                    "✅ Prediction completed successfully!"
                )

            except Exception as e:

                st.error(
                    "An error occurred during prediction."
                )

                st.exception(e)

    finally:

        # -----------------------------------------------------------
        # DELETE TEMPORARY FILE
        # -----------------------------------------------------------

        if temp_path is not None:

            try:

                if os.path.exists(temp_path):

                    os.remove(temp_path)

            except Exception:

                pass


# ====================================================================
# DISPLAY RESULTS ONLY AFTER PREDICTION
# ====================================================================

if st.session_state.prediction_done:

    predicted_emotion = (
        st.session_state.predicted_emotion
    )

    predicted_confidence = (
        st.session_state.predicted_confidence
    )

    prediction_vector = (
        st.session_state.prediction_vector
    )

    spectrogram = (
        st.session_state.spectrogram
    )


    # ================================================================
    # PREDICTION RESULTS
    # ================================================================

    st.divider()

    st.header(
        "🎯 Prediction Results"
    )


    # ================================================================
    # MAIN RESULT CARDS
    # ================================================================

    result_col1, result_col2, result_col3 = st.columns(3)


    # ---------------------------------------------------------------
    # PREDICTED EMOTION
    # ---------------------------------------------------------------

    with result_col1:

        st.markdown(
            "### Predicted Emotion"
        )

        icon = EMOTION_ICONS.get(
            predicted_emotion,
            "🎤"
        )

        st.success(
            f"{icon} {predicted_emotion}"
        )


    # ---------------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------------

    with result_col2:

        st.markdown(
            "### Confidence"
        )

        st.metric(

            label="Prediction Confidence",

            value=f"{predicted_confidence:.2f}%"
        )


    # ---------------------------------------------------------------
    # INPUT DURATION
    # ---------------------------------------------------------------

    with result_col3:

        st.markdown(
            "### Processed Audio"
        )

        st.metric(
            label="CNN Input",
            value="3.00 sec"
        )


    # ================================================================
    # CONFIDENCE PROGRESS BAR
    # ================================================================

    st.progress(
        min(
            int(predicted_confidence),
            100
        )
    )

    st.divider()


    # ================================================================
    # LOG-MEL SPECTROGRAM
    # ================================================================

    st.subheader(
        "🎼 Log-Mel Spectrogram"
    )

    st.markdown(
        """
The audio signal is transformed into a **Log-Mel Spectrogram**,
which represents the distribution of acoustic energy across
time and Mel-frequency bands.
"""
    )

    fig = plot_spectrogram(
        spectrogram
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(
        fig
    )

    st.divider()


    # ================================================================
    # EMOTION PROBABILITIES
    # ================================================================

    st.subheader(
        "📊 Emotion Probabilities"
    )

    probabilities_percent = (
        prediction_vector * 100
    )

    probability_dict = {

        class_names[i]:
        float(probabilities_percent[i])

        for i in range(
            len(class_names)
        )
    }

    chart_data = pd.DataFrame(

        {

            "Probability (%)":
            probability_dict

        }
    )

    st.bar_chart(
        chart_data
    )

    st.divider()


    # ================================================================
    # DETAILED PREDICTION SCORES
    # ================================================================

    st.subheader(
        "📋 Detailed Prediction Scores"
    )

    prediction_df = pd.DataFrame(

        {

            "Emotion":
            class_names,

            "Probability (%)":
            np.round(
                probabilities_percent,
                2
            )

        }
    )

    prediction_df = (
        prediction_df
        .sort_values(
            by="Probability (%)",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    prediction_df.index = (
        prediction_df.index + 1
    )

    prediction_df.index.name = (
        "Rank"
    )

    st.dataframe(

        prediction_df,

        use_container_width=True
    )

    st.divider()


    # ================================================================
    # TOP 3 PREDICTIONS
    # ================================================================

    st.subheader(
        "🏆 Top 3 Predictions"
    )

    top3 = prediction_df.head(
        3
    )

    cols = st.columns(3)

    rank_labels = [
        "🥇 Top Prediction",
        "🥈 Second Prediction",
        "🥉 Third Prediction"
    ]

    for i in range(3):

        with cols[i]:

            emotion_name = (
                top3.iloc[i]["Emotion"]
            )

            probability = (
                top3.iloc[i]["Probability (%)"]
            )

            icon = EMOTION_ICONS.get(
                emotion_name,
                "🎤"
            )

            st.markdown(
                f"""
                <div style="
                    padding:20px;
                    border-radius:12px;
                    border:1px solid #cccccc;
                    text-align:center;
                    margin-bottom:10px;
                ">

                <h4>{rank_labels[i]}</h4>

                <h2>{icon} {emotion_name}</h2>

                <h3>{probability:.2f}%</h3>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.divider()


    # ================================================================
    # MODEL INFORMATION
    # ================================================================

    st.subheader(
        "🧠 Model Information"
    )

    info_col1, info_col2 = st.columns(2)


    # ---------------------------------------------------------------
    # CNN ARCHITECTURE
    # ---------------------------------------------------------------

    with info_col1:

        st.info(
            """
### CNN Architecture

• Input: **128 × 128 × 1**

• Convolution Block 1: **32 filters**

• Convolution Block 2: **64 filters**

• Convolution Block 3: **128 filters**

• Batch Normalization

• Max Pooling

• Dropout Regularization

• Global Average Pooling

• Dense Layer: **256 neurons**

• Softmax Output: **8 classes**
"""
        )


    # ---------------------------------------------------------------
    # DATASET AND FEATURES
    # ---------------------------------------------------------------

    with info_col2:

        st.info(
            """
### Dataset & Features

• Dataset: **RAVDESS**

• Total Audio Files: **1440**

• Emotion Classes: **8**

• Sampling Rate: **22,050 Hz**

• Standardized Duration: **3 seconds**

• Feature: **Log-Mel Spectrogram**

• Mel Bands: **128**

• CNN Input: **128 × 128 × 1**

• Test Accuracy: **69.44%**
"""
        )


    st.divider()


    # ================================================================
    # INFERENCE PIPELINE
    # ================================================================

    st.subheader(
        "🔄 Inference Pipeline"
    )

    st.markdown(
        """
**Speech Audio**

⬇

**Load at 22,050 Hz**

⬇

**Trim / Zero-Pad to 3 Seconds**

⬇

**Mel-Spectrogram Extraction**

⬇

**Log-Mel Conversion**

⬇

**Resize to 128 × 128**

⬇

**Min-Max Normalization**

⬇

**CNN Model**

⬇

**8-Class Softmax Prediction**

⬇

**Predicted Emotion + Confidence**
"""
    )

    st.divider()


    # ================================================================
    # RESEARCH PERFORMANCE
    # ================================================================

    st.subheader(
        "📈 Research Performance"
    )

    performance_col1, performance_col2 = st.columns(2)


    with performance_col1:

        st.metric(
            "CNN Test Accuracy",
            "69.44%"
        )


    with performance_col2:

        st.metric(
            "Number of Emotion Classes",
            "8"
        )


    st.caption(
        """
The reported 69.44% accuracy is the performance of CNN Version 2
on the held-out test set of 216 samples.
"""
    )


    st.divider()


    # ================================================================
    # SUMMARY
    # ================================================================

    icon = EMOTION_ICONS.get(
        predicted_emotion,
        "🎤"
    )

    st.success(
        f"""
### ✅ Prediction Completed Successfully

**Predicted Emotion:** {icon} {predicted_emotion}

**Confidence:** {predicted_confidence:.2f}%

**Model:** CNN Version 2

**Test Accuracy:** {TEST_ACCURACY:.2f}%

**Feature Representation:** 128 × 128 Log-Mel Spectrogram
"""
    )


# ====================================================================
# INITIAL STATE MESSAGE
# ====================================================================

else:

    st.info(
        """
### 👋 Ready for Prediction

Upload a **WAV speech file** above and click
**Predict Emotion** to start the analysis.

The system will:

1. Process the speech audio
2. Standardize it to 3 seconds
3. Extract a Log-Mel Spectrogram
4. Convert it into a 128 × 128 feature representation
5. Pass the feature through CNN Version 2
6. Predict one of 8 emotions
7. Display confidence scores and Top-3 predictions
"""
    )


# ====================================================================
# FOOTER
# ====================================================================

st.markdown("---")

st.markdown(
    """
<div style="text-align:center;">

<h3>🎤 Speech Emotion Recognition Research Demonstrator</h3>

<p>
CNN-Based Speech Emotion Recognition
</p>

<p>
Developed for MPhil Thesis
</p>

<p>
Department of Computer Science
</p>

<p>
Powered by TensorFlow • Streamlit • Librosa • OpenCV
</p>

</div>
""",
    unsafe_allow_html=True
)
