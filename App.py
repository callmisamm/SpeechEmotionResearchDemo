"""
============================================================
Speech Emotion Recognition
Streamlit Thesis Demonstrator
Part 1 : Imports + Model Loading + Helper Functions
============================================================
"""

import os
import tempfile

import cv2
import joblib
import librosa
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from tensorflow.keras.models import load_model


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# ==========================================================
# CONSTANTS
# ==========================================================

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

MODEL_PATH = "models/cnn_model.keras"
LABEL_PATH = "saved_features/label_encoder.pkl"

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_cnn_model():

    model = load_model(MODEL_PATH)

    return model


# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

@st.cache_resource
def load_encoder():

    encoder = joblib.load(LABEL_PATH)

    return encoder


model = load_cnn_model()

label_encoder = load_encoder()

class_names = label_encoder.classes_

# ==========================================================
# AUDIO LOADER
# ==========================================================

def load_audio(audio_path):

    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    # Trim

    if len(signal) > SAMPLES:

        signal = signal[:SAMPLES]

    # Pad

    else:

        padding = SAMPLES - len(signal)

        signal = np.pad(
            signal,
            (0, padding),
            mode="constant"
        )

    return signal


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_log_mel(signal):

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

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = cv2.resize(
        mel_db,
        (128, 128)
    )

    mel_db = (
        mel_db - mel_db.min()
    ) / (
        mel_db.max() - mel_db.min() + 1e-8
    )

    return mel_db.astype(np.float32)


# ==========================================================
# DISPLAY SPECTROGRAM
# ==========================================================

def plot_spectrogram(feature):

    fig, ax = plt.subplots(figsize=(6,4))

    img = ax.imshow(
        feature,
        origin="lower",
        aspect="auto",
        cmap="viridis"
    )

    ax.set_title("Log-Mel Spectrogram")

    ax.set_xlabel("Time")

    ax.set_ylabel("Mel Bands")

    plt.colorbar(img)

    return fig


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_emotion(audio_path):

    signal = load_audio(audio_path)

    feature = extract_log_mel(signal)

    image = feature[np.newaxis, ..., np.newaxis]

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    emotion = class_names[predicted_index]

    confidence = prediction[predicted_index] * 100

    return (

        emotion,

        confidence,

        prediction,

        feature

    )


# ==========================================================
# PAGE HEADER
# ==========================================================

st.title("🎤 Speech Emotion Recognition")

st.markdown(
    """
This Streamlit application predicts **human emotions from speech**
using a Convolutional Neural Network (CNN) trained on
the **RAVDESS emotional speech dataset**.
"""
)

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Model Information")

st.sidebar.markdown("### CNN Model")

st.sidebar.success("Convolutional Neural Network")

st.sidebar.markdown("---")

st.sidebar.write("**Dataset:** RAVDESS")

st.sidebar.write("**Feature:** Log-Mel Spectrogram")

st.sidebar.write("**Input Size:** 128 × 128")

st.sidebar.write("**Classes:** 8")

st.sidebar.write("**Training Dataset:** 1440 Audio Files")

st.sidebar.write("**Test Accuracy:** 68.52%")

st.sidebar.markdown("---")

st.sidebar.write("Developed for")
st.sidebar.write("**MPhil Thesis Demonstrator**")

# ==========================================================
# FILE UPLOADER
# ==========================================================

st.header("📂 Upload Speech Audio")

uploaded_file = st.file_uploader(

    "Choose a WAV audio file",

    type=["wav"]

)

# ==========================================================
# WAIT UNTIL USER UPLOADS FILE
# ==========================================================

if uploaded_file is not None:

    st.success("Audio uploaded successfully.")

    # ------------------------------------------------------
    # SAVE TEMP FILE
    # ------------------------------------------------------

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".wav"

    ) as temp_audio:

        temp_audio.write(uploaded_file.read())

        temp_path = temp_audio.name

    # ------------------------------------------------------
    # PLAY AUDIO
    # ------------------------------------------------------

    st.subheader("▶ Uploaded Audio")

    st.audio(temp_path)

    # ------------------------------------------------------
    # PREDICT BUTTON
    # ------------------------------------------------------

    predict_button = st.button(

        "Predict Emotion"

    )

    # ------------------------------------------------------
    # RUN PREDICTION
    # ------------------------------------------------------

    if predict_button:

        with st.spinner("Analyzing speech..."):

            emotion, confidence, probabilities, feature = predict_emotion(
                temp_path
            )

        st.success("Prediction Completed Successfully")

        # --------------------------------------------------
        # DISPLAY SPECTROGRAM
        # --------------------------------------------------

        st.subheader("Log-Mel Spectrogram")

        fig = plot_spectrogram(feature)

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        # --------------------------------------------------
        # SAVE RESULTS
        # --------------------------------------------------

        predicted_emotion = emotion

        predicted_confidence = confidence

        prediction_vector = probabilities

        spectrogram = feature
        
# ==========================================================
# PREDICTION RESULTS
# ==========================================================

st.header("Prediction Results")

col1, col2 = st.columns(2)

# ----------------------------------------------------------
# PREDICTED EMOTION
# ----------------------------------------------------------

with col1:

    st.markdown("### Predicted Emotion")

    emotion_icons = {

        "Happy": "😊",
        "Sad": "😢",
        "Angry": "😠",
        "Fear": "😨",
        "Disgust": "🤢",
        "Surprise": "😲",
        "Neutral": "😐",
        "Calm": "😌"

    }

    icon = emotion_icons.get(
        predicted_emotion,
        "🎤"
    )

    st.success(
        f"{icon} {predicted_emotion}"
    )

# ----------------------------------------------------------
# CONFIDENCE
# ----------------------------------------------------------

with col2:

    st.markdown("### Confidence")

    st.metric(

        label="Prediction Confidence",

        value=f"{predicted_confidence:.2f}%"

    )

st.divider()

# ==========================================================
# PROBABILITY BAR CHART
# ==========================================================

st.subheader("Emotion Probabilities")

probabilities_percent = prediction_vector * 100

probability_dict = {

    class_names[i]: probabilities_percent[i]

    for i in range(len(class_names))

}

chart_data = (
    probability_dict
)

st.bar_chart(chart_data)

st.divider()

# ==========================================================
# DETAILED PREDICTIONS
# ==========================================================

st.subheader("Detailed Prediction Scores")

import pandas as pd

prediction_df = pd.DataFrame({

    "Emotion": class_names,

    "Probability (%)": np.round(
        probabilities_percent,
        2
    )

})

prediction_df = prediction_df.sort_values(

    by="Probability (%)",

    ascending=False

).reset_index(drop=True)

st.dataframe(

    prediction_df,

    use_container_width=True

)

st.divider()

# ==========================================================
# TOP 3 PREDICTIONS
# ==========================================================

st.subheader("Top 3 Predictions")

top3 = prediction_df.head(3)

gold = "#FFD700"
silver = "#C0C0C0"
bronze = "#CD7F32"

colors = [

    gold,

    silver,

    bronze

]

cols = st.columns(3)

for i in range(3):

    with cols[i]:

        st.markdown(

            f"""
<div style="background-color:{colors[i]};
padding:15px;
border-radius:12px;
text-align:center;
color:black;">

<h3>{top3.iloc[i]['Emotion']}</h3>

<h2>{top3.iloc[i]['Probability (%)']:.2f}%</h2>

</div>
""",

            unsafe_allow_html=True

        )

st.divider()

# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.subheader("Model Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.info(
        """
**Model**

• CNN

• 3 Convolution Blocks

• Batch Normalization

• Dropout

• Global Average Pooling

• Softmax Output
"""
    )

with info_col2:

    st.info(
        """
**Dataset**

• RAVDESS

• 8 Emotions

• Log-Mel Spectrogram

• Input Size : 128 × 128

• Test Accuracy : 68.52%
"""
    )

st.divider()

# ==========================================================
# SUMMARY
# ==========================================================

st.success(

    f"""
Prediction Completed Successfully!

Predicted Emotion : {predicted_emotion}

Confidence : {predicted_confidence:.2f}%
"""
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(

"""
<center>

### Speech Emotion Recognition Research Demonstrator

Developed for MPhil Thesis

Department of Computer Science

Powered by TensorFlow • Streamlit • Librosa

</center>

""",

unsafe_allow_html=True

)       