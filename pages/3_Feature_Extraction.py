"""
===========================================================
Speech Emotion Recognition Research Demonstrator
Feature Extraction Page
===========================================================
"""

import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tempfile

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Feature Extraction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Audio Feature Extraction")

st.markdown("---")

st.write(
    """
This page visualizes the speech features extracted from the uploaded
audio signal.

The displayed features are identical to those commonly used in
Speech Emotion Recognition research.
"""
)

# ==========================================================
# AUDIO UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file is None:

    st.info("Please upload an audio file.")

    st.stop()

# ==========================================================
# SAVE TEMP FILE
# ==========================================================

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:

    temp_audio.write(uploaded_file.read())

    audio_path = temp_audio.name

# ==========================================================
# LOAD AUDIO
# ==========================================================

signal, sr = librosa.load(audio_path, sr=22050)

st.success("Audio Loaded Successfully")

# ==========================================================
# AUDIO PLAYER
# ==========================================================

st.audio(audio_path)

st.markdown("---")

# ==========================================================
# BASIC INFORMATION
# ==========================================================

duration = librosa.get_duration(y=signal, sr=sr)

col1, col2, col3 = st.columns(3)

col1.metric("Sample Rate", sr)

col2.metric("Duration", f"{duration:.2f} sec")

col3.metric("Samples", len(signal))
# ==========================================================
# WAVEFORM
# ==========================================================

st.subheader("Waveform")

fig, ax = plt.subplots(figsize=(12,4))

librosa.display.waveshow(
    signal,
    sr=sr,
    ax=ax
)

ax.set_title("Audio Waveform")

st.pyplot(fig)

# ==========================================================
# SPECTROGRAM
# ==========================================================

st.subheader("Spectrogram")

D = librosa.amplitude_to_db(
    np.abs(librosa.stft(signal)),
    ref=np.max
)

fig, ax = plt.subplots(figsize=(12,5))

img = librosa.display.specshow(
    D,
    sr=sr,
    x_axis="time",
    y_axis="hz",
    cmap="magma",
    ax=ax
)

fig.colorbar(img)

st.pyplot(fig)

# ==========================================================
# LOG MEL
# ==========================================================

st.subheader("Log-Mel Spectrogram")

mel = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    n_mels=128
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)

fig, ax = plt.subplots(figsize=(12,5))

img = librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis="time",
    y_axis="mel",
    cmap="viridis",
    ax=ax
)

fig.colorbar(img)

st.pyplot(fig)
# ==========================================================
# MFCC
# ==========================================================

st.subheader("MFCC")

mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=40
)

fig, ax = plt.subplots(figsize=(12,5))

img = librosa.display.specshow(
    mfcc,
    x_axis="time",
    sr=sr,
    cmap="coolwarm",
    ax=ax
)

fig.colorbar(img)

st.pyplot(fig)

# ==========================================================
# CHROMA
# ==========================================================

st.subheader("Chroma Features")

chroma = librosa.feature.chroma_stft(
    y=signal,
    sr=sr
)

fig, ax = plt.subplots(figsize=(12,5))

img = librosa.display.specshow(
    chroma,
    y_axis="chroma",
    x_axis="time",
    cmap="plasma",
    ax=ax
)

fig.colorbar(img)

st.pyplot(fig)
# ==========================================================
# RMS ENERGY
# ==========================================================

st.subheader("RMS Energy")

rms = librosa.feature.rms(y=signal)[0]

fig, ax = plt.subplots(figsize=(12,3))

ax.plot(rms)

ax.set_title("Root Mean Square Energy")

st.pyplot(fig)

# ==========================================================
# ZERO CROSSING RATE
# ==========================================================

st.subheader("Zero Crossing Rate")

zcr = librosa.feature.zero_crossing_rate(signal)[0]

fig, ax = plt.subplots(figsize=(12,3))

ax.plot(zcr)

ax.set_title("Zero Crossing Rate")

st.pyplot(fig)

# ==========================================================
# FEATURE SUMMARY
# ==========================================================

st.markdown("---")

st.subheader("Feature Summary")

feature_table = {
    "Feature":[
        "Waveform",
        "Spectrogram",
        "Log-Mel Spectrogram",
        "MFCC",
        "Chroma",
        "RMS Energy",
        "Zero Crossing Rate"
    ],
    "Purpose":[
        "Raw speech signal",
        "Frequency analysis",
        "CNN input representation",
        "Cepstral coefficients",
        "Pitch class information",
        "Speech energy",
        "Signal complexity"
    ]
}

st.table(feature_table)

st.success("Feature Extraction Completed Successfully.")
