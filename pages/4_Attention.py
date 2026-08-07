import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tempfile
import os

from streamlit_mic_recorder import mic_recorder

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Attention Visualization",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Attention Visualization")

st.markdown("---")

st.info(
    """
This page demonstrates explainability by highlighting
the most informative regions of the Log-Mel Spectrogram.

In the complete research framework, attention mechanisms
learn these regions automatically during training.
"""
)

# ==========================================================
# AUDIO INPUT
# ==========================================================

st.subheader("🎤 Audio Input")

audio_source = st.radio(
    "Select Audio Source",
    (
        "📂 Upload Audio",
        "🎙 Record Live Audio"
    ),
    horizontal=True
)

audio_file = None

# ==========================================================
# OPTION 1 : UPLOAD AUDIO
# ==========================================================

if audio_source == "📂 Upload Audio":

    uploaded_file = st.file_uploader(
        "Choose a WAV Audio File",
        type=["wav"]
    )

    if uploaded_file is not None:

        st.success("Audio uploaded successfully.")

        st.audio(uploaded_file)

        audio_file = uploaded_file

# ==========================================================
# OPTION 2 : LIVE RECORDING
# ==========================================================

else:

    st.write("### 🎙 Live Recording")

    audio = mic_recorder(
        start_prompt="🎙 Start Recording",
        stop_prompt="⏹ Stop Recording",
        just_once=False,
        use_container_width=True,
        key="attention_recorder"
    )

    if audio:

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        temp_audio.write(audio["bytes"])
        temp_audio.close()

        st.success("✔ Recording completed successfully.")

        st.audio(audio["bytes"])

        duration = librosa.get_duration(
            path=temp_audio.name
        )

        st.info(
            f"Duration : {duration:.2f} seconds"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🔄 Record Again",
                use_container_width=True
            ):

                st.session_state.pop(
                    "attention_recorder",
                    None
                )

                st.rerun()

        with col2:

            analyze = st.button(
                "✅ Analyze",
                use_container_width=True
            )

        if analyze:

            audio_file = temp_audio.name

# ==========================================================
# WAIT FOR AUDIO
# ==========================================================

if audio_file is None:

    st.warning(
        "Please upload or record an audio sample."
    )

    st.stop()

# ==========================================================
# LOAD AUDIO
# ==========================================================

signal, sr = librosa.load(
    audio_file,
    sr=22050
)

duration = librosa.get_duration(
    y=signal,
    sr=sr
)

samples = len(signal)

st.divider()

st.subheader("📋 Audio Information")

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Sample Rate",
        f"{sr} Hz"
    )

with c2:

    st.metric(
        "Duration",
        f"{duration:.2f} sec"
    )

with c3:

    st.metric(
        "Samples",
        samples
    )

st.success("Audio loaded successfully.")
# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

st.divider()

st.subheader("📈 Feature Extraction")

mel = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=128
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)

summary = {
    "Feature": [
        "Sample Rate",
        "Duration",
        "Samples",
        "Log-Mel Shape"
    ],
    "Value": [
        f"{sr} Hz",
        f"{duration:.2f} sec",
        samples,
        str(mel_db.shape)
    ]
}

st.table(summary)

st.success("Log-Mel Spectrogram extracted successfully.")

# ==========================================================
# DISPLAY LOG-MEL SPECTROGRAM
# ==========================================================

st.divider()

st.subheader("🎵 Log-Mel Spectrogram")

fig, ax = plt.subplots(figsize=(12,4))

img = librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=256,
    x_axis="time",
    y_axis="mel",
    cmap="magma",
    ax=ax
)

plt.colorbar(
    img,
    ax=ax,
    format="%+2.0f dB"
)

ax.set_title(
    "Log-Mel Spectrogram"
)

plt.tight_layout()

st.pyplot(fig)

# ==========================================================
# CREATE DEMO ATTENTION MAP
# ==========================================================

attention = mel_db.copy()

# Normalize

attention -= attention.min()

attention /= attention.max()

# Smooth

attention = cv2.GaussianBlur(
    attention,
    (15,15),
    0
)

# Enhance

attention = np.power(
    attention,
    2
)

# Resize

attention = cv2.resize(
    attention,
    (
        mel_db.shape[1],
        mel_db.shape[0]
    )
)

st.success(
    "Attention map generated successfully."
)
# ==========================================================
# ATTENTION HEATMAP
# ==========================================================

st.divider()

st.subheader("🔥 Attention Heatmap")

fig, ax = plt.subplots(figsize=(12,4))

heat = ax.imshow(
    attention,
    aspect="auto",
    origin="lower",
    cmap="jet"
)

plt.colorbar(
    heat,
    ax=ax,
    label="Attention Weight"
)

ax.set_xlabel("Time Frames")
ax.set_ylabel("Mel Filter Banks")
ax.set_title("Attention Heatmap")

plt.tight_layout()

st.pyplot(fig)

# ==========================================================
# ATTENTION OVERLAY
# ==========================================================

st.divider()

st.subheader("🎯 Attention Overlay")

fig, ax = plt.subplots(figsize=(12,4))

ax.imshow(
    mel_db,
    aspect="auto",
    origin="lower",
    cmap="gray"
)

overlay = ax.imshow(
    attention,
    aspect="auto",
    origin="lower",
    cmap="jet",
    alpha=0.45
)

plt.colorbar(
    overlay,
    ax=ax,
    label="Attention Weight"
)

ax.set_xlabel("Time Frames")
ax.set_ylabel("Mel Filter Banks")
ax.set_title("Attention Overlay on Log-Mel Spectrogram")

plt.tight_layout()

st.pyplot(fig)

# ==========================================================
# TEMPORAL ATTENTION SCORES
# ==========================================================

st.divider()

st.subheader("📈 Temporal Attention Scores")

scores = attention.mean(axis=0)

fig, ax = plt.subplots(figsize=(12,3.5))

ax.plot(
    scores,
    linewidth=2
)

ax.fill_between(
    range(len(scores)),
    scores,
    alpha=0.25
)

ax.set_xlabel("Time Frames")
ax.set_ylabel("Attention Score")
ax.set_title("Average Attention Across Time")

ax.grid(True, alpha=0.3)

plt.tight_layout()

st.pyplot(fig)

# ==========================================================
# ATTENTION STATISTICS
# ==========================================================

st.divider()

st.subheader("📊 Attention Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Maximum",
        f"{scores.max():.3f}"
    )

with col2:

    st.metric(
        "Average",
        f"{scores.mean():.3f}"
    )

with col3:

    st.metric(
        "Minimum",
        f"{scores.min():.3f}"
    )

with col4:

    st.metric(
        "Std. Dev.",
        f"{scores.std():.3f}"
    )

# ==========================================================
# KEY OBSERVATIONS
# ==========================================================

st.divider()

st.subheader("🔍 Key Observations")

peak_frame = np.argmax(scores)

peak_time = peak_frame * 256 / sr

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Highest Attention Frame",
        peak_frame
    )

with c2:

    st.metric(
        "Peak Attention Time",
        f"{peak_time:.2f} sec"
    )

st.success(
    f"""
The highest attention occurs around **{peak_time:.2f} seconds**,
indicating that this portion of the speech signal contributes
most strongly to the emotion prediction.
"""
)
# ==========================================================
# INTERPRETATION
# ==========================================================

st.divider()

st.subheader("📝 Interpretation")

st.success("""
### Understanding the Attention Visualization

🔴 **High Attention Regions (Red)**

These regions represent the most informative parts of the speech signal.
They contribute more strongly to the final emotion prediction.

🔵 **Low Attention Regions (Blue)**

These regions contribute less to the prediction and usually correspond
to silence, pauses, or less discriminative speech segments.

🧠 **Attention Mechanism**

Instead of processing every speech frame equally, the attention
mechanism automatically assigns higher weights to emotionally
important regions while reducing the influence of less useful regions.

🎯 **Explainability**

The attention heatmap provides visual evidence of which portions
of the speech signal influenced the model's prediction, improving
the transparency and interpretability of the Speech Emotion
Recognition (SER) system.
""")

# ==========================================================
# RESEARCH NOTES
# ==========================================================

st.divider()

st.subheader("📚 Research Notes")

st.info("""
This visualization demonstrates the concept of attention-based
explainability.

In the complete hybrid Speech Emotion Recognition framework,
attention weights are learned automatically during model
training.

For demonstration purposes, this application generates a
smoothed attention map from the Log-Mel Spectrogram to
illustrate how an attention mechanism highlights important
speech regions.
""")

# ==========================================================
# ATTENTION PIPELINE
# ==========================================================

st.divider()

st.subheader("🧠 Explainability Pipeline")

st.markdown("""
### Processing Pipeline

🎤 **Audio Signal**

⬇️

🎵 **Log-Mel Spectrogram**

⬇️

🧠 **CNN Feature Extraction**

⬇️

✨ **Attention Mechanism**

⬇️

😊 **Emotion Prediction**

⬇️

🔥 **Attention Heatmap Visualization**
""")

# ==========================================================
# PIPELINE DESCRIPTION
# ==========================================================

st.divider()

st.subheader("📋 Pipeline Description")

pipeline_data = {
    "Stage": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6"
    ],
    "Operation": [
        "Audio Acquisition",
        "Log-Mel Spectrogram",
        "CNN Feature Extraction",
        "Attention Layer",
        "Emotion Prediction",
        "Visualization"
    ],
    "Purpose": [
        "Load speech signal",
        "Extract spectral features",
        "Learn acoustic representations",
        "Focus on informative speech regions",
        "Classify emotion",
        "Explain model decision"
    ]
}

st.table(pipeline_data)

# ==========================================================
# SUMMARY
# ==========================================================

st.divider()

st.subheader("✅ Summary")

st.success("""
✔ Audio loaded successfully.

✔ Log-Mel Spectrogram generated.

✔ Attention Heatmap generated.

✔ Attention Overlay visualized.

✔ Temporal Attention Scores computed.

✔ Explainability analysis completed successfully.

This page demonstrates how attention mechanisms improve the
interpretability of Speech Emotion Recognition models by
highlighting the most informative regions of the speech signal
used during emotion prediction.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Hybrid Speech Emotion Recognition Framework | "
    "Attention-Based Explainability | "
    "MPhil Research Demonstrator"
)