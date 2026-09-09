"""
=========================================================
Visualization Utilities
Speech Emotion Recognition
=========================================================
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# WAVEFORM
# ==========================================================

def plot_waveform(signal, sr=22050):

    fig, ax = plt.subplots(figsize=(10, 3))

    librosa.display.waveshow(
        signal,
        sr=sr,
        ax=ax
    )

    ax.set_title("Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    plt.tight_layout()

    return fig


# ==========================================================
# SPECTROGRAM
# ==========================================================

def plot_spectrogram(signal, sr=22050):

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(signal)),
        ref=np.max
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    img = librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax
    )

    fig.colorbar(
        img,
        ax=ax,
        format="%+2.0f dB"
    )

    ax.set_title("Spectrogram")

    plt.tight_layout()

    return fig


# ==========================================================
# LOG-MEL
# ==========================================================

def plot_logmel(signal, sr=22050):

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        cmap="viridis",
        ax=ax
    )

    fig.colorbar(
        img,
        ax=ax,
        format="%+2.0f dB"
    )

    ax.set_title("Log-Mel Spectrogram")

    plt.tight_layout()

    return fig


# ==========================================================
# MFCC
# ==========================================================

def plot_mfcc(signal, sr=22050):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        cmap="coolwarm",
        ax=ax
    )

    fig.colorbar(img, ax=ax)

    ax.set_title("MFCC")

    plt.tight_layout()

    return fig


# ==========================================================
# CHROMA
# ==========================================================

def plot_chroma(signal, sr=22050):

    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr
    )

    fig, ax = plt.subplots(figsize=(10, 3))

    img = librosa.display.specshow(
        chroma,
        x_axis="time",
        y_axis="chroma",
        cmap="plasma",
        ax=ax
    )

    fig.colorbar(img, ax=ax)

    ax.set_title("Chroma Features")

    plt.tight_layout()

    return fig


# ==========================================================
# ATTENTION HEATMAP (Demo)
# ==========================================================

def plot_attention_heatmap(length=128):

    attention = np.random.rand(length)

    # make one region dominant
    attention[45:70] += 1.5

    attention = attention / attention.max()

    fig, ax = plt.subplots(figsize=(10, 2))

    img = ax.imshow(
        attention.reshape(1, -1),
        cmap="hot",
        aspect="auto"
    )

    ax.set_title("Attention Heatmap")
    ax.set_xlabel("Speech Frames")
    ax.set_yticks([])

    fig.colorbar(
        img,
        ax=ax
    )

    plt.tight_layout()

    return fig


# ==========================================================
# EMOTION PROBABILITY BAR CHART
# ==========================================================

def plot_probability_chart(probabilities, class_names):

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = [
        "green" if p == np.max(probabilities)
        else "steelblue"
        for p in probabilities
    ]

    ax.bar(
        class_names,
        probabilities * 100,
        color=colors
    )

    ax.set_ylim(0, 100)

    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Emotion")
    ax.set_title("Emotion Probability Distribution")

    plt.xticks(rotation=30)

    plt.tight_layout()

    return fig

# ==========================================================
# TEMPORAL EMOTION TIMELINE
# ==========================================================

def plot_emotion_timeline(chunk_results):
    """
    Create a research-ready temporal emotion timeline.

    Each horizontal bar represents one complete 3-second
    audio chunk positioned according to its actual start
    and end time.

    The visualization preserves the 50% overlap between
    consecutive chunks.

    Parameters:
        chunk_results: list of dictionaries returned by
                       predict_emotion_by_chunks()

    Returns:
        matplotlib figure
    """

    if not chunk_results:
        return None

    # ======================================================
    # EXTRACT CHUNK INFORMATION
    # ======================================================

    start_times = []
    end_times = []
    emotions = []
    confidences = []

    for chunk in chunk_results:

        start_times.append(
            float(chunk["start_time"])
        )

        end_times.append(
            float(chunk["end_time"])
        )

        emotions.append(
            chunk["emotion"]
        )

        confidences.append(
            float(chunk["confidence"])
        )

    # ======================================================
    # UNIQUE EMOTIONS
    # ======================================================

    unique_emotions = []

    for emotion in emotions:

        if emotion not in unique_emotions:
            unique_emotions.append(emotion)

    # ======================================================
    # CREATE FIGURE
    # ======================================================

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    # ======================================================
    # EMOTION POSITIONS
    # ======================================================

    emotion_positions = {
        emotion: index
        for index, emotion in enumerate(
            unique_emotions
        )
    }

    # ======================================================
    # DRAW TEMPORAL CHUNKS
    # ======================================================

    for i in range(
        len(chunk_results)
    ):

        start = start_times[i]
        end = end_times[i]

        emotion = emotions[i]
        confidence = confidences[i]

        y_position = emotion_positions[
            emotion
        ]

        duration = end - start

        # --------------------------------------------------
        # Draw chunk as horizontal bar
        # --------------------------------------------------

        ax.barh(
            y_position,
            duration,
            left=start,
            height=0.55,
            alpha=0.75
        )

        # --------------------------------------------------
        # Chunk label
        # --------------------------------------------------

        label = (
            f"Chunk {i + 1}\n"
            f"{emotion} ({confidence:.1f}%)"
        )

        ax.text(
            start + duration / 2,
            y_position,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    # ======================================================
    # Y-AXIS
    # ======================================================

    ax.set_yticks(
        range(
            len(unique_emotions)
        )
    )

    ax.set_yticklabels(
        unique_emotions
    )

    ax.set_ylabel(
        "Predicted Emotion"
    )

    # ======================================================
    # X-AXIS
    # ======================================================

    max_time = max(
        end_times
    )

    ax.set_xlim(
        0,
        max_time
    )

    ax.set_xlabel(
        "Audio Time (seconds)"
    )

    # ======================================================
    # TITLE
    # ======================================================

    ax.set_title(
        "Temporal Emotion Analysis"
    )

    # ======================================================
    # GRID
    # ======================================================

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    # ======================================================
    # TIME MARKERS
    # ======================================================

    for start in start_times:

        ax.axvline(
            start,
            linestyle=":",
            alpha=0.25
        )

    # ======================================================
    # INVERT Y-AXIS
    # ======================================================

    ax.invert_yaxis()

    # ======================================================
    # LAYOUT
    # ======================================================

    plt.tight_layout()

    return fig
# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def plot_confusion_matrix(cm, class_names):

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        cm,
        cmap="Blues"
    )

    fig.colorbar(im)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()

    return fig
