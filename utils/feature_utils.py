"""
Feature Extraction Utilities
Speech Emotion Recognition
Improved Version
"""

import librosa
import numpy as np
import cv2

# ==========================================================
# AUDIO PARAMETERS
# ==========================================================

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

# ==========================================================
# LOAD AUDIO
# ==========================================================

def load_audio(path):

    signal, sr = librosa.load(
        path,
        sr=SAMPLE_RATE
    )

    # Remove silence
    signal, _ = librosa.effects.trim(
        signal,
        top_db=20
    )

    # Normalize amplitude
    signal = librosa.util.normalize(signal)

    # Pad or crop to fixed length
    if len(signal) > SAMPLES:
        signal = signal[:SAMPLES]
    else:
        signal = np.pad(
            signal,
            (0, SAMPLES - len(signal)),
            mode="constant"
        )

    return signal


# ==========================================================
# LOG-MEL FEATURE EXTRACTION
# ==========================================================

def extract_log_mel(signal):

    # Pre-emphasis filter
    signal = np.append(
        signal[0],
        signal[1:] - 0.97 * signal[:-1]
    )

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,

        n_fft=2048,
        win_length=1024,
        hop_length=256,

        n_mels=128,

        fmin=20,
        fmax=8000,

        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Standardization
    log_mel = (
        log_mel - np.mean(log_mel)
    ) / (
        np.std(log_mel) + 1e-8
    )

    # Resize to CNN input
    log_mel = cv2.resize(
        log_mel,
        (128, 128),
        interpolation=cv2.INTER_CUBIC
    )

    # Min-Max Normalization
    log_mel = (
        log_mel - log_mel.min()
    ) / (
        log_mel.max() - log_mel.min() + 1e-8
    )

    return log_mel.astype(np.float32)

# ==========================================================
# SPLIT AUDIO INTO COMPLETE 3-SECOND CHUNKS
# ==========================================================

def split_audio_into_chunks(path):
    """
    Load the complete audio and split it into complete
    3-second chunks.

    Only complete 3-second chunks are returned.
    Any remaining audio shorter than 3 seconds is ignored.

    Returns:
        chunks: list of numpy arrays
        sample_rate: audio sample rate
    """

    signal, sr = librosa.load(
        path,
        sr=SAMPLE_RATE
    )

    # Remove silence from beginning/end
    signal, _ = librosa.effects.trim(
        signal,
        top_db=20
    )

    # Normalize amplitude
    signal = librosa.util.normalize(signal)

    # Number of complete 3-second chunks
    num_chunks = len(signal) // SAMPLES

    chunks = []

    for i in range(num_chunks):

        start = i * SAMPLES
        end = start + SAMPLES

        chunk = signal[start:end]

        # Every returned chunk must be exactly 3 seconds
        if len(chunk) == SAMPLES:

            chunks.append(
                chunk.astype(np.float32)
            )

    return chunks, sr