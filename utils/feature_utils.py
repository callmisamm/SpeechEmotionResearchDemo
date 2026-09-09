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
# SPLIT AUDIO INTO 3-SECOND CHUNKS WITH 50% OVERLAP
# ==========================================================

def split_audio_into_chunks(path):
    """
    Split complete audio into 3-second chunks
    with 50% overlap.

    Chunk duration:
        3 seconds

    Overlap:
        50%

    Hop size:
        1.5 seconds

    Only complete 3-second chunks are returned.
    Any remaining audio shorter than 3 seconds
    at the end is ignored.

    Returns:
        chunks: list of numpy arrays
        sample_rate: audio sample rate
        chunk_times: list of (start_time, end_time)
    """

    # ------------------------------------------------------
    # Load complete audio
    # ------------------------------------------------------

    signal, sr = librosa.load(
        path,
        sr=SAMPLE_RATE
    )

    # ------------------------------------------------------
    # Remove silence from beginning/end
    # ------------------------------------------------------

    signal, _ = librosa.effects.trim(
        signal,
        top_db=20
    )

    # ------------------------------------------------------
    # Normalize amplitude
    # ------------------------------------------------------

    signal = librosa.util.normalize(
        signal
    )

    # ------------------------------------------------------
    # Chunk configuration
    # ------------------------------------------------------

    chunk_duration = 3.0

    overlap = 0.50

    chunk_samples = int(
        SAMPLE_RATE * chunk_duration
    )

    hop_samples = int(
        chunk_samples * (1 - overlap)
    )

    # ------------------------------------------------------
    # Prepare output
    # ------------------------------------------------------

    chunks = []

    chunk_times = []

    # ------------------------------------------------------
    # Generate overlapping chunks
    # ------------------------------------------------------

    start = 0

    while start + chunk_samples <= len(signal):

        end = start + chunk_samples

        chunk = signal[start:end]

        # --------------------------------------------------
        # Ensure complete 3-second chunk
        # --------------------------------------------------

        if len(chunk) == chunk_samples:

            chunks.append(
                chunk.astype(np.float32)
            )

            # Convert sample positions to seconds
            start_time = (
                start / SAMPLE_RATE
            )

            end_time = (
                end / SAMPLE_RATE
            )

            chunk_times.append(
                (
                    start_time,
                    end_time
                )
            )

        # --------------------------------------------------
        # Move forward by 1.5 seconds
        # --------------------------------------------------

        start += hop_samples

    return (
        chunks,
        sr,
        chunk_times
    )