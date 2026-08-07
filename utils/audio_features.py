"""
Audio Feature Extraction Utilities
"""

import librosa
import numpy as np


def extract_all_features(audio_path):
    """
    Extract all audio features required by the Streamlit app.
    """

    signal, sr = librosa.load(audio_path, sr=22050)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40
    )

    # Log-Mel
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr
    )

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal)[0]

    # RMS
    rms = librosa.feature.rms(y=signal)[0]

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )[0]

    return {
        "signal": signal,
        "sample_rate": sr,
        "mfcc": mfcc,
        "logmel": logmel,
        "chroma": chroma,
        "zcr": zcr,
        "rms": rms,
        "centroid": centroid
    }