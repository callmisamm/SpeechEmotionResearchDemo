"""
Audio Augmentation Utilities
Speech Emotion Recognition

These augmentations are applied ONLY to the training data.
"""

import numpy as np
import librosa

# ==========================================================
# AUDIO PARAMETERS
# ==========================================================

SAMPLE_RATE = 22050

# ==========================================================
# ADD RANDOM GAUSSIAN NOISE
# ==========================================================

def add_noise(signal):

    noise_factor = np.random.uniform(0.002, 0.008)

    noise = np.random.randn(len(signal))

    augmented = signal + noise_factor * noise

    return np.clip(augmented, -1.0, 1.0)


# ==========================================================
# RANDOM PITCH SHIFT
# ==========================================================

def pitch_shift(signal, sr=SAMPLE_RATE):

    n_steps = np.random.uniform(-2.0, 2.0)

    return librosa.effects.pitch_shift(
        y=signal,
        sr=sr,
        n_steps=n_steps
    )


# ==========================================================
# RANDOM TIME STRETCH
# ==========================================================

def time_stretch(signal):

    rate = np.random.uniform(0.90, 1.10)

    stretched = librosa.effects.time_stretch(
        y=signal,
        rate=rate
    )

    return stretched


# ==========================================================
# RANDOM VOLUME SCALING
# ==========================================================

def volume_scale(signal):

    gain = np.random.uniform(0.8, 1.2)

    signal = signal * gain

    return np.clip(signal, -1.0, 1.0)


# ==========================================================
# RANDOM TIME SHIFT
# ==========================================================

def time_shift(signal):

    shift = np.random.randint(
        int(0.05 * len(signal)),
        int(0.15 * len(signal))
    )

    if np.random.rand() > 0.5:
        shift = -shift

    return np.roll(signal, shift)