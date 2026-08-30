import librosa
import soundfile as sf
import numpy as np


# ==========================================================
# INPUT / OUTPUT
# ==========================================================

input_path = r"C:\Users\PMLS\Desktop\Thesis\Ravdess\Actor_04\03-01-02-01-01-01-04.wav"

output_path = "test_long_audio.wav"


# ==========================================================
# LOAD AUDIO
# ==========================================================

signal, sample_rate = librosa.load(
    input_path,
    sr=22050
)


# ==========================================================
# REPEAT AUDIO
# ==========================================================

long_signal = np.tile(
    signal,
    4
)


# ==========================================================
# SAVE
# ==========================================================

sf.write(
    output_path,
    long_signal,
    sample_rate
)


# ==========================================================
# INFORMATION
# ==========================================================

duration = len(long_signal) / sample_rate

print("\n==========================================")
print("LONG TEST AUDIO CREATED")
print("==========================================")
print(f"Sample Rate: {sample_rate} Hz")
print(f"Duration: {duration:.2f} seconds")
print(f"Output: {output_path}")
print("==========================================\n")