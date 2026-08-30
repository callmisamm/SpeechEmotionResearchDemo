from utils.feature_utils import split_audio_into_chunks


# ==========================================================
# GET AUDIO PATH
# ==========================================================

import sys

if len(sys.argv) < 2:
    print("Please provide an audio file path.")
    print('Example: python test_chunking.py "C:\\path\\audio.wav"')
    sys.exit(1)

audio_path = sys.argv[1]


# ==========================================================
# SPLIT AUDIO
# ==========================================================

chunks, sample_rate = split_audio_into_chunks(audio_path)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("\n==========================================")
print("CHUNKING TEST")
print("==========================================")

print(f"Audio: {audio_path}")
print(f"Sample Rate: {sample_rate} Hz")
print(f"Number of Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):

    duration = len(chunk) / sample_rate

    print(
        f"Chunk {i + 1}: "
        f"{len(chunk)} samples | "
        f"{duration:.2f} seconds"
    )

print("==========================================")