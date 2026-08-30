from utils.prediction import (
    predict_emotion_by_chunks,
    calculate_chunk_statistics
)

import sys


# ==========================================================
# AUDIO PATH
# ==========================================================

if len(sys.argv) < 2:

    print(
        'Usage: python test_statistics.py "audio.wav"'
    )

    sys.exit(1)


audio_path = sys.argv[1]


# ==========================================================
# CHUNK PREDICTION
# ==========================================================

print("\n==========================================")
print("CHUNK STATISTICS TEST")
print("==========================================")

results = predict_emotion_by_chunks(
    audio_path
)


# ==========================================================
# CALCULATE STATISTICS
# ==========================================================

statistics = calculate_chunk_statistics(
    results
)


# ==========================================================
# DISPLAY CHUNK RESULTS
# ==========================================================

print("\nChunk Results")
print("------------------------------------------")

for result in results:

    print(
        f"Chunk {result['chunk']} | "
        f"{result['start_time']:.1f}s - "
        f"{result['end_time']:.1f}s | "
        f"{result['emotion']} | "
        f"{result['confidence']:.2f}%"
    )


# ==========================================================
# DISPLAY STATISTICS
# ==========================================================

print("\nEmotion Statistics")
print("------------------------------------------")

for emotion, count in statistics[
    "emotion_counts"
].items():

    percentage = statistics[
        "emotion_percentages"
    ][emotion]

    print(
        f"{emotion}: "
        f"{count} chunks "
        f"({percentage:.2f}%)"
    )


# ==========================================================
# SUMMARY
# ==========================================================

print("\n==========================================")

print(
    f"Total Chunks: {len(results)}"
)

print(
    f"Dominant Emotion: "
    f"{statistics['dominant_emotion']}"
)

print(
    f"Dominant Percentage: "
    f"{statistics['dominant_percentage']:.2f}%"
)

print(
    f"Average Confidence: "
    f"{statistics['average_confidence']:.2f}%"
)

print("==========================================\n")