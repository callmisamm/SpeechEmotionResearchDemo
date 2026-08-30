from utils.prediction import predict_emotion_by_chunks

import sys


# ==========================================================
# GET AUDIO PATH
# ==========================================================

if len(sys.argv) < 2:

    print("Please provide an audio file path.")

    print(
        'Example: python test_chunk_prediction.py "C:\\path\\audio.wav"'
    )

    sys.exit(1)


audio_path = sys.argv[1]


# ==========================================================
# PREDICT
# ==========================================================

print("\n==========================================")
print("CHUNK-LEVEL EMOTION PREDICTION")
print("==========================================")

results = predict_emotion_by_chunks(audio_path)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

for result in results:

    print(
        f"Chunk {result['chunk']} | "
        f"{result['start_time']:.1f}s - "
        f"{result['end_time']:.1f}s | "
        f"{result['emotion']} | "
        f"{result['confidence']:.2f}%"
    )


print("==========================================")
print(
    f"Total Chunks: {len(results)}"
)
print("==========================================\n")