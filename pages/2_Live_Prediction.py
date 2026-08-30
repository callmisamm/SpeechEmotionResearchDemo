
"""
==========================================================
Speech Emotion Recognition
Live Prediction Page
CNN Version 2 Deployment
==========================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import librosa
import tempfile
import joblib
import streamlit as st
import pandas as pd
import numpy as np

from streamlit_mic_recorder import mic_recorder
from tensorflow.keras.models import load_model


# ==========================================================
# CUSTOM MODULES
# ==========================================================

from utils.feature_utils import load_audio

from utils.audio_features import (
    extract_all_features
)

from utils.prediction import (
    predict_emotion,
    emotion_emoji,
    get_top_predictions,
    predict_emotion_by_chunks,
    calculate_chunk_statistics
)

from utils.visualizations import (
    plot_waveform,
    plot_spectrogram,
    plot_logmel,
    plot_mfcc,
    plot_chroma,
    plot_attention_heatmap,
    plot_probability_chart
)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Live Emotion Prediction - CNN V2",
    page_icon="🎤",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
"""
<style>

.main{
    background:#0E1117;
}

.title{
    color:#00C8FF;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    color:#DDDDDD;
    font-size:18px;
}

.card{
    background:#1E1E1E;
    padding:18px;
    border-radius:12px;
    border:1px solid #333333;
}

</style>
""",
unsafe_allow_html=True
)


# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

MODEL_PATH = "models/cnn_model_v2.keras"

LABEL_PATH = "saved_features/label_encoder.pkl"

MODEL_NAME = "CNN Version 2"

TEST_ACCURACY = 69.44


# ==========================================================
# LOAD CNN V2 MODEL
# ==========================================================

@st.cache_resource
def load_cnn():

    model = load_model(
        MODEL_PATH
    )

    return model


model = load_cnn()


# ==========================================================
# VERIFY MODEL
# ==========================================================

if model.input_shape != (None, 128, 128, 1):

    st.error(
        f"Unexpected CNN V2 input shape: {model.input_shape}"
    )

    st.stop()


if model.output_shape[-1] != 8:

    st.error(
        f"Unexpected CNN V2 output shape: {model.output_shape}"
    )

    st.stop()


# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

@st.cache_resource
def load_encoder():

    return joblib.load(
        LABEL_PATH
    )


label_encoder = load_encoder()

class_names = label_encoder.classes_


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "🎓 Research Demonstrator"
)

st.sidebar.success(
    "Speech Emotion Recognition"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### CNN Version 2"
)

st.sidebar.write(
    "Framework : TensorFlow"
)

st.sidebar.write(
    "Frontend : Streamlit"
)

st.sidebar.write(
    "Feature : Log-Mel Spectrogram"
)

st.sidebar.write(
    "Input Size : 128 × 128 × 1"
)

st.sidebar.write(
    f"Emotion Classes : {len(class_names)}"
)

st.sidebar.write(
    "Dataset : RAVDESS"
)

st.sidebar.write(
    "Test Accuracy : 69.44%"
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
The deployed model is CNN Version 2.

CNN Version 2 achieved 69.44% test
accuracy on the held-out RAVDESS test set.

The remaining models (CNN-LSTM, Attention-LSTM,
ProtoNet, Tiny Transformer and TimeXer) were
evaluated during offline experimentation and
are included for research comparison.
"""
)


# ==========================================================
# PAGE HEADER
# ==========================================================

st.markdown(
"""
<div class="title">

🎤 Live Speech Emotion Prediction

</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="subtitle">

Record your voice or upload a WAV file.
The trained CNN Version 2 model will analyze the
speech and predict one of eight emotional categories.

</div>
""",
unsafe_allow_html=True
)

st.divider()


# ==========================================================
# MODEL SELECTION
# ==========================================================

selected_model = st.selectbox(

    "Select Research Model",

    [
        "CNN Version 2 (Deployed)",
        "CNN-LSTM",
        "Attention-LSTM",
        "ProtoNet",
        "Tiny Transformer",
        "TimeXer"
    ]

)


if selected_model != "CNN Version 2 (Deployed)":

    st.info(
        f"{selected_model} was evaluated during the research.\n\n"
        "For live prediction, CNN Version 2 is used."
    )


st.divider()


# ==========================================================
# AUDIO INPUT
# ==========================================================

st.subheader(
    "🎙 Audio Input"
)

tab1, tab2 = st.tabs(
    [
        "📁 Upload Audio",
        "🎤 Record Voice"
    ]
)


audio_path = None


# ==========================================================
# TAB 1 : UPLOAD AUDIO
# ==========================================================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload a WAV file",
        type=["wav"]
    )

    if uploaded_file is not None:

        st.success(
            "Audio uploaded successfully."
        )

        st.audio(
            uploaded_file
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(
                uploaded_file.getbuffer()
            )

            audio_path = tmp.name


# ==========================================================
# TAB 2 : RECORD MICROPHONE
# ==========================================================

with tab2:

    st.info(
        "Press Start Recording, speak for a few seconds, then Stop."
    )

    recorded_audio = mic_recorder(

        start_prompt="🎙 Start Recording",

        stop_prompt="⏹ Stop Recording",

        just_once=False,

        use_container_width=True,

        key="live_prediction_recorder"

    )


    if recorded_audio:

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        temp_audio.write(
            recorded_audio["bytes"]
        )

        temp_audio.close()

        audio_path = temp_audio.name


        st.success(
            "✔ Recording completed successfully."
        )


        st.audio(
            recorded_audio["bytes"]
        )


        # --------------------------------------------------
        # DURATION
        # --------------------------------------------------

        try:

            y, sr = librosa.load(
                audio_path,
                sr=22050
            )

            duration = len(y) / sr

            st.write(
                f"Duration: {duration:.2f} seconds"
            )

        except Exception as e:

            st.error(
                f"Unable to read recorded audio: {e}"
            )


        # --------------------------------------------------
        # RECORD AGAIN / ANALYZE
        # --------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            if st.button(
                "🔄 Record Again",
                use_container_width=True
            ):

                st.session_state.pop(
                    "live_prediction_recorder",
                    None
                )

                st.rerun()


        with col2:

            analyze = st.button(
                "✅ Analyze",
                use_container_width=True
            )


        if analyze:

            if audio_path is None:

                st.warning(
                    "Please record an audio sample first."
                )

                st.stop()

            st.info(
                "Analyzing recorded audio using CNN Version 2..."
            )


# ==========================================================
# WAIT UNTIL AUDIO EXISTS
# ==========================================================

if audio_path is None:

    st.warning(
        "Please upload or record an audio sample."
    )

    st.stop()


# ==========================================================
# LOAD AUDIO
# ==========================================================

with st.spinner(
    "Loading audio..."
):

    features = extract_all_features(
        audio_path
    )


signal = features["signal"]

sample_rate = features["sample_rate"]

duration = len(signal) / sample_rate


# ==========================================================
# AUDIO PLAYER
# ==========================================================

st.divider()

st.subheader(
    "🔊 Audio Preview"
)

with open(
    audio_path,
    "rb"
) as f:

    st.audio(
        f.read()
    )


# ==========================================================
# AUDIO INFORMATION
# ==========================================================

st.divider()

st.subheader(
    "📋 Audio Information"
)

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Sample Rate",
        f"{sample_rate} Hz"
    )


with c2:

    st.metric(
        "Duration",
        f"{duration:.2f} sec"
    )


with c3:

    st.metric(
        "Samples",
        len(signal)
    )


st.success(
    "Audio loaded successfully."
)


# ==========================================================
# FEATURE EXTRACTION SUMMARY
# ==========================================================

st.divider()

st.subheader(
    "📈 Feature Extraction"
)

st.success(
    "Features extracted successfully."
)


summary_df = pd.DataFrame({

    "Feature": [

        "Sample Rate",
        "Duration",
        "Samples",
        "MFCC Shape",
        "Log-Mel Shape",
        "Chroma Shape"

    ],

    "Value": [

        f"{sample_rate} Hz",

        f"{duration:.2f} sec",

        len(signal),

        str(features["mfcc"].shape),

        str(features["logmel"].shape),

        str(features["chroma"].shape)

    ]

})


st.table(
    summary_df
)


# ==========================================================
# AUDIO VISUALIZATIONS
# ==========================================================

st.divider()

st.header(
    "📈 Audio Feature Visualization"
)


# ==========================================================
# WAVEFORM
# ==========================================================

st.subheader(
    "🌊 Waveform"
)

fig = plot_waveform(
    signal,
    sample_rate
)

st.pyplot(fig)


# ==========================================================
# SPECTROGRAM
# ==========================================================

st.subheader(
    "🎼 Spectrogram"
)

fig = plot_spectrogram(
    signal,
    sample_rate
)

st.pyplot(fig)


# ==========================================================
# LOG-MEL SPECTROGRAM
# ==========================================================

st.subheader(
    "🔥 Log-Mel Spectrogram"
)

fig = plot_logmel(
    signal,
    sample_rate
)

st.pyplot(fig)


# ==========================================================
# MFCC
# ==========================================================

st.subheader(
    "🎵 MFCC"
)

fig = plot_mfcc(
    signal,
    sample_rate
)

st.pyplot(fig)


# ==========================================================
# CHROMA
# ==========================================================

st.subheader(
    "🎹 Chroma Features"
)

fig = plot_chroma(
    signal,
    sample_rate
)

st.pyplot(fig)


# ==========================================================
# ATTENTION HEATMAP
# ==========================================================

st.subheader(
    "🧠 Attention Heatmap"
)

st.info(
"""
This visualization demonstrates the attention mechanism
used during the research.

It is a qualitative visualization and is NOT generated
by the deployed CNN Version 2 model.
"""
)

fig = plot_attention_heatmap()

st.pyplot(fig)


# ==========================================================
# AUDIO STATISTICS
# ==========================================================

st.divider()

st.header(
    "📊 Audio Statistics"
)

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Zero Crossing Rate",
        f"{np.mean(features['zcr']):.4f}"
    )


with c2:

    st.metric(
        "RMS Energy",
        f"{np.mean(features['rms']):.4f}"
    )


with c3:

    st.metric(
        "Spectral Centroid",
        f"{np.mean(features['centroid']):.2f} Hz"
    )


# ==========================================================
# FEATURE SUMMARY TABLE
# ==========================================================

st.divider()

st.subheader(
    "📋 Feature Summary"
)


feature_df = pd.DataFrame({

    "Feature": [

        "Signal Length",
        "Sample Rate",
        "MFCC",
        "Log-Mel",
        "Chroma",
        "Zero Crossing Rate",
        "RMS",
        "Spectral Centroid"

    ],

    "Value": [

        len(signal),

        sample_rate,

        str(features["mfcc"].shape),

        str(features["logmel"].shape),

        str(features["chroma"].shape),

        f"{np.mean(features['zcr']):.4f}",

        f"{np.mean(features['rms']):.4f}",

        f"{np.mean(features['centroid']):.2f}"

    ]

})


st.dataframe(
    feature_df,
    use_container_width=True
)


st.success(
    "✅ Audio preprocessing completed successfully."
)


# ==========================================================
# LIVE PREDICTION
# ==========================================================

st.divider()

st.header(
    "🧠 CNN Version 2 Emotion Prediction"
)


if st.button(
    "🚀 Predict Emotion",
    use_container_width=True
):

    with st.spinner(
        "Running CNN Version 2 model..."
    ):

        result = predict_emotion(
            audio_path
        )


    predicted_emotion = result["emotion"]

    confidence = result["confidence"]

    probabilities = result["probabilities"]

    inference_time = result["inference_time"]


    # ======================================================
    # RESULTS
    # ======================================================

    st.success(
        "Prediction Completed Successfully!"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Detected Emotion",
            predicted_emotion
        )


    with c2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    with c3:

        st.metric(
            "Inference Time",
            f"{inference_time:.3f} sec"
        )


    # ======================================================
    # EMOJI
    # ======================================================

    emoji = {

        "Happy": "😊",
        "Sad": "😢",
        "Angry": "😠",
        "Fear": "😨",
        "Disgust": "🤢",
        "Surprise": "😲",
        "Neutral": "😐",
        "Calm": "😌"

    }


    st.markdown("---")


    st.markdown(
        f"# {emoji.get(predicted_emotion, '🎤')} "
        f"{predicted_emotion}"
    )


    st.markdown(
        f"### Confidence : **{confidence:.2f}%**"
    )


    # ======================================================
    # MODEL PERFORMANCE
    # ======================================================

    st.info(
        f"""
**Deployed Model:** CNN Version 2

**RAVDESS Test Accuracy:** {TEST_ACCURACY:.2f}%

The reported test accuracy was obtained on the
held-out test set of 216 samples.
"""
    )


    # ======================================================
    # PROBABILITY TABLE
    # ======================================================

    st.divider()

    st.subheader(
        "📊 Emotion Probabilities"
    )


    prob_df = pd.DataFrame({

        "Emotion": class_names,

        "Probability (%)": probabilities * 100

    })


    st.dataframe(

        prob_df.style.format(
            {
                "Probability (%)": "{:.2f}"
            }
        ),

        use_container_width=True

    )


    # ======================================================
    # BAR CHART
    # ======================================================

    st.subheader(
        "📈 Prediction Confidence"
    )


    fig = plot_probability_chart(

        probabilities,

        class_names

    )

    st.pyplot(fig)


    # ======================================================
    # TOP 3 PREDICTIONS
    # ======================================================

    st.subheader(
        "🏆 Top 3 Predictions"
    )


    top3 = np.argsort(
        probabilities
    )[::-1][:3]


    col1, col2, col3 = st.columns(3)


    with col1:

        st.success(

            f"""
🥇 {class_names[top3[0]]}

{probabilities[top3[0]] * 100:.2f}%
"""
        )


    with col2:

        st.info(

            f"""
🥈 {class_names[top3[1]]}

{probabilities[top3[1]] * 100:.2f}%
"""
        )


    with col3:

        st.warning(

            f"""
🥉 {class_names[top3[2]]}

{probabilities[top3[2]] * 100:.2f}%
"""
        )


    # ======================================================
    # INTERPRETATION
    # ======================================================

    st.divider()

    st.subheader(
        "📖 Interpretation"
    )


    st.write(

        f"""
The speech sample has been classified as
**{predicted_emotion}**.

The deployed CNN Version 2 extracts
Log-Mel spectrogram features and performs
eight-class emotion classification.

Inference was completed in
**{inference_time:.3f} seconds**.

The predicted class has a probability of
**{confidence:.2f}%**.

The CNN Version 2 achieved **69.44% test accuracy**
on the held-out RAVDESS test set.
"""
    )


    # ======================================================
    # SAVE HISTORY
    # ======================================================

    if "prediction_history" not in st.session_state:

        st.session_state.prediction_history = []


    st.session_state.prediction_history.append({

        "Model":
            "CNN Version 2",

        "Emotion":
            predicted_emotion,

        "Confidence (%)":
            round(
                confidence,
                2
            ),

        "Inference Time (s)":
            round(
                inference_time,
                3
            )

    })


# ==========================================================
# CHUNK-LEVEL EMOTION ANALYSIS
# ==========================================================

st.divider()

st.header(
    "🧩 Chunk-Level Emotion Analysis"
)


st.write(
"""
The uploaded or recorded audio is divided into complete
3-second chunks. Each chunk is independently analyzed
using the deployed CNN Version 2 model.
"""
)


if st.button(
    "🔍 Analyze Every 3-Second Chunk",
    use_container_width=True
):

    with st.spinner(
        "Analyzing audio chunks using CNN Version 2..."
    ):

        chunk_results = predict_emotion_by_chunks(
            audio_path
        )


    if not chunk_results:

        st.warning(
            "The audio does not contain a complete "
            "3-second chunk."
        )


    else:

        st.success(
            f"Analysis completed: "
            f"{len(chunk_results)} complete "
            f"3-second chunks detected."
        )


        # --------------------------------------------------
        # CHUNK RESULTS TABLE
        # --------------------------------------------------

        st.subheader(
            "📋 Chunk-by-Chunk Predictions"
        )


        chunk_table = []


        for result in chunk_results:

            chunk_table.append({

                "Chunk":
                    result["chunk"],

                "Time":
                    f"{result['start_time']:.1f} – "
                    f"{result['end_time']:.1f} sec",

                "Emotion":
                    result["emotion"],

                "Confidence (%)":
                    round(
                        result["confidence"],
                        2
                    )

            })


        chunk_df = pd.DataFrame(
            chunk_table
        )


        st.dataframe(
            chunk_df,
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------
        # STATISTICS
        # --------------------------------------------------

        statistics = calculate_chunk_statistics(
            chunk_results
        )


        st.divider()

        st.subheader(
            "📊 Emotion Statistics"
        )


        statistics_rows = []


        for emotion, count in statistics[
            "emotion_counts"
        ].items():

            percentage = statistics[
                "emotion_percentages"
            ][emotion]


            statistics_rows.append({

                "Emotion":
                    emotion,

                "Number of Chunks":
                    count,

                "Percentage of Chunks":
                    f"{percentage:.2f}%"

            })


        statistics_df = pd.DataFrame(
            statistics_rows
        )


        st.dataframe(
            statistics_df,
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------
        # SUMMARY METRICS
        # --------------------------------------------------

        st.subheader(
            "📌 Overall Summary"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Chunks",
                len(chunk_results)
            )


        with col2:

            st.metric(
                "Dominant Emotion",
                statistics[
                    "dominant_emotion"
                ]
            )


        with col3:

            st.metric(
                "Average Confidence",
                f"{statistics['average_confidence']:.2f}%"
            )


        # --------------------------------------------------
        # EMOTION DISTRIBUTION
        # --------------------------------------------------

        st.subheader(
            "📈 Emotion Distribution"
        )


        distribution_df = pd.DataFrame({

            "Emotion":
                list(
                    statistics[
                        "emotion_percentages"
                    ].keys()
                ),

            "Percentage":
                list(
                    statistics[
                        "emotion_percentages"
                    ].values()
                )

        })


        distribution_df = (
            distribution_df
            .set_index("Emotion")
        )


        st.bar_chart(
            distribution_df
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
"""
<center>

### 🎤 Speech Emotion Recognition Research Demonstrator

**CNN Version 2 — Live Deployment**

MPhil Thesis Demonstrator

Department of Computer Science

Powered by TensorFlow • Streamlit • Librosa

</center>
""",
unsafe_allow_html=True
)

