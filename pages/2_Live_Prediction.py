"""
==========================================================
Speech Emotion Recognition
Live Prediction Page
Part 1A
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
    get_top_predictions
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
    page_title="Live Emotion Prediction",
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
# LOAD CNN MODEL
# ==========================================================

@st.cache_resource
def load_cnn():

    return load_model(
        "models/cnn_model.keras"
    )

model = load_cnn()

# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

@st.cache_resource
def load_encoder():

    return joblib.load(
        "saved_features/label_encoder.pkl"
    )

label_encoder = load_encoder()

class_names = label_encoder.classes_

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🎓 Research Demonstrator")

st.sidebar.success(
    "Speech Emotion Recognition"
)

st.sidebar.markdown("---")

st.sidebar.markdown("### CNN Model")

st.sidebar.write("Framework : TensorFlow")

st.sidebar.write("Frontend : Streamlit")

st.sidebar.write("Feature : Log-Mel Spectrogram")

st.sidebar.write(f"Emotion Classes : {len(class_names)}")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Only the CNN model is deployed for live prediction.

The remaining models (CNN-LSTM, Attention-LSTM, ProtoNet,
Tiny Transformer and TimeXer) were evaluated during
offline experimentation and are included for comparison
within this research.
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
The trained CNN model will automatically predict the
speaker's emotion in real time.

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

        "CNN (Deployed)",

        "CNN-LSTM",

        "Attention-LSTM",

        "ProtoNet",

        "Tiny Transformer",

        "TimeXer"

    ]

)

if selected_model != "CNN (Deployed)":

    st.info(
        f"{selected_model} was evaluated during the research.\n\n"
        "For live prediction, the deployed CNN model is used."
    )

st.divider()
# ==========================================================
# AUDIO INPUT
# ==========================================================

st.subheader("🎙 Audio Input")

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

        st.success("Audio uploaded successfully.")

        st.audio(uploaded_file)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(uploaded_file.read())

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

        # Save recording temporarily

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        temp_audio.write(recorded_audio["bytes"])
        temp_audio.close()

        st.success("✔ Recording completed successfully.")

        # Audio Player

        st.audio(recorded_audio["bytes"])

        # Duration

        duration = librosa.get_duration(path=temp_audio.name)

        st.info(
            f"Recording Duration : {duration:.2f} seconds"
        )

        col1, col2 = st.columns(2)

        # ------------------------------------------
        # RECORD AGAIN
        # ------------------------------------------

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

        # ------------------------------------------
        # ANALYZE
        # ------------------------------------------

        with col2:

            analyze = st.button(
                "✅ Analyze",
                use_container_width=True
            )

        if analyze:

            audio_path = temp_audio.name
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

with st.spinner("Loading audio..."):

    features = extract_all_features(audio_path)

signal = features["signal"]

sample_rate = features["sample_rate"]

duration = len(signal) / sample_rate

# ==========================================================
# AUDIO PLAYER
# ==========================================================

st.divider()

st.subheader("🔊 Audio Preview")

with open(audio_path, "rb") as f:

    st.audio(f.read())

# ==========================================================
# AUDIO INFORMATION
# ==========================================================

st.divider()

st.subheader("📋 Audio Information")

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

st.success("Audio loaded successfully.")

# ==========================================================
# FEATURE EXTRACTION SUMMARY
# ==========================================================

# ==========================================================
# FEATURE EXTRACTION SUMMARY
# ==========================================================

st.divider()

st.subheader("📈 Feature Extraction")

st.success("Features extracted successfully.")

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

st.table(summary_df)
# ==========================================================
# AUDIO VISUALIZATIONS
# ==========================================================

st.divider()

st.header("📈 Audio Feature Visualization")

# ==========================================================
# WAVEFORM
# ==========================================================

st.subheader("🌊 Waveform")

fig = plot_waveform(
    signal,
    sample_rate
)

st.pyplot(fig)

# ==========================================================
# SPECTROGRAM
# ==========================================================

st.subheader("🎼 Spectrogram")

fig = plot_spectrogram(
    signal,
    sample_rate
)

st.pyplot(fig)

# ==========================================================
# LOG-MEL SPECTROGRAM
# ==========================================================

st.subheader("🔥 Log-Mel Spectrogram")

fig = plot_logmel(
    signal,
    sample_rate
)

st.pyplot(fig)

# ==========================================================
# MFCC
# ==========================================================

st.subheader("🎵 MFCC")

fig = plot_mfcc(
    signal,
    sample_rate
)

st.pyplot(fig)

# ==========================================================
# CHROMA FEATURES
# ==========================================================

st.subheader("🎹 Chroma Features")

fig = plot_chroma(
    signal,
    sample_rate
)

st.pyplot(fig)

# ==========================================================
# ATTENTION HEATMAP (Demo)
# ==========================================================

st.subheader("🧠 Attention Heatmap")

st.info(
    "This visualization demonstrates the attention mechanism "
    "used during the research. It is a qualitative explanation "
    "and not generated by the deployed CNN."
)

fig = plot_attention_heatmap()

st.pyplot(fig)

# ==========================================================
# AUDIO STATISTICS
# ==========================================================

st.divider()

st.header("📊 Audio Statistics")

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

st.subheader("📋 Feature Summary")

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

st.success("✅ Audio preprocessing completed successfully.")
# ==========================================================
# LIVE PREDICTION
# ==========================================================

st.divider()

st.header("🧠 Emotion Prediction")

if st.button(
    "🚀 Predict Emotion",
    use_container_width=True
):

    with st.spinner("Running CNN model..."):

        result = predict_emotion(audio_path)

    predicted_emotion = result["emotion"]

    confidence = result["confidence"]

    probabilities = result["probabilities"]

    inference_time = result["inference_time"]

    # ======================================================
    # RESULTS
    # ======================================================

    st.success("Prediction Completed Successfully!")

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

        "Happy":"😊",
        "Sad":"😢",
        "Angry":"😠",
        "Fear":"😨",
        "Disgust":"🤢",
        "Surprise":"😲",
        "Neutral":"😐",
        "Calm":"😌"

    }

    st.markdown("---")

    st.markdown(
        f"# {emoji.get(predicted_emotion,'🎤')} {predicted_emotion}"
    )

    st.markdown(
        f"### Confidence : **{confidence:.2f}%**"
    )

    # ======================================================
    # PROBABILITY TABLE
    # ======================================================

    st.divider()

    st.subheader("📊 Emotion Probabilities")

    prob_df = pd.DataFrame({

        "Emotion": class_names,

        "Probability (%)": probabilities * 100

    })

    st.dataframe(

        prob_df.style.format(

            {"Probability (%)":"{:.2f}"}

        ),

        use_container_width=True

    )

    # ======================================================
    # BAR CHART
    # ======================================================

    st.subheader("📈 Prediction Confidence")

    fig = plot_probability_chart(

        probabilities,

        class_names

    )

    st.pyplot(fig)

    # ======================================================
    # TOP 3 PREDICTIONS
    # ======================================================

    st.subheader("🏆 Top 3 Predictions")

    top3 = np.argsort(probabilities)[::-1][:3]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(

            f"""

🥇 {class_names[top3[0]]}

{probabilities[top3[0]]*100:.2f}%

"""

        )

    with col2:

        st.info(

            f"""

🥈 {class_names[top3[1]]}

{probabilities[top3[1]]*100:.2f}%

"""

        )

    with col3:

        st.warning(

            f"""

🥉 {class_names[top3[2]]}

{probabilities[top3[2]]*100:.2f}%

"""

        )

    # ======================================================
    # INTERPRETATION
    # ======================================================

    st.divider()

    st.subheader("📖 Interpretation")

    st.write(

        f"""
The speech sample has been classified as **{predicted_emotion}**.

The deployed CNN extracted Log-Mel spectrogram features and
performed inference in **{inference_time:.3f} seconds**.

Prediction confidence is **{confidence:.2f}%**.

This demonstrates the real-time deployment of the Speech
Emotion Recognition model developed during this research.
"""

    )

    # ======================================================
    # SAVE HISTORY
    # ======================================================

    if "prediction_history" not in st.session_state:

        st.session_state.prediction_history = []

    st.session_state.prediction_history.append({

        "Emotion": predicted_emotion,

        "Confidence (%)": round(confidence,2),

        "Inference Time (s)": round(inference_time,3)

    })
    
    