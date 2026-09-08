
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

import os
import tempfile
import time

import librosa
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_mic_recorder import mic_recorder
from tensorflow.keras.models import load_model


# ==========================================================
# CUSTOM MODULES
# ==========================================================

from utils.feature_utils import load_audio
from utils.audio_features import extract_all_features

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
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Live Speech Emotion Prediction",
    page_icon="🎙️",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .prediction-emotion {
        font-size: 42px;
        font-weight: 700;
    }

    .prediction-confidence {
        font-size: 24px;
        margin-top: 8px;
    }

    .emotion-loader-wrapper {
        text-align: center;
        padding: 25px 10px;
    }

    .emotion-loader {
        position: relative;
        width: 100%;
        height: 90px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .emotion {
        position: absolute;
        font-size: 60px;
        opacity: 0;
    }

    .emotion:nth-child(1) {
        animation: showEmoji 2.4s infinite;
        animation-delay: 0s;
    }

    .emotion:nth-child(2) {
        animation: showEmoji 2.4s infinite;
        animation-delay: 0.3s;
    }

    .emotion:nth-child(3) {
        animation: showEmoji 2.4s infinite;
        animation-delay: 0.6s;
    }

    .emotion:nth-child(4) {
        animation: showEmoji 2.4s infinite;
        animation-delay: 0.9s;
    }

    .emotion:nth-child(5) {
        animation: showEmoji 2.4s infinite;
        animation-delay: 1.2s;
    }

    .emotion:nth-child(6) {
        animation: showEmoji 2.4s infinite;
        animation-delay: 1.5s;
    }

    .emotion:nth-child(7) {
        animation: showEmoji 2.4s infinite;
        animation-delay: 1.8s;
    }

    .emotion:nth-child(8) {
        animation: showEmoji 2.4s infinite;
        animation-delay: 2.1s;
    }

    @keyframes showEmoji {

        0% {
            opacity: 0;
            transform: scale(0.7);
        }

        8% {
            opacity: 1;
            transform: scale(1.15);
        }

        16% {
            opacity: 1;
            transform: scale(1);
        }

        24% {
            opacity: 0;
            transform: scale(0.7);
        }

        100% {
            opacity: 0;
            transform: scale(0.7);
        }

    }

    .loading-text {
        font-size: 18px;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# EMOTION LOADER
# ==========================================================

def show_emotion_loader(placeholder):

    placeholder.markdown(
        """
        <div class="emotion-loader-wrapper">

            <div class="emotion-loader">

                <span class="emotion">😐</span>
                <span class="emotion">😌</span>
                <span class="emotion">😊</span>
                <span class="emotion">😢</span>
                <span class="emotion">😠</span>
                <span class="emotion">😨</span>
                <span class="emotion">🤢</span>
                <span class="emotion">😲</span>

            </div>

            <div class="loading-text">
                Analyzing speech and extracting emotional features...
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

MODEL_PATH = "models/cnn_model_v2.keras"

LABEL_PATH = (
    "saved_features/label_encoder.pkl"
)

MODEL_NAME = "CNN Version 2"

TEST_ACCURACY = 69.44


# ==========================================================
# LOAD CNN MODEL
# ==========================================================

@st.cache_resource
def load_cnn():

    model = load_model(
        MODEL_PATH
    )

    return model


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model = load_cnn()

except Exception as e:

    st.error(
        "Unable to load CNN Version 2 model."
    )

    st.exception(e)

    st.stop()


# ==========================================================
# VERIFY MODEL INPUT
# ==========================================================

if model.input_shape != (
    None,
    128,
    128,
    1
):

    st.error(
        f"""
        CNN Version 2 expects input shape
        `(None, 128, 128, 1)`.

        Current model input shape:
        `{model.input_shape}`
        """
    )

    st.stop()


# ==========================================================
# VERIFY OUTPUT CLASSES
# ==========================================================

if model.output_shape[-1] != 8:

    st.error(
        f"""
        CNN Version 2 should output 8 emotion classes.

        Current output shape:
        `{model.output_shape}`
        """
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


try:

    label_encoder = load_encoder()

    class_names = label_encoder.classes_

except Exception as e:

    st.error(
        "Unable to load label encoder."
    )

    st.exception(e)

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("🎙️ SER Demonstrator")

    st.markdown(
        """
        ### Model Information

        **Model:** CNN Version 2

        **Framework:** TensorFlow / Keras

        **Feature:** Log-Mel Spectrogram

        **Input Size:** 128 × 128 × 1

        **Classes:** 8

        **Dataset:** RAVDESS

        **Training Dataset:** 1440 Audio Files

        **Test Accuracy:** 69.44%
        """
    )

    st.divider()

    st.markdown(
        """
        ### Research Models

        - CNN Version 2
        - CNN-LSTM
        - Attention-LSTM
        - ProtoNet
        - Tiny Transformer
        - TimeXer

        **Deployment Status**

        CNN Version 2 is currently deployed
        for live prediction.
        """
    )


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="main-title">
        🎙️ Speech Emotion Recognition
    </div>

    <div class="sub-title">
        Live Emotion Prediction using CNN Version 2
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# MODEL SELECTION
# ==========================================================

st.subheader("🤖 Research Model")

selected_model = st.selectbox(
    "Select Research Model",
    [
        "CNN Version 2 (Deployed)",
        "CNN-LSTM",
        "Attention-LSTM",
        "ProtoNet",
        "Tiny Transformer",
        "TimeXer"
    ],
    key="research_model_selection"
)


if selected_model != "CNN Version 2 (Deployed)":

    st.info(
        f"""
        **{selected_model}** is available as an
        offline research model but is not currently
        deployed in the live demonstrator.

        CNN Version 2 is used for live prediction.
        """
    )


# ==========================================================
# AUDIO INPUT
# ==========================================================

st.divider()

st.header("🎤 Audio Input")

upload_tab, record_tab = st.tabs(
    [
        "📁 Upload Audio",
        "🎙️ Record Audio"
    ]
)


audio_path = None


# ==========================================================
# UPLOAD AUDIO
# ==========================================================

with upload_tab:

    uploaded_file = st.file_uploader(
        "Upload a WAV file",
        type=["wav"],
        key="upload_audio_file"
    )

    if uploaded_file is not None:

        try:

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_file.close()

            audio_path = temp_file.name

            st.success(
                f"Audio uploaded successfully: "
                f"{uploaded_file.name}"
            )

        except Exception as e:

            st.error(
                "Unable to process uploaded audio."
            )

            st.exception(e)


# ==========================================================
# RECORD AUDIO
# ==========================================================

with record_tab:

    st.markdown(
        "Click the microphone button below to record your speech."
    )

    audio_recording = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        key="live_prediction_recorder"
    )

    if audio_recording:

        try:

            recorded_bytes = audio_recording["bytes"]

            recorded_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )

            recorded_file.write(
                recorded_bytes
            )

            recorded_file.close()

            st.session_state[
                "recorded_audio_path"
            ] = recorded_file.name

            st.audio(
                recorded_bytes,
                format="audio/wav"
            )

            st.success(
                "Recording captured successfully."
            )

        except Exception as e:

            st.error(
                "Unable to process recording."
            )

            st.exception(e)


    if (
        "recorded_audio_path"
        in st.session_state
    ):

        audio_path = st.session_state[
            "recorded_audio_path"
        ]

        st.button(
            "🔄 Record Again",
            key="record_again_button",
            use_container_width=True
        )


# ==========================================================
# CHECK AUDIO
# ==========================================================

if audio_path is None:

    st.info(
        "Please upload a WAV file or record your speech to continue."
    )

    st.stop()


# ==========================================================
# AUDIO PREVIEW
# ==========================================================

st.divider()

st.header("🔊 Audio Preview")

try:

    with open(
        audio_path,
        "rb"
    ) as audio_file:

        audio_bytes = audio_file.read()

    st.audio(
        audio_bytes,
        format="audio/wav"
    )

except Exception as e:

    st.warning(
        "Audio preview could not be loaded."
    )


# ==========================================================
# LOAD AUDIO AND EXTRACT FEATURES
# ==========================================================

try:

    features = extract_all_features(
        audio_path
    )

except Exception as e:

    st.error(
        "Unable to extract audio features."
    )

    st.exception(e)

    st.stop()


# ==========================================================
# AUDIO DATA
# ==========================================================

signal = features[
    "signal"
]

sample_rate = features[
    "sample_rate"
]

duration = (
    len(signal)
    / sample_rate
)


# ==========================================================
# AUDIO INFORMATION
# ==========================================================

st.header("📊 Audio Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Duration",
        f"{duration:.2f} sec"
    )

with col2:

    st.metric(
        "Sample Rate",
        f"{sample_rate} Hz"
    )

with col3:

    st.metric(
        "Samples",
        f"{len(signal):,}"
    )


# ==========================================================
# FEATURE EXTRACTION SUMMARY
# ==========================================================

st.divider()

st.header("🧬 Extracted Audio Features")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:

    st.info(
        """
        **Log-Mel Spectrogram**

        Used as the primary CNN input feature.
        """
    )

with feature_col2:

    st.info(
        """
        **MFCC**

        Captures spectral characteristics
        of the speech signal.
        """
    )

with feature_col3:

    st.info(
        """
        **Chroma**

        Represents pitch-class information
        within the audio.
        """
    )


# ==========================================================
# VISUALIZATIONS
# ==========================================================

st.divider()

st.header("📈 Audio Feature Visualization")


# ==========================================================
# WAVEFORM
# ==========================================================

st.subheader("🌊 Waveform")

try:

    waveform_fig = plot_waveform(
        signal,
        sample_rate
    )

    st.pyplot(
        waveform_fig,
        use_container_width=True
    )

except Exception as e:

    st.warning(
        "Waveform visualization could not be generated."
    )

    st.exception(e)


# ==========================================================
# SPECTROGRAM
# ==========================================================

st.subheader("📡 Spectrogram")

try:

    spectrogram_fig = plot_spectrogram(
        signal,
        sample_rate
    )

    st.pyplot(
        spectrogram_fig,
        use_container_width=True
    )

except Exception as e:

    st.warning(
        "Spectrogram visualization could not be generated."
    )

    st.exception(e)


# ==========================================================
# LOG-MEL SPECTROGRAM
# ==========================================================

st.subheader("🔥 Log-Mel Spectrogram")

try:

    logmel_fig = plot_logmel(
        signal,
        sample_rate
    )

    st.pyplot(
        logmel_fig,
        use_container_width=True
    )

except Exception as e:

    st.warning(
        "Log-Mel visualization could not be generated."
    )

    st.exception(e)


# ==========================================================
# MFCC
# ==========================================================

st.subheader("🎵 MFCC")

try:

    mfcc_fig = plot_mfcc(
        signal,
        sample_rate
    )

    st.pyplot(
        mfcc_fig,
        use_container_width=True
    )

except Exception as e:

    st.warning(
        "MFCC visualization could not be generated."
    )

    st.exception(e)


# ==========================================================
# CHROMA
# ==========================================================

st.subheader("🎼 Chroma")

try:

    chroma_fig = plot_chroma(
        signal,
        sample_rate
    )

    st.pyplot(
        chroma_fig,
        use_container_width=True
    )

except Exception as e:

    st.warning(
        "Chroma visualization could not be generated."
    )

    st.exception(e)


# ==========================================================
# ATTENTION HEATMAP
# ==========================================================

st.subheader("🧠 Feature Attention / Activation View")

try:

    attention_fig = plot_attention_heatmap(
        signal,
        sample_rate
    )

    st.pyplot(
        attention_fig,
        use_container_width=True
    )

except Exception:

    st.info(
        "Attention visualization is provided as a research-oriented feature view."
    )


# ==========================================================
# FEATURE STATISTICS
# ==========================================================

st.divider()

st.header("📋 Feature Statistics")

try:

    numeric_signal = np.asarray(
        signal
    )

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:

        st.metric(
            "Mean",
            f"{np.mean(numeric_signal):.4f}"
        )

    with stat_col2:

        st.metric(
            "Std",
            f"{np.std(numeric_signal):.4f}"
        )

    with stat_col3:

        st.metric(
            "Minimum",
            f"{np.min(numeric_signal):.4f}"
        )

    with stat_col4:

        st.metric(
            "Maximum",
            f"{np.max(numeric_signal):.4f}"
        )

except Exception as e:

    st.warning(
        "Feature statistics could not be calculated."
    )


# ==========================================================
# LIVE PREDICTION
# ==========================================================

st.divider()

st.header(
    "🧠 CNN Version 2 Emotion Prediction"
)


predict_button = st.button(
    "🚀 Predict Emotion",
    use_container_width=True,
    key="cnn_v2_predict_button"
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    loading_placeholder = st.empty()

    try:

        # --------------------------------------------------
        # SHOW EMOTION LOADER
        # --------------------------------------------------

        show_emotion_loader(
            loading_placeholder
        )

        # --------------------------------------------------
        # CNN PREDICTION
        #
        # IMPORTANT:
        # predict_emotion() returns a DICTIONARY.
        # --------------------------------------------------

        result = predict_emotion(
            audio_path
        )

        # --------------------------------------------------
        # EXTRACT VALUES FROM RESULT DICTIONARY
        # --------------------------------------------------

        predicted_emotion = result[
            "emotion"
        ]

        confidence = result[
            "confidence"
        ]

        probabilities = result[
            "probabilities"
        ]

        inference_time = result[
            "inference_time"
        ]

        class_names_from_result = result[
            "class_names"
        ]

        prediction_index = result[
            "prediction_index"
        ]

        # --------------------------------------------------
        # REMOVE LOADER
        # --------------------------------------------------

        loading_placeholder.empty()

    except Exception as e:

        loading_placeholder.empty()

        st.error(
            "An error occurred during prediction."
        )

        st.exception(e)

        st.stop()


    # ======================================================
    # PREDICTION RESULT
    # ======================================================

    st.success(
        "Emotion prediction completed successfully."
    )


    # ======================================================
    # MAIN RESULT
    # ======================================================

    prediction_emoji = emotion_emoji(
        predicted_emotion
    )

    st.markdown(
        f"""
        <div class="prediction-box">

            <div class="prediction-emotion">
                {prediction_emoji}
                {predicted_emotion}
            </div>

            <div class="prediction-confidence">
                Confidence: <strong>{confidence:.2f}%</strong>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ======================================================
    # RESULT METRICS
    # ======================================================

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Detected Emotion",
            predicted_emotion
        )

    with metric2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with metric3:

        st.metric(
            "Inference Time",
            f"{inference_time:.4f} sec"
        )


    # ======================================================
    # MODEL PERFORMANCE
    # ======================================================

    st.subheader(
        "📊 Model Performance"
    )

    performance_col1, performance_col2 = st.columns(2)

    with performance_col1:

        st.metric(
            "CNN V2 Test Accuracy",
            f"{TEST_ACCURACY:.2f}%"
        )

    with performance_col2:

        st.metric(
            "Predicted Class Index",
            prediction_index
        )


    # ======================================================
    # PROBABILITY TABLE
    # ======================================================

    st.subheader(
        "📈 Emotion Probability Distribution"
    )

    probability_df = pd.DataFrame(
        {
            "Emotion": class_names_from_result,
            "Probability (%)": [
                float(p * 100)
                for p in probabilities
            ]
        }
    )

    probability_df = probability_df.sort_values(
        "Probability (%)",
        ascending=False
    ).reset_index(
        drop=True
    )

    probability_df[
        "Probability (%)"
    ] = probability_df[
        "Probability (%)"
    ].round(2)

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )


    # ======================================================
    # PROBABILITY CHART
    # ======================================================

    st.subheader(
        "📊 Emotion Probability Chart"
    )

    try:

        probability_fig = plot_probability_chart(
            probabilities,
            class_names_from_result
        )

        st.pyplot(
            probability_fig,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            "Probability chart could not be generated."
        )


    # ======================================================
    # TOP 3 PREDICTIONS
    # ======================================================

    st.subheader(
        "🏆 Top 3 Emotion Predictions"
    )

    top_predictions = get_top_predictions(
        result,
        k=3
    )

    top_col1, top_col2, top_col3 = st.columns(3)

    top_columns = [
        top_col1,
        top_col2,
        top_col3
    ]

    for i, prediction_item in enumerate(
        top_predictions
    ):

        emotion = prediction_item[
            "emotion"
        ]

        prediction_confidence = prediction_item[
            "confidence"
        ]

        emoji = emotion_emoji(
            emotion
        )

        with top_columns[i]:

            st.markdown(
                f"""
                ### {emoji} {emotion}

                **{prediction_confidence:.2f}%**
                """
            )


    # ======================================================
    # INTERPRETATION
    # ======================================================

    st.subheader(
        "📝 Prediction Interpretation"
    )

    st.write(
        f"""
        The CNN Version 2 model predicts **{predicted_emotion}**
        as the dominant emotional state in the provided speech,
        with a confidence of **{confidence:.2f}%**.

        The model processes the speech through the Log-Mel
        spectrogram representation and produces probability
        scores for all eight RAVDESS emotion classes.
        """
    )


    # ======================================================
    # PREDICTION HISTORY
    # ======================================================

    if (
        "prediction_history"
        not in st.session_state
    ):

        st.session_state[
            "prediction_history"
        ] = []


    st.session_state[
        "prediction_history"
    ].append(
        {
            "Emotion": predicted_emotion,
            "Confidence (%)": round(
                confidence,
                2
            ),
            "Inference Time (sec)": round(
                inference_time,
                4
            )
        }
    )


    # ======================================================
    # DISPLAY HISTORY
    # ======================================================

    if st.session_state[
        "prediction_history"
    ]:

        st.subheader(
            "🕘 Prediction History"
        )

        history_df = pd.DataFrame(
            st.session_state[
                "prediction_history"
            ]
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# CHUNK-LEVEL ANALYSIS
# ==========================================================

st.divider()

st.header(
    "⏱️ Chunk-Level Emotion Analysis"
)

st.markdown(
    """
    Longer speech recordings can be analyzed using
    independent **3-second audio chunks**.

    Each complete chunk is processed separately by
    CNN Version 2.
    """
)


chunk_analyze_button = st.button(
    "🔍 Analyze Every 3-Second Chunk",
    use_container_width=True,
    key="chunk_analysis_button"
)


# ==========================================================
# CHUNK ANALYSIS
# ==========================================================

if chunk_analyze_button:

    loading_placeholder = st.empty()

    show_emotion_loader(
        loading_placeholder
    )

    try:

        chunk_results = predict_emotion_by_chunks(
            audio_path
        )

        loading_placeholder.empty()

    except Exception as e:

        loading_placeholder.empty()

        st.error(
            "An error occurred during chunk-level analysis."
        )

        st.exception(e)

        st.stop()


    # ======================================================
    # NO CHUNKS
    # ======================================================

    if not chunk_results:

        st.warning(
            "The audio does not contain a complete 3-second chunk."
        )

    else:

        st.success(
            f"""
            Chunk-level analysis completed.
            {len(chunk_results)} complete 3-second chunk(s)
            were analyzed.
            """
        )


        # ==================================================
        # CHUNK TABLE
        # ==================================================

        st.subheader(
            "📋 Chunk-by-Chunk Predictions"
        )

        chunk_table_data = []


        for chunk in chunk_results:

            chunk_table_data.append(
                {
                    "Chunk": chunk[
                        "chunk"
                    ],

                    "Start Time (s)": round(
                        chunk[
                            "start_time"
                        ],
                        2
                    ),

                    "End Time (s)": round(
                        chunk[
                            "end_time"
                        ],
                        2
                    ),

                    "Emotion": chunk[
                        "emotion"
                    ],

                    "Confidence (%)": round(
                        chunk[
                            "confidence"
                        ],
                        2
                    ),

                    "Inference Time (s)": round(
                        chunk[
                            "inference_time"
                        ],
                        4
                    )
                }
            )


        chunk_df = pd.DataFrame(
            chunk_table_data
        )


        st.dataframe(
            chunk_df,
            use_container_width=True,
            hide_index=True
        )


        # ==================================================
        # TOP EMOTION FOR EACH CHUNK
        # ==================================================

        st.subheader(
            "🎭 Chunk Emotion Details"
        )


        for chunk in chunk_results:

            chunk_number = chunk[
                "chunk"
            ]

            start_time_audio = chunk[
                "start_time"
            ]

            end_time_audio = chunk[
                "end_time"
            ]

            emotion = chunk[
                "emotion"
            ]

            confidence = chunk[
                "confidence"
            ]

            probabilities = chunk[
                "probabilities"
            ]


            with st.expander(
                f"Chunk {chunk_number}: "
                f"{start_time_audio:.1f}s – "
                f"{end_time_audio:.1f}s | "
                f"{emotion_emoji(emotion)} {emotion} "
                f"({confidence:.2f}%)"
            ):

                detail_col1, detail_col2 = st.columns(2)


                with detail_col1:

                    st.metric(
                        "Predicted Emotion",
                        f"{emotion_emoji(emotion)} {emotion}"
                    )


                with detail_col2:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )


                # ------------------------------------------
                # TOP 3 EMOTIONS FOR THIS CHUNK
                # ------------------------------------------

                chunk_indices = np.argsort(
                    probabilities
                )[::-1][:3]


                chunk_top_data = []


                for idx in chunk_indices:

                    chunk_top_data.append(
                        {
                            "Emotion": class_names[idx],
                            "Probability (%)": round(
                                float(
                                    probabilities[idx]
                                    * 100
                                ),
                                2
                            )
                        }
                    )


                chunk_top_df = pd.DataFrame(
                    chunk_top_data
                )


                st.markdown(
                    "**Top 3 emotion probabilities:**"
                )


                st.dataframe(
                    chunk_top_df,
                    use_container_width=True,
                    hide_index=True
                )


        # ==================================================
        # CHUNK STATISTICS
        # ==================================================

        st.divider()

        st.subheader(
            "📊 Chunk-Level Statistics"
        )


        statistics = calculate_chunk_statistics(
            chunk_results
        )


        dominant_emotion = statistics[
            "dominant_emotion"
        ]

        dominant_percentage = statistics[
            "dominant_percentage"
        ]

        average_confidence = statistics[
            "average_confidence"
        ]


        # ==================================================
        # SUMMARY METRICS
        # ==================================================

        stat_col1, stat_col2, stat_col3 = st.columns(3)


        with stat_col1:

            st.metric(
                "Total Chunks",
                len(chunk_results)
            )


        with stat_col2:

            if dominant_emotion:

                st.metric(
                    "Dominant Emotion",
                    f"{emotion_emoji(dominant_emotion)} "
                    f"{dominant_emotion}"
                )

            else:

                st.metric(
                    "Dominant Emotion",
                    "N/A"
                )


        with stat_col3:

            st.metric(
                "Average Confidence",
                f"{average_confidence:.2f}%"
            )


        # ==================================================
        # EMOTION DISTRIBUTION
        # ==================================================

        st.subheader(
            "📈 Emotion Distribution Across Chunks"
        )


        emotion_counts = statistics[
            "emotion_counts"
        ]

        emotion_percentages = statistics[
            "emotion_percentages"
        ]


        distribution_data = []


        for emotion, count in emotion_counts.items():

            distribution_data.append(
                {
                    "Emotion": emotion,

                    "Chunks": count,

                    "Percentage (%)": round(
                        emotion_percentages[
                            emotion
                        ],
                        2
                    )
                }
            )


        distribution_df = pd.DataFrame(
            distribution_data
        )


        if not distribution_df.empty:

            distribution_df = distribution_df.sort_values(
                "Chunks",
                ascending=False
            ).reset_index(
                drop=True
            )


            st.dataframe(
                distribution_df,
                use_container_width=True,
                hide_index=True
            )


        # ==================================================
        # CHUNK SUMMARY
        # ==================================================

        st.subheader(
            "📝 Chunk-Level Interpretation"
        )


        if dominant_emotion:

            st.write(
                f"""
                Across the analyzed 3-second chunks,
                **{dominant_emotion}** was the most frequently
                predicted emotion.

                It appeared in approximately
                **{dominant_percentage:.2f}%**
                of the analyzed chunks.

                The average prediction confidence across
                all chunks was
                **{average_confidence:.2f}%**.
                """
            )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    """
    Speech Emotion Recognition Research Demonstrator |
    CNN Version 2 | RAVDESS Dataset |
    Log-Mel Spectrogram Features
    """
)

