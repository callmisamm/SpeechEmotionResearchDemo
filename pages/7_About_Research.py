import streamlit as st

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="About Research",
    page_icon="📖",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: white;
}

.card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 15px;
    border-left: 6px solid #00C8FF;
    margin-bottom: 15px;
}

.metric-card {
    background-color: #18212F;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.title("📖 About This Research")

st.markdown("""
## Hybrid Speech Emotion Recognition Framework

This research integrates:

- 🎯 Attention Mechanisms
- 🧠 Deep Learning
- ⚡ Few-Shot Learning
- 📈 Time-Series Models
- 🌍 Cross-Lingual Evaluation
""")

st.divider()

# ==========================================================
# RESEARCH GAP
# ==========================================================

st.header("Research Gap")

st.markdown("""
<div class="card">

Existing Speech Emotion Recognition (SER) systems suffer from several limitations:

<ul>
<li>Limited generalization across datasets</li>
<li>Poor performance with limited training data</li>
<li>Lack of explainability</li>
<li>Weak cross-lingual performance</li>
<li>Domain shift between English and Urdu speech</li>
</ul>

This thesis proposes a hybrid framework to address these challenges.

</div>
""", unsafe_allow_html=True)

# ==========================================================
# OBJECTIVES
# ==========================================================

st.header("Research Objectives")

st.markdown("""
<div class="card">

✔ Develop a Hybrid SER Framework<br><br>

✔ Compare CNN, CNN-LSTM, Attention, Time-Series and Few-Shot models<br><br>

✔ Evaluate multiple benchmark datasets<br><br>

✔ Investigate Cross-Lingual Transfer Learning<br><br>

✔ Improve Explainability using Attention Visualization

</div>
""", unsafe_allow_html=True)

# ==========================================================
# METHODOLOGY
# ==========================================================

st.header("Research Methodology")

st.code("""
Speech Audio
      │
      ▼
Audio Preprocessing
      │
      ▼
Feature Extraction
(Log-Mel Spectrogram)
      │
      ▼
Data Augmentation
      │
      ▼
CNN / Attention / Few-Shot /
Time-Series Models
      │
      ▼
Emotion Prediction
      │
      ▼
Performance Evaluation
""")

# ==========================================================
# DATASETS
# ==========================================================

st.header("Datasets Used")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    <div class="metric-card">
    <h3>RAVDESS</h3>
    24 Actors<br>
    8 Emotions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
    <h3>TESS</h3>
    2 Speakers<br>
    7 Emotions
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="metric-card">
    <h3>CREMA-D</h3>
    91 Actors<br>
    6 Emotions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
    <h3>Urdu Dataset</h3>
    Cross-Lingual Evaluation
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# THESIS CONTRIBUTIONS
# ==========================================================

st.header("Thesis Contributions")

contributions = [
    "Hybrid Attention + Time-Series + Few-Shot Learning Framework",
    "Evaluation on Multiple Public Speech Emotion Datasets",
    "Few-Shot Learning Baseline",
    "Cross-Lingual English → Urdu Evaluation",
    "Digital Signal Processing Filter Analysis",
    "Attention Visualization for Explainability"
]

for i, contribution in enumerate(contributions, start=1):
    st.success(f"Contribution {i}: {contribution}")

# ==========================================================
# RESULTS
# ==========================================================

st.header("Experimental Highlights")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="Best CNN Accuracy",
        value="92%"
    )

with c2:
    st.metric(
        label="Cross-Lingual Accuracy",
        value="44%"
    )

with c3:
    st.metric(
        label="Few-Shot Accuracy",
        value="70.9%"
    )

# ==========================================================
# SOFTWARE
# ==========================================================

st.header("Software & Libraries")

st.markdown("""
- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- Matplotlib
- Streamlit
""")

# ==========================================================
# FUTURE WORK
# ==========================================================

st.header("Future Work")

st.markdown("""
<div class="card">

Future directions include:

<ul>

<li>Real-Time Speech Emotion Recognition</li>

<li>Transformer-Based Models</li>

<li>Self-Supervised Learning</li>

<li>Large Multilingual Speech Datasets</li>

<li>Deployment on Mobile & Edge Devices</li>

<li>Healthcare and Clinical Applications</li>

</ul>

</div>
""", unsafe_allow_html=True)

# ==========================================================
# RESEARCH SUMMARY
# ==========================================================

st.header("Research Summary")

st.info("""
This research presents a Hybrid Speech Emotion Recognition framework
combining Deep Learning, Attention Mechanisms, Few-Shot Learning,
Time-Series Modeling, and Cross-Lingual Evaluation.

The proposed framework demonstrates improved robustness,
better interpretability, and strong performance across
multiple benchmark speech emotion datasets.
""")

# ==========================================================
# THESIS INFORMATION
# ==========================================================

st.header("Thesis Information")

st.write("**Degree:** MPhil Information Technology")

st.write("**Research Area:** Artificial Intelligence")

st.write("**Research Domain:** Speech Emotion Recognition")

st.write("**University:** University of the Punjab")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
<div class="footer">

<h3>Speech Emotion Recognition Research Demonstrator</h3>

Department of Information Technology<br>

University of the Punjab<br><br>

Developed for MPhil Thesis Defense<br><br>

© 2026

</div>
""", unsafe_allow_html=True)