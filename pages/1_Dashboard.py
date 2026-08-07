import streamlit as st


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.title{
    font-size:42px;
    color:#00C8FF;
    font-weight:bold;
}

.subtitle{
    font-size:22px;
    color:white;
}

.card{
    background:#1B1F27;
    padding:20px;
    border-radius:15px;
    border:1px solid #2C3440;
    margin-bottom:15px;
}

.metric{
    background:#161B22;
    padding:15px;
    border-radius:12px;
    text-align:center;
}

.big{
    font-size:30px;
    color:#00C8FF;
    font-weight:bold;
}

.small{
    font-size:18px;
    color:white;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
"""
<div class='title'>
🎤 Speech Emotion Recognition Research Platform
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
Hybrid Attention • Few-Shot Learning • Time-Series Models • Cross-Lingual Evaluation
</div>
""",
unsafe_allow_html=True
)

st.divider()

# ==========================================================
# INFORMATION
# ==========================================================

left,right=st.columns([2,1])

with left:

    st.markdown("## 🎓 Research Information")

    st.markdown("""
**University**

University of the Punjab

**Department**

Department of Information Technology

**Research Area**

Speech Emotion Recognition using Deep Learning

**Supervisor**

*Dr.Muhammad Adeel Nisar*

**Student**

*Sameen Ali*
""")

with right:

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Artificial_neural_network.svg/640px-Artificial_neural_network.svg.png",
        use_container_width=True
    )

# ==========================================================
# CONTRIBUTIONS
# ==========================================================

st.divider()

st.markdown("## 🚀 Thesis Contributions")

c1,c2,c3=st.columns(3)

with c1:

    st.success("✅ Attention Mechanism")

    st.success("✅ Few-Shot Learning")

with c2:

    st.success("✅ Time-Series Models")

    st.success("✅ Transfer Learning")

with c3:

    st.success("✅ Cross-Lingual Evaluation")

    st.success("✅ Explainable AI")

# ==========================================================
# DATASETS
# ==========================================================

st.divider()

st.markdown("## 📂 Datasets")

d1,d2,d3,d4=st.columns(4)

with d1:
    st.info("🎤 RAVDESS")

with d2:
    st.info("🎤 CREMA-D")

with d3:
    st.info("🎤 TESS")

with d4:
    st.info("🎤 Urdu Dataset")

# ==========================================================
# METRICS
# ==========================================================

st.divider()

st.markdown("## 📊 Research Statistics")

m1,m2,m3,m4=st.columns(4)

with m1:

    st.metric(
        "Datasets",
        "7"
    )

with m2:

    st.metric(
        "Models Evaluated",
        "12"
    )

with m3:

    st.metric(
        "Emotion Classes",
        "8"
    )

with m4:

    st.metric(
        "Best Accuracy",
        "92%"
    )

# ==========================================================
# MODEL SUMMARY
# ==========================================================

st.divider()

st.markdown("## 🧠 Evaluated Models")

col1,col2=st.columns(2)

with col1:

    st.markdown("""
- ✅ CNN
- ✅ 1D CNN
- ✅ CNN-LSTM
- ✅ Attention-LSTM
- ✅ Tiny Transformer
- ✅ ProtoNet
""")

with col2:

    st.markdown("""
- ✅ TimeXer
- ✅ iTransformer
- ✅ TimesNet
- ✅ Transfer Learning
- ✅ Meta Learning
- ✅ Cross-Lingual Learning
""")

# ==========================================================
# WORKFLOW
# ==========================================================

st.divider()

st.markdown("## 🔄 Research Workflow")

st.markdown("""
```text
Speech Input
      │
      ▼
Audio Preprocessing
      │
      ▼
Feature Extraction
(Log-Mel Spectrogram)
      │
      ▼
CNN / Attention / Time-Series Models
      │
      ▼
Emotion Prediction
      │
      ▼
Visualization & Analysis""")