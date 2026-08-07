import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Cross-Lingual Analysis",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("🌍 Cross-Lingual Speech Emotion Recognition")

st.markdown("""
This page demonstrates the cross-lingual evaluation performed in the thesis.

The objective is to analyze how a model trained on **English emotional speech**
generalizes to **Urdu emotional speech** without retraining.
""")

st.divider()

# =====================================================
# PERFORMANCE OVERVIEW
# =====================================================

st.subheader("📊 English → Urdu Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "English Validation Accuracy",
        "87.0%"
    )

with col2:
    st.metric(
        "Urdu Test Accuracy",
        "44.0%"
    )

# =====================================================
# BAR CHART
# =====================================================

performance = pd.DataFrame({
    "Dataset": ["English", "Urdu"],
    "Accuracy": [87, 44]
})

fig = px.bar(
    performance,
    x="Dataset",
    y="Accuracy",
    color="Dataset",
    text="Accuracy",
    title="Cross-Lingual Performance"
)

fig.update_traces(textposition="outside")

fig.update_layout(
    yaxis_title="Accuracy (%)",
    xaxis_title="Language",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# DOMAIN SHIFT
# =====================================================

st.divider()

st.subheader("📉 Domain Shift")

fig2 = go.Figure()

fig2.add_trace(go.Indicator(
    mode="number+delta",
    value=44,
    delta={
        "reference":87,
        "relative":False
    },
    title={
        "text":"Accuracy after Cross-Lingual Transfer"
    }
))

fig2.update_layout(height=350)

st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# WHY PERFORMANCE DROPS
# =====================================================

st.divider()

st.subheader("🔍 Why Does Performance Drop?")

col1, col2 = st.columns(2)

with col1:

    st.info("""
**Language Differences**

• Different phonetic structures

• Different pronunciation

• Different speaking styles

• Vocabulary mismatch
""")

with col2:

    st.warning("""
**Recording Differences**

• Recording environment

• Microphone quality

• Speaker diversity

• Accent variations
""")

# =====================================================
# THESIS OBSERVATIONS
# =====================================================

st.divider()

st.subheader("📖 Thesis Observations")

st.success("""
✔ High performance on English speech

✔ Noticeable performance degradation on Urdu speech

✔ Domain adaptation remains an open research challenge

✔ Cross-lingual SER requires language-invariant feature learning

✔ Transfer learning and domain adaptation can improve generalization
""")

# =====================================================
# PIPELINE
# =====================================================

st.divider()

st.subheader("🔄 Cross-Lingual Experimental Pipeline")

st.markdown("""

English Dataset
│
▼
Feature Extraction
│
▼
CNN Training
│
▼
Learn Emotional Features
│
▼
Urdu Dataset
│
▼
Prediction
│
▼
Performance Evaluation

""")

# =====================================================
# FUTURE WORK
# =====================================================

st.divider()

st.subheader("🚀 Future Research Directions")

future = pd.DataFrame({
    "Future Direction":[
        "Domain Adaptation",
        "Meta Learning",
        "Self-Supervised Learning",
        "Transformer Models",
        "Large Multilingual Datasets"
    ],
    "Potential Impact":[
        90,
        82,
        95,
        88,
        97
    ]
})

fig3 = px.bar(
    future,
    x="Potential Impact",
    y="Future Direction",
    orientation="h",
    color="Potential Impact",
    title="Potential Future Improvements"
)

st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# SUMMARY
# =====================================================

st.divider()

st.subheader("📌 Summary")

st.write("""
The cross-lingual experiments demonstrate that although the CNN model performs
well on English emotional speech, performance decreases when evaluated on Urdu
speech due to domain shift.

This highlights the importance of transfer learning, domain adaptation,
and multilingual representation learning for robust Speech Emotion Recognition.
""")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Speech Emotion Recognition Research Demonstrator | "
    "University of the Punjab | Cross-Lingual Analysis"
)