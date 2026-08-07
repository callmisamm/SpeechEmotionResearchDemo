import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Comparison")
st.markdown("---")

st.markdown("""
This page summarizes the performance of the deep learning,
attention-based, few-shot learning, transfer learning,
and time-series models evaluated throughout this research.
""")

# ==========================================================
# MODEL RESULTS
# ==========================================================

results = pd.DataFrame({

    "Model":[
        "CNN",
        "1D CNN",
        "Baseline Supervised",
        "Attention-LSTM",
        "CNN-LSTM",
        "ProtoNet",
        "TimeXer",
        "iTransformer",
        "TimesNet"
    ],

    "Accuracy":[
        92,
        84,
        87,
        75,
        75,
        70.9,
        88,
        74,
        88
    ],

    "Category":[
        "CNN",
        "CNN",
        "Transfer",
        "Attention",
        "Hybrid",
        "Few-Shot",
        "Time-Series",
        "Time-Series",
        "Time-Series"
    ]
})

# ==========================================================
# KPI CARDS
# ==========================================================

col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric(
        "Models Evaluated",
        len(results)
    )

with col2:
    st.metric(
        "Best Accuracy",
        "92%"
    )

with col3:
    st.metric(
        "Datasets",
        "7"
    )

with col4:
    st.metric(
        "Best Model",
        "CNN"
    )

st.markdown("---")

# ==========================================================
# TABLE
# ==========================================================

st.subheader("Performance Summary")

st.dataframe(
    results,
    use_container_width=True
)

# ==========================================================
# BAR CHART
# ==========================================================

st.subheader("Accuracy Comparison")

fig = px.bar(

    results,

    x="Model",

    y="Accuracy",

    color="Category",

    text="Accuracy",

    height=500

)

fig.update_traces(textposition="outside")

fig.update_layout(
    yaxis_title="Accuracy (%)",
    xaxis_title=""
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# RANKING
# ==========================================================

st.subheader("Overall Ranking")

ranking = results.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

ranking.index += 1

st.dataframe(
    ranking,
    use_container_width=True
)

# ==========================================================
# PIE CHART
# ==========================================================

st.subheader("Distribution of Model Categories")

category_counts = results.groupby(
    "Category"
).size().reset_index(name="Count")

fig = px.pie(
    category_counts,
    values="Count",
    names="Category",
    hole=0.45
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# RADAR CHART
# ==========================================================

st.subheader("Top Model Performance")

categories = [
    "Accuracy",
    "Explainability",
    "Generalization",
    "Speed",
    "Robustness"
]

cnn = [92,80,88,95,90]
attention=[75,95,75,72,80]
protonet=[71,85,90,80,88]
timexer=[88,70,87,75,90]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=cnn,
    theta=categories,
    fill='toself',
    name='CNN'
))

fig.add_trace(go.Scatterpolar(
    r=attention,
    theta=categories,
    fill='toself',
    name='Attention-LSTM'
))

fig.add_trace(go.Scatterpolar(
    r=protonet,
    theta=categories,
    fill='toself',
    name='ProtoNet'
))

fig.add_trace(go.Scatterpolar(
    r=timexer,
    theta=categories,
    fill='toself',
    name='TimeXer'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0,100]
        )
    ),
    showlegend=True,
    height=650
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# THESIS OBSERVATIONS
# ==========================================================

st.markdown("---")

st.subheader("Research Findings")

st.success("""
**Key Findings**

• CNN achieved the highest recognition accuracy (92%).

• TimeXer demonstrated competitive performance on merged datasets.

• ProtoNet provided effective few-shot learning capability.

• Attention-LSTM improved interpretability through attention maps.

• Cross-lingual performance decreased because of domain shift.

• The hybrid framework successfully combines feature extraction,
attention mechanisms, few-shot learning, and time-series modeling.
""")

# ==========================================================
# FINAL CONCLUSION
# ==========================================================

st.markdown("---")

st.info("""
**Conclusion**

The experimental evaluation indicates that CNN-based architectures
remain highly effective for speech emotion recognition, while
few-shot learning, attention mechanisms, transfer learning, and
time-series models provide complementary strengths for
generalization, explainability, and cross-domain adaptation.
""")