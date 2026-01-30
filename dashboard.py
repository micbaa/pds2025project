import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Job Title Classification - Model Comparison", layout="wide")

st.title("Job Title Classification Dashboard")

# Model results data
results = pd.DataFrame({
    "Notebook": [
        "pretrained_finetuned_xlmr_model.ipynb",
        "BoW_Models (1).ipynb",
        "Embeddings_Model.ipynb",
        "bow_and_emb_with_extended_v2 (2).ipynb",
        "model_pipeline_extended Kopie.ipynb",
        "Hai_PDSfinalproject_data.ipynb",
        "Hai_PDSfinalproject_data.ipynb",
        "Hai_PDSfinalproject_data.ipynb",
    ],
    "Approach": [
        "XLM-RoBERTa Fine-tuning",
        "BoW + Logistic Regression",
        "Embeddings + Logistic Regression",
        "Self-Training (Pseudo-labels)",
        "Career Heuristics",
        "Rule-based (baseline)",
        "Embeddings (no threshold)",
        "Embeddings (with threshold)",
    ],
    "Department Accuracy": [0.40, 0.40, 0.40, 0.70, 0.56, 0.19, 0.35, 0.59],
    "Department Recall": [0.51, 0.48, 0.48, 0.59, 0.29, 0.29, 0.43, 0.52],
    "Seniority Accuracy": [0.42, 0.40, 0.40, 0.63, 0.37, 0.26, 0.37, 0.52],
    "Seniority Recall": [0.62, 0.58, 0.58, 0.64, 0.29, 0.43, 0.61, 0.65],
    "ML Model": ["Yes", "Yes", "Yes", "Yes", "Partial", "No", "Yes", "Yes"],
})

# Summary metrics
st.header("Best Results")
col1, col2, col3, col4 = st.columns(4)

best_dept_acc = results.loc[results["Department Accuracy"].idxmax()]
best_dept_rec = results.loc[results["Department Recall"].idxmax()]
best_sen_acc = results.loc[results["Seniority Accuracy"].idxmax()]
best_sen_rec = results.loc[results["Seniority Recall"].idxmax()]

col1.metric("Best Dept Accuracy", f"{best_dept_acc['Department Accuracy']:.0%}", best_dept_acc["Approach"])
col2.metric("Best Dept Recall", f"{best_dept_rec['Department Recall']:.0%}", best_dept_rec["Approach"])
col3.metric("Best Sen Accuracy", f"{best_sen_acc['Seniority Accuracy']:.0%}", best_sen_acc["Approach"])
col4.metric("Best Sen Recall", f"{best_sen_rec['Seniority Recall']:.0%}", best_sen_rec["Approach"])

st.divider()

# Comparison charts
st.header("Model Comparison")

tab1, tab2 = st.tabs(["Accuracy", "Recall"])

with tab1:
    fig_acc = px.bar(
        results.melt(id_vars=["Approach"], value_vars=["Department Accuracy", "Seniority Accuracy"]),
        x="Approach",
        y="value",
        color="variable",
        barmode="group",
        labels={"value": "Accuracy", "variable": "Task"},
        title="Accuracy by Model and Task",
    )
    fig_acc.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
    st.plotly_chart(fig_acc, use_container_width=True)

with tab2:
    fig_rec = px.bar(
        results.melt(id_vars=["Approach"], value_vars=["Department Recall", "Seniority Recall"]),
        x="Approach",
        y="value",
        color="variable",
        barmode="group",
        labels={"value": "Macro Recall", "variable": "Task"},
        title="Macro Recall by Model and Task",
    )
    fig_rec.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
    st.plotly_chart(fig_rec, use_container_width=True)

st.divider()

# Results table
st.header("Detailed Results")
st.dataframe(
    results.style.format({
        "Department Accuracy": "{:.0%}",
        "Department Recall": "{:.0%}",
        "Seniority Accuracy": "{:.0%}",
        "Seniority Recall": "{:.0%}",
    }).background_gradient(subset=["Department Accuracy", "Department Recall", "Seniority Accuracy", "Seniority Recall"], cmap="Greens"),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# Approach explanations
st.header("Approach Descriptions")

with st.expander("XLM-RoBERTa Fine-tuning"):
    st.markdown("""
    - **Model:** `xlm-roberta-base` transformer
    - **Training:** Fine-tuned on v2 label data
    - **Issue:** Poor domain adaptation to LinkedIn titles
    """)

with st.expander("BoW + Logistic Regression"):
    st.markdown("""
    - **Model:** TF-IDF vectorizer + Logistic Regression
    - **Training:** Trained on v2 label data only
    - **Issue:** Limited vocabulary, no multilingual understanding
    """)

with st.expander("Embeddings + Logistic Regression"):
    st.markdown("""
    - **Model:** `paraphrase-multilingual-mpnet-base-v2` embeddings + Logistic Regression
    - **Training:** Trained on v2 label data only
    - **Benefit:** Multilingual support
    """)

with st.expander("Self-Training (Pseudo-labels) - BEST"):
    st.markdown("""
    - **Model:** BoW/Embeddings + Logistic Regression
    - **Training:** v2 data + pseudo-labeled LinkedIn data (1,886 titles)
    - **Key insight:** Model learns LinkedIn vocabulary through self-training
    - **Result:** Best accuracy (70% dept, 63% seniority)
    """)

with st.expander("Career Heuristics"):
    st.markdown("""
    - **Model:** Embeddings + Logistic Regression + handwritten rules
    - **Rules:** Career trajectory, org patterns, experience length
    - **Issue:** Heuristics don't generalize well
    """)

with st.expander("Rule-based (baseline)"):
    st.markdown("""
    - **Model:** Predefined keyword matching
    - **Training:** No training, just rules
    - **Issue:** Low coverage (~25%), misses most titles
    """)

with st.expander("Embeddings (no threshold)"):
    st.markdown("""
    - **Model:** Sentence embeddings + Logistic Regression
    - **Training:** Trained on v2 label data
    - **Issue:** No confidence handling for ambiguous titles
    """)

with st.expander("Embeddings (with threshold)"):
    st.markdown("""
    - **Model:** Sentence embeddings + Logistic Regression + confidence thresholds
    - **Training:** Trained on v2 label data with threshold tuning
    - **Benefit:** Falls back to "Other"/"Professional" when uncertain
    """)
