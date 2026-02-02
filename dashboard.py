import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import joblib

st.set_page_config(page_title="Job Title Classification - Model Comparison", layout="wide")

st.title("Job Title Classification Dashboard")

# Model results data - grouped by approach type
# When updating beware of the order. Since everything is hardcoded
results = pd.DataFrame({
    "Approach Type": [
        # BoW
        "BoW",
        "BoW",
        # XLM-R
        "XLM-R",
        "XLM-R",
        "XLM-R",
        # Embeddings
        "Embeddings",
        "Embeddings",
        # Emb with extended
        "Emb + Extended",
        "Emb + Extended",
        # BoW with extended
        "BoW + Extended",
        # Career Heuristics
        "Career Heuristics",
    ],
    "Method": [
        # BoW
        "TF-IDF char_wb (3,5) + LR + tailored threshold (0.50; Senior 0.97)",
        "TF-IDF char n-grams (3,5) + LR + threshold (0.40) → Other",
        # XLM-R
        "90/10, 3 epochs (no threshold)",
        "threshold 0.55 → Professional",
        "general + tailored per-class thresholds (junior, senior)",
        # Embeddings
        "mpnet-base-v2 50/50 + LR",
        "mpnet-base-v2 50/50 + LR + threshold",
        # Emb with extended
        "v2 (capped 500) + BOW (both TH) + mpnet 70/30 + LR",
        "v2 (capped 400) + BOW (both TH) + mpnet 70/30 + LR",
        # BoW with extended
        "v2 (capped 500) + Emb (both TH) + 80/20 + TF-IDF char n-grams (3,5) + LR",
        # Career Heuristics
        "Embeddings + LR + 5 heuristics",
    ],
    "Notebook": [
        # BoW
        "Bag_of_Words_Models.ipynb",
        "Bag_of_Words_Models.ipynb",
        # XLM-R
        "finetuned_xlmr_models.ipynb",
        "finetuned_xlmr_models.ipynb",
        "finetuned_xlmr_models.ipynb",
        # Embeddings
        "Bag_of_Words_Models.ipynb",
        "Bag_of_Words_Models.ipynb",
        # Emb with extended
        "Data_Insights (1).ipynb",
        "Data_Insights (1).ipynb",
        # BoW with extended
        "Data_Insights (1).ipynb",
        # Career Heuristics
        "model_pipeline_extended.ipynb",
    ],
    "Seniority Accuracy": [
        # BoW
        0.6497,
        None,
        # XLM-R
        0.3772,
        0.3901,
        0.4591,
        # Embeddings
        0.3999,
        0.5402,
        # Emb with extended
        0.5743,
        0.5709,
        # BoW with extended
        0.60,
        # Career Heuristics
        0.366,
    ],
    "Seniority Recall": [
        # BoW
        0.59,
        None,
        # XLM-R
        0.6189,
        0.6233,
        0.6426,
        # Embeddings
        0.6505,
        0.6391,
        # Emb with extended
        0.6297,
        0.6232,
        # BoW with extended
        0.67,
        # Career Heuristics
        0.292,
    ],
    "Department Accuracy": [
        # BoW
        None,
        0.6660,
        # XLM-R
        None,
        None,
        None,
        # Embeddings
        0.4105,
        0.5963,
        # Emb with extended
        0.6994,
        0.6903,
        # BoW with extended
        0.62,
        # Career Heuristics
        0.559,
    ],
    "Department Recall": [
        # BoW
        None,
        0.48,
        # XLM-R
        None,
        None,
        None,
        # Embeddings
        0.5555,
        0.4789,
        # Emb with extended
        0.5836,
        0.5709,
        # BoW with extended
        0.51,
        # Career Heuristics
        0.289,
    ],
})

# Sidebar filter
st.sidebar.header("Filters")
approach_types = st.sidebar.multiselect(
    "Select Approach Types",
    options=results["Approach Type"].unique(),
    default=results["Approach Type"].unique()
)

filtered_results = results[results["Approach Type"].isin(approach_types)]

# Summary metrics
st.header("Best Results")
col1, col2, col3, col4 = st.columns(4)

# Filter out None values for finding best
dept_acc_valid = filtered_results.dropna(subset=["Department Accuracy"])
dept_rec_valid = filtered_results.dropna(subset=["Department Recall"])
sen_acc_valid = filtered_results.dropna(subset=["Seniority Accuracy"])
sen_rec_valid = filtered_results.dropna(subset=["Seniority Recall"])

if not dept_acc_valid.empty:
    best_dept_acc = dept_acc_valid.loc[dept_acc_valid["Department Accuracy"].idxmax()]
    col1.metric("Best Dept Accuracy", f"{best_dept_acc['Department Accuracy']:.1%}", best_dept_acc["Approach Type"])

if not dept_rec_valid.empty:
    best_dept_rec = dept_rec_valid.loc[dept_rec_valid["Department Recall"].idxmax()]
    col2.metric("Best Dept Recall", f"{best_dept_rec['Department Recall']:.1%}", best_dept_rec["Approach Type"])

if not sen_acc_valid.empty:
    best_sen_acc = sen_acc_valid.loc[sen_acc_valid["Seniority Accuracy"].idxmax()]
    col3.metric("Best Sen Accuracy", f"{best_sen_acc['Seniority Accuracy']:.1%}", best_sen_acc["Approach Type"])

if not sen_rec_valid.empty:
    best_sen_rec = sen_rec_valid.loc[sen_rec_valid["Seniority Recall"].idxmax()]
    col4.metric("Best Sen Recall", f"{best_sen_rec['Seniority Recall']:.1%}", best_sen_rec["Approach Type"])

st.divider()

# Comparison charts
st.header("Model Comparison")
#tabs can be expanded maybe F1 etc
tab1, tab2 = st.tabs(["Accuracy", "Recall"])

with tab1:
    chart_data = filtered_results.melt(
        id_vars=["Method", "Approach Type"],
        value_vars=["Department Accuracy", "Seniority Accuracy"],
        var_name="Task",
        value_name="Accuracy"
    ).dropna()

    if not chart_data.empty:
        fig_acc = px.bar(
            chart_data,
            x="Method",
            y="Accuracy",
            color="Task",
            barmode="group",
            hover_data=["Approach Type"],
            title="Accuracy by Model and Task",
        )
        fig_acc.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
        st.plotly_chart(fig_acc, use_container_width=True)

with tab2:
    chart_data = filtered_results.melt(
        id_vars=["Method", "Approach Type"],
        value_vars=["Department Recall", "Seniority Recall"],
        var_name="Task",
        value_name="Recall"
    ).dropna()

    if not chart_data.empty:
        fig_rec = px.bar(
            chart_data,
            x="Method",
            y="Recall",
            color="Task",
            barmode="group",
            hover_data=["Approach Type"],
            title="Macro Recall by Model and Task",
        )
        fig_rec.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
        st.plotly_chart(fig_rec, use_container_width=True)

st.divider()

# Results table grouped by approach type
st.header("Detailed Results")

for approach in filtered_results["Approach Type"].unique():
    with st.expander(f"**{approach}**", expanded=True):
        subset = filtered_results[filtered_results["Approach Type"] == approach][
            ["Method", "Notebook", "Seniority Accuracy", "Seniority Recall", "Department Accuracy", "Department Recall"]
        ]
        st.dataframe(
            subset.style.format({
                #lambdas in order to ensure possible N/A
                "Seniority Accuracy": lambda x: f"{x:.1%}" if pd.notna(x) else "-",
                "Seniority Recall": lambda x: f"{x:.1%}" if pd.notna(x) else "-",
                "Department Accuracy": lambda x: f"{x:.1%}" if pd.notna(x) else "-",
                "Department Recall": lambda x: f"{x:.1%}" if pd.notna(x) else "-",
            }),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# Approach explanations
st.header("Approach Descriptions")

with st.expander("BoW (Bag of Words)"):
    st.markdown("""
    - **Model:** TF-IDF vectorizer (words or character n-grams) + Logistic Regression
    - **Training:** Trained on v2 label data
    - **Variants:** Different n-gram ranges, with/without confidence thresholds
    """)

with st.expander("XLM-R (Transformer Fine-tuning)"):
    st.markdown("""
    - **Model:** `xlm-roberta-base` transformer
    - **Training:** Fine-tuned on seniority-v2 data (90/10 split, 3 epochs)
    - **Variants:** With/without thresholding to "Professional", per-class thresholds
    """)

with st.expander("Embeddings"):
    st.markdown("""
    - **Model:** `paraphrase-multilingual-mpnet-base-v2` sentence embeddings + Logistic Regression
    - **Training:** Trained on v2 label data (50/50 split)
    - **Variants:** With/without confidence thresholds
    """)

with st.expander("Emb + Extended (Self-Training with Embeddings)"):
    st.markdown("""
    - **Model:** Sentence embeddings + Logistic Regression
    - **Training:** v2 data (capped 400-500) + pseudo-labeled LinkedIn data from BoW predictions
    - **Key insight:** Self-training helps model learn LinkedIn vocabulary
    - **Best results for Department classification**
    """)

with st.expander("BoW + Extended (Self-Training with BoW)"):
    st.markdown("""
    - **Model:** TF-IDF character n-grams + Logistic Regression
    - **Training:** v2 data (capped 500) + pseudo-labeled LinkedIn data from Embeddings predictions
    - **Best results for Seniority classification**
    """)

with st.expander("Career Heuristics"):
    st.markdown("""
    - **Model:** Sentence embeddings + Logistic Regression + 5 heuristic rules
    - **Heuristics:** Department consistency, seniority trajectory, organization context, career length, position count
    - **Result:** Minimal improvement (+1% dept, +0.2% seniority) — career paths are too messy for rigid rules
    """)

st.divider()

# Live Prediction Section
st.header("Try the Model (experimental, only carreer heuristics)")

st.markdown("Enter a job title to see how the model classifies it.")

# Load the embedding model
content_dir = Path("content")
model_path = content_dir / "extended_pipeline_model.joblib"

if model_path.exists():
    @st.cache_resource
    def load_prediction_model():
        model_data = joblib.load(model_path)
        # Load encoder
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_data['encoder_name'])
            return model_data, encoder
        except ImportError:
            return model_data, None

    model_data, encoder = load_prediction_model()

    if encoder is None:
        st.error("sentence-transformers not installed. Run: `pip install sentence-transformers`")
    else:
        dept_clf = model_data['dept_classifier']
        sen_clf = model_data['sen_classifier']

        # Input
        test_input = st.text_input("Job Title:", "Senior Software Engineer")

        if st.button("Classify", type="primary"):
            # Encode and predict
            embedding = encoder.encode([test_input])

            dept_proba = dept_clf.predict_proba(embedding)[0]
            sen_proba = sen_clf.predict_proba(embedding)[0]

            dept_pred = dept_clf.classes_[np.argmax(dept_proba)]
            sen_pred = sen_clf.classes_[np.argmax(sen_proba)]

            dept_conf = np.max(dept_proba)
            sen_conf = np.max(sen_proba)

            # Results
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Department", dept_pred, f"{dept_conf:.0%} confidence")

            with col2:
                st.metric("Seniority", sen_pred, f"{sen_conf:.0%} confidence")

            # Detailed probabilities
            st.subheader("All Probabilities")

            tab1, tab2 = st.tabs(["Department", "Seniority"])

            with tab1:
                dept_df = pd.DataFrame({
                    'Department': dept_clf.classes_,
                    'Probability': dept_proba
                }).sort_values('Probability', ascending=True)

                fig = go.Figure(go.Bar(
                    x=dept_df['Probability'],
                    y=dept_df['Department'],
                    orientation='h',
                    marker_color=['#2ecc71' if p == dept_pred else '#3498db' for p in dept_df['Department']]
                ))
                fig.update_layout(
                    title="Department Probabilities",
                    xaxis_title="Probability",
                    xaxis_tickformat=".0%",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                sen_df = pd.DataFrame({
                    'Seniority': sen_clf.classes_,
                    'Probability': sen_proba
                }).sort_values('Probability', ascending=True)

                fig = go.Figure(go.Bar(
                    x=sen_df['Probability'],
                    y=sen_df['Seniority'],
                    orientation='h',
                    marker_color=['#2ecc71' if p == sen_pred else '#e74c3c' for p in sen_df['Seniority']]
                ))
                fig.update_layout(
                    title="Seniority Probabilities",
                    xaxis_title="Probability",
                    xaxis_tickformat=".0%",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            # Confidence warning
            if dept_conf < 0.4 or sen_conf < 0.4:
                st.warning("Low confidence.")

else:
    st.warning("no model")
