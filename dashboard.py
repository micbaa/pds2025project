import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Job Title Classification - Model Comparison", layout="wide")

st.title("Job Title Classification Dashboard")

# Model results data - grouped by approach type
# When updating beware of the order. Since everything is hardcoded
results = pd.DataFrame({
    "Approach Type": [
        # BoW
        "BoW",
        "BoW",
        "BoW",
        # XLM-R
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
        "TF-IDF char n-grams (3,5) + LR",
        "TF-IDF char_wb (3,5) + LR + threshold (0.50)",
        "TF-IDF char n-grams (3,5) + LR + threshold (0.40) → Other",
        # XLM-R
        "seniority-v2 90/10, 3 epochs (no threshold)",
        "seniority-v2 90/10, 3 epochs + threshold 0.55 → Professional",
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
        "Bag_of_Words_Models.ipynb",
        # XLM-R
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
        0.3582,
        0.5679,
        None,
        # XLM-R
        0.3772,
        0.3901,
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
        0.56,
        0.60,
        None,
        # XLM-R
        0.6189,
        0.6233,
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
        0.3302,
        None,
        0.6660,
        # XLM-R
        None,
        None,
        # Embeddings
        0.4105,
        0.5963,
        # Emb with extended
        0.6994,
        0.6903,
        # BoW with extended
        0.63,
        # Career Heuristics
        0.559,
    ],
    "Department Recall": [
        # BoW
        0.47,
        None,
        0.48,
        # XLM-R
        None,
        None,
        # Embeddings
        0.5555,
        0.4789,
        # Emb with extended
        0.5836,
        0.5709,
        # BoW with extended
        0.50,
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
    - **Variants:** With/without thresholding to "Professional"
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
