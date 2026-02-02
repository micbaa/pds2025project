# Job Title Classification

Classifying job titles into **Department** (11 classes) and **Seniority** (6 classes) using various ML approaches.

## Problem Statement

Given a job title like "Senior Softwareentwickler" or "Geschäftsführer", predict:
- **Department**: Which business function does this role belong to?
- **Seniority**: What level is this position?

### Challenges

1. **Multilingual titles**: Job titles appear in English, German, French, and other languages
2. **Domain shift**: Training data contains clean, standardized titles while LinkedIn data has creative, inconsistent formatting
3. **Class imbalance**: "Other" dominates department labels, "Professional" is underrepresented in training data
4. **Ambiguous titles**: Many titles could belong to multiple departments or seniority levels

## Best Results

| Task | Best Accuracy | Best Recall | Best Approach |
|------|---------------|-------------|---------------|
| Department | 69.9% | 58.4% | Embeddings + Self-Training |
| Seniority | 60.0% | 67.0% | BoW + Self-Training |

## Project Structure

```
├── self_training_models.ipynb      # Self-training with pseudo-labels (best results)
├── Bag_of_Words_Models.ipynb       # TF-IDF + Logistic Regression experiments
├── finetuned_xlmr_models.ipynb     # XLM-RoBERTa transformer fine-tuning
├── model_pipeline_extended.ipynb   # Career history heuristics
├── data_insights.ipynb             # Data visualization and analysis
├── Hai_PDSfinalproject_data.ipynb  # Rule-based baseline experiments
│
├── content/                        # Data files
│   ├── department-v2.csv           # Department training labels (~10,000)
│   ├── seniority-v2.csv            # Seniority training labels (~9,400)
│   ├── linkedin-cvs-annotated.json # Evaluation data (2,638 positions)
│   └── linkedin-cvs-not-annotated.json  # Unlabeled LinkedIn data (1,886 positions)
│
├── dashboard.py                    # Streamlit dashboard for model comparison
├── requirements.txt                # Python dependencies
└── README.md
```

## Approaches

### 1. Bag of Words (BoW)
**Notebook:** `Bag_of_Words_Models.ipynb`

- TF-IDF vectorizer with character n-grams (3,5) + Logistic Regression
- Fast to train, interpretable
- Best with confidence thresholds: predict "Other"/"Professional" when uncertain
- Results: 66.6% dept accuracy, 56.8% seniority accuracy (with thresholds)

### 2. XLM-RoBERTa Fine-tuning
**Notebook:** `finetuned_xlmr_models.ipynb`

- Multilingual transformer (`xlm-roberta-base`) fine-tuned on v2 data
- 90/10 train/test split, 3 epochs
- Struggles with domain shift — high accuracy on v2 test set (~99%) but poor on LinkedIn (~40%)
- Thresholding to "Professional" improves seniority: 37.7% → 39.0%

### 3. Sentence Embeddings
**Notebook:** `Bag_of_Words_Models.ipynb` (embeddings section)

- `paraphrase-multilingual-mpnet-base-v2` embeddings + Logistic Regression
- Good multilingual support out of the box
- Results: 59.6% dept accuracy (with threshold), 54.0% seniority accuracy

### 4. Self-Training (Best Results)
**Notebook:** `self_training_models.ipynb`

The key insight: training data (v2) doesn't represent LinkedIn vocabulary. Self-training bridges this gap.

**Process:**
1. Train base model (BoW or Embeddings) on v2 data
2. Generate pseudo-labels for 1,886 unlabeled LinkedIn titles
3. Combine v2 data + pseudo-labeled LinkedIn data (cap at 400-500 per class)
4. Retrain model on combined data
5. Evaluate on held-out annotated LinkedIn data (2,638 titles)

**Why it works:**
- The model learns LinkedIn-style vocabulary (German titles, abbreviations, creative formats)
- Even imperfect pseudo-labels help because they expose the model to the target domain
- The original v2 data anchors correct label meanings

**Results:**
| Variant | Dept Accuracy | Dept Recall | Sen Accuracy | Sen Recall |
|---------|---------------|-------------|--------------|------------|
| Emb + Extended (capped 500) | 69.9% | 58.4% | 57.4% | 63.0% |
| BoW + Extended (capped 500) | 63.0% | 50.0% | 60.0% | 67.0% |

### 5. Career History Heuristics
**Notebook:** `model_pipeline_extended.ipynb`

Attempted to improve predictions using career context:
- Department consistency across career
- Seniority should increase over time
- Organization type (university → "Other", bank → "Finance")
- Career length (15+ years ≠ Junior)

**Results:** Minimal improvement (+1% department, +0.2% seniority). Heuristics don't generalize well — career paths are messier than assumed.

### 6. Rule-Based Baseline
**Notebook:** `Hai_PDSfinalproject_data.ipynb`

Simple keyword matching:
- "Director" in title → Director seniority
- "Sales" in title → Sales department

**Results:** Low coverage (~25% of titles matched), but high precision when matched.

## Key Findings

1. **Self-training is the winner**: Adding pseudo-labeled LinkedIn data improved accuracy from ~40% to ~70%

2. **Simpler models can outperform transformers**: BoW + LogReg beat fine-tuned XLM-R when combined with self-training

3. **Confidence thresholds help**: Predicting "Other"/"Professional" for uncertain cases significantly improves accuracy

4. **Domain adaptation matters more than model complexity**: The gap between v2 and LinkedIn vocabulary was the main bottleneck

## Dashboard

Run the Streamlit dashboard to interactively compare all models:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install streamlit plotly sentence-transformers
streamlit run dashboard.py
```

Features:
- Filter by approach type
- Compare accuracy and recall across models
- View detailed results grouped by approach
- See which notebook each result comes from
- **Try the Model**: Enter any job title and get department/seniority predictions with confidence scores

## Data

| Dataset | Size | Description |
|---------|------|-------------|
| department-v2.csv | 10,145 rows | Curated job titles with department labels |
| seniority-v2.csv | 9,428 rows | Curated job titles with seniority labels |
| linkedin-cvs-annotated.json | 2,638 positions | Real LinkedIn titles with human labels (evaluation) |
| linkedin-cvs-not-annotated.json | 1,886 positions | Real LinkedIn titles without labels (self-training) |

## Labels

**Department (11 classes):**
- Administrative
- Business Development
- Consulting
- Customer Support
- Human Resources
- Information Technology
- Marketing
- Other (catch-all: legal, finance, executive, academia, healthcare)
- Project Management
- Purchasing
- Sales

**Seniority (6 classes):**
- Junior (entry-level, intern, trainee)
- Professional (mid-level, standard roles)
- Senior (senior title, specialist)
- Lead (team lead, principal)
- Management (manager, director of team)
- Director (C-suite, VP, board, executive)

## Requirements

```
numpy
pandas
scikit-learn
transformers
datasets
accelerate
evaluate
torch
sentence-transformers
lets-plot
scipy
```

## Future Work

- Add more training data for underrepresented classes ("Professional" seniority, minority departments)
- Ensemble approaches: combine BoW and Embeddings predictions
- Investigate per-class thresholds for better precision/recall trade-offs
- Collect more labeled LinkedIn data to reduce domain shift

**Dashboard extensions:**
- Upload unannotated JSON files and get predicted department/seniority for each job title
- Export predictions as CSV
- Show per-class precision/recall breakdown
- Compare predictions across multiple models side-by-side
- Visualize confidence scores for predictions
