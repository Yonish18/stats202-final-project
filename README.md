# Predicting Query-URL Relevance (Stanford University STATS202 Final Project)

Machine learning pipeline for predicting whether a URL is **relevant to a search query**, built using classification models and feature engineering techniques from statistical learning.

I developed this project for **STATS 202: Statistical Learning and Data Science (Stanford Summer Session)**.

The goal is to classify each query-URL pair as:

- **1 → Relevant**
- **0 → Not Relevant**

using engineered ranking signals and supervised learning models.

---

# Problem

Search engines must determine which URLs are most relevant for a given query. This project builds a predictive model that learns this relationship from labeled data.

Dataset:

- **80,046 training observations**
- **30,001 test observations**
- 10 base features describing the query‑URL pair

Examples of raw signals:

- `sig1` – `sig8` ranking signals
- `query_length`
- `is_homepage`

---

# Dataset Exploration

### Class Balance

The dataset is moderately imbalanced.

![Class Balance](report/figures/class_balance_bar.png)

Approximately:

- **56% not relevant**
- **44% relevant**

---

# Feature Engineering

Several transformations were applied to improve model performance.

### Log Transformations

Count‑like features such as `sig3` showed heavy right skew. A `log1p` transform stabilizes scale and reduces the influence of extreme values.

![Log Transform](report/figures/fe_log_demo_sig3_panel.png)

---

### Per‑Query Context Features

Signals were normalized within each query using:

- percentile ranks
- z‑scores

This captures **relative relevance within the query context**.

![Query Context Features](report/figures/fe_query_context_sig2_panel.png)

---

### Univariate Feature Strength

We evaluated features individually using ROC‑AUC.

Top predictors included:

- `sig2`
- `sig2_qz`
- `sig2_qrank`
- `log_sig3`

![Univariate AUC](report/figures/bar_univariate_auc_top20.png)

---

# Models

Two classification models were trained.

## HistGradientBoostingClassifier

Tree‑based gradient boosting.

Advantages:

- captures nonlinear relationships
- handles feature interactions automatically
- robust to feature scaling

Key parameters:

```
max_depth = 6
learning_rate = 0.06
max_iter = 350
```

---

## Logistic Regression

Standard statistical classification model.

Used for:

- interpretability
- coefficient inspection

Features were **standardized using StandardScaler**.

---

# Model Evaluation

### Validation Accuracy

| Model | Accuracy |
|------|------|
| HistGradientBoosting | **67.93%** |
| Logistic Regression | 65.30% |

---

### Confusion Matrices

#### Gradient Boosting

![HGB Confusion Matrix](report/figures/cm_val_hgb.png)

#### Logistic Regression

![Logistic Confusion Matrix](report/figures/cm_val_logit.png)

Gradient Boosting achieved higher overall accuracy and better balance across classes.

---

# Feature Importance

The most important features learned by the boosted model:

- `sig2`
- `sig6`
- `query_length`

![Feature Importances](report/figures/hgb_importances_top20.png)

---

# Logistic Regression Interpretation

### Positive Predictors

![Positive Coefficients](report/figures/logit_top_pos_coeffs.png)

### Negative Predictors

![Negative Coefficients](report/figures/logit_top_neg_coeffs.png)

These coefficients help explain which signals increase or decrease predicted relevance.

---

# Model Blending

Predictions from both models were combined:

```
p_blend = w * p_hgb + (1 - w) * p_logistic
```

Best validation configuration:

```
weight = 0.96
threshold = 0.498
```

Validation performance:

| Model | Accuracy |
|------|------|
| HGB baseline | 67.93% |
| Blended model | **67.96%** |

The blend produced small but measurable gains.

---

# Repository Structure

```
stats202-final-project
│
├── data/                # dataset files
├── notebooks/           # exploratory analysis
├── src/                 # modeling and feature engineering
│
├── report/
│   ├── STATS202_Final_Project_Report.pdf
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

# How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Train models:

```
python src/train_models.py
```

Generate predictions:

```
python src/generate_submission.py
```

---

# Key Takeaways

- **Feature engineering was the most important factor** in improving model performance.
- Per‑query normalization significantly improved signal usefulness.
- Tree‑based models captured nonlinear relationships better than logistic regression.
- Model blending provided marginal improvements by combining strengths of both approaches.

---

# Report

Full technical report:

```
report/STATS202_Final_Project_Report.pdf
```

---

# Author

**Yonish Tayal**  
Boston University — Computer Science

