# Ensemble Early Parkinson Detection Using Voice Biomarkers

**Course:** EECE5644 — Machine Learning and Pattern Recognition  
**Team:** Achintya Shah & Sheshang Ramesh

## Overview

A stacking ensemble classifier for early Parkinson's disease detection from voice biomarkers. Combines SVM, KNN, Random Forest, and Logistic Regression under a meta-learner framework, with rigorous subject-wise cross-validation.

## Dataset

Oxford Parkinson's Disease Detection Dataset (UCI ML Repository #174)  
- 195 recordings from 31 subjects (23 PD, 8 healthy)
- 22 biomedical voice features
- [Download here](https://archive.ics.uci.edu/dataset/174/parkinsons)

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/parkinsons-detection.git
cd parkinsons-detection

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Place dataset in data/ folder
mkdir -p data
# Copy parkinsons.data into data/
```

## Project Structure

```
notebooks/       # Jupyter notebooks (01–08, run in order)
src/             # Reusable utility modules
figures/         # Saved plots for the report
results/         # Experiment result CSVs
```

## Key Methodological Choices

- **Subject-wise GroupKFold CV** to prevent data leakage
- **Nested CV** for unbiased hyperparameter tuning
- **SMOTE applied only within training folds** via imbalanced-learn pipelines
- **PCA vs. RFE** feature selection comparison