# CLAUDE.md — Parkinson's Voice Detection Project

## Project Overview
- **Course**: EECE5644 — Machine Learning and Pattern Recognition (Instructor: Vinay Ingle)
- **Team**: Achintya Shah and Sheshang Ramesh
- **Goal**: Stacking ensemble classifier for early Parkinson's disease detection using voice biomarkers
- **Dataset**: UCI Oxford Parkinson's Disease Detection Dataset (ID 174)
  - 195 sustained vowel recordings, **32 unique subjects** (not 31 as UCI states — verified by extracting subject IDs at index `[2]` from the `name` column)
  - 23 PD subjects, 8 healthy controls, ~6 recordings per subject
  - 22 real-valued acoustic features, binary target (`status`: 1=PD, 0=healthy)
  - Class imbalance: 147 PD (75.4%) vs 48 healthy (24.6%)
  - No missing values

## Environment
- Python 3.9.6, macOS (MacBook Air)
- Virtual environment via `venv` (not conda)
- IDE: VS Code with Jupyter notebooks
- Key libraries: scikit-learn, imbalanced-learn, seaborn, pandas, numpy, matplotlib

## Project Structure
```
parkinsons-detection/
├── data/parkinsons/parkinsons.data   # UCI dataset (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_preprocessing.ipynb        # Pipeline validation, leakage demo, baselines
│   ├── 03_imbalance_and_features.ipynb  # SMOTE vs class_weight; PCA vs RFE
│   ├── 04_hyperparameter_tuning.ipynb   # Nested CV for unbiased tuning
│   ├── 05_stacking_ensemble.ipynb       # Stacking with ablation-driven refinement
│   └── 06_final_results.ipynb           # (next notebook to build)
├── src/
│   ├── __init__.py
│   ├── data_utils.py           # load_parkinsons(), get_X_y_groups(), FEATURE_COLS, FEATURE_GROUPS
│   ├── eval_utils.py           # compute_metrics(), results_to_dataframe(), print_cv_summary()
│   └── pipeline_utils.py       # build_pipeline(), run_grouped_cv(), run_nested_cv()
├── figures/                    # Saved plots
├── results/                    # Experiment result CSVs
├── requirements.txt
├── .gitignore
└── README.md
```

## Critical Methodological Rules (DO NOT VIOLATE)

1. **Always use subject-wise GroupKFold** — never random/stratified splits. Multiple recordings per subject means random splitting causes data leakage. Group by `subject_id` extracted from the `name` column.

2. **All preprocessing inside CV folds** — StandardScaler and SMOTE must be fit only on training data within each fold. Use `imblearn.pipeline.Pipeline` (not sklearn's Pipeline) when SMOTE is involved.

3. **Nested cross-validation for tuning** — outer loop (5-fold GroupKFold) for performance estimation, inner loop (5-fold GroupKFold) for hyperparameter selection via GridSearchCV. Single-pass CV inflates results.

4. **SMOTE only within training folds** — never apply to full dataset before splitting.

## Key Findings (Verified by Experiments)

### Feature Selection & Imbalance (Notebook 03)
- **`class_weight='balanced'` + SVM-RFE consistently dominated** across all model types
- SMOTE did NOT outperform class weighting on this dataset
- PCA did NOT help — supervised feature selection (RFE) was superior
- High multicollinearity: `Jitter:DDP = 3 × MDVP:RAP`, `Shimmer:DDA = 3 × Shimmer:APQ3`
- Most discriminative features: PPE, spread1, RPDE, HNR (nonlinear dynamical measures)

### Best Configurations (from Notebook 03 → fed into Notebook 04)
- SVM_RBF, SVM_Linear, Logistic_L1, Logistic_L2 → SVM-RFE features + class_weight='balanced'
- KNN, Random_Forest → RF-RFE features + class_weight='balanced' (where supported)

### Hyperparameter Tuning (Notebook 04)
- Nested CV produces lower (more honest) estimates than single-pass CV
- Best individual model: **Logistic_L2** (balanced accuracy ~0.756)
- Expected realistic range with proper validation: 65–85% balanced accuracy

### Stacking Ensemble (Notebook 05)
- Three iterations of progressive improvement:
  1. Original (4 base learners, basic meta-features): BA = 0.658
  2. Improved (5 base learners incl. Logistic_L2, enriched meta-features, class-weighted meta-learner): BA = 0.700
  3. Refined (4 models, KNN removed via ablation): BA = 0.749
- KNN was actively hurting the ensemble — ablation showed removing it improved specificity from 0.583 → 0.700
- Enriched meta-features: probabilities + hard predictions + confidence margins (15 features instead of 5)
- Final stacking is competitive with but does not beat best individual model — this is an honest finding for small datasets

### Final Model Rankings
| Rank | Model | Balanced Acc | Sensitivity | Specificity |
|------|-------|-------------|-------------|-------------|
| 1 | Logistic_L2 (tuned) | 0.756 | 0.778 | 0.733 |
| 2 | Stacking (4-model refined) | 0.749 | 0.798 | 0.700 |
| 3 | Logistic_L1 (tuned) | 0.737 | 0.774 | 0.700 |

## Code Conventions

- All notebooks import shared utilities from `src/` via `sys.path.insert(0, '..')`
- Evaluation uses 8 metrics: accuracy, balanced accuracy, sensitivity, specificity, precision, F1, MCC, AUC-ROC
- Sensitivity is the priority metric (missing a PD case is worse than a false alarm)
- Use `balanced_accuracy` as the primary model selection criterion
- Results saved to `results/` as CSVs, figures to `figures/`

## Known Gotchas

- **Python 3.9 f-string nesting**: Cannot nest f-strings with matching quotes. Extract list comprehensions to a variable first.
- **seaborn boxplot deprecation**: Use `hue='status'` with `legend=False` instead of the deprecated `x`/`y` only syntax.
- **Subject ID extraction**: Use `name.split('_')[2]` (index 2, not 0 or 1) to get the correct 32 unique subject IDs.
- **imblearn vs sklearn Pipeline**: Must use `imblearn.pipeline.Pipeline` when SMOTE is in the pipeline — sklearn's Pipeline doesn't support `fit_resample`.

## What's Left
- Notebook 06: Final results compilation, publication-quality figures, statistical significance tests
- Report writing and presentation preparation
