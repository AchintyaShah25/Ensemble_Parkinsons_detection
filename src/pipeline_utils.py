"""
pipeline_utils.py — Reusable pipeline builders, cross-validation, and hyperparameter tuning.

Key design decisions:
  - Uses imblearn.pipeline.Pipeline (not sklearn's) to support SMOTE's fit_resample.
  - All scaling and resampling happens INSIDE CV folds to prevent data leakage.
  - GroupKFold ensures subject-wise splitting throughout.

Usage:
    from src.pipeline_utils import (
        build_pipeline, run_grouped_cv, run_nested_cv, get_default_param_grids
    )
"""

import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.eval_utils import compute_metrics


# ── Pipeline builders ────────────────────────────────────────────────────────

def build_pipeline(classifier, use_smote=False, use_pca=False, n_components=10,
                   smote_random_state=42):
    """
    Build an imblearn Pipeline with optional scaling, SMOTE, PCA, and a classifier.

    Parameters
    ----------
    classifier : sklearn estimator — The classifier to use.
    use_smote : bool — Whether to include SMOTE oversampling.
    use_pca : bool — Whether to include PCA dimensionality reduction.
    n_components : int — Number of PCA components (only used if use_pca=True).
    smote_random_state : int — Random state for SMOTE reproducibility.

    Returns
    -------
    ImbPipeline
    """
    steps = [('scaler', StandardScaler())]

    if use_smote:
        steps.append(('smote', SMOTE(random_state=smote_random_state)))

    if use_pca:
        steps.append(('pca', PCA(n_components=n_components)))

    steps.append(('clf', classifier))

    return ImbPipeline(steps)


def get_classifiers(class_weight=None, random_state=42):
    """
    Return a dict of default classifiers.

    Parameters
    ----------
    class_weight : str or None
        Pass 'balanced' to enable class weighting for SVM and LR.
    random_state : int

    Returns
    -------
    dict of {name: classifier}
    """
    cw = class_weight  # shorthand

    classifiers = {
        'SVM_RBF': SVC(kernel='rbf', class_weight=cw, probability=True,
                       random_state=random_state),
        'SVM_Linear': SVC(kernel='linear', class_weight=cw, probability=True,
                          random_state=random_state),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Random_Forest': RandomForestClassifier(n_estimators=100, class_weight=cw,
                                                random_state=random_state),
        'Logistic_L1': LogisticRegression(penalty='l1', solver='saga', max_iter=5000,
                                          class_weight=cw, random_state=random_state),
        'Logistic_L2': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000,
                                          class_weight=cw, random_state=random_state),
    }
    return classifiers


# ── Cross-validation with subject-wise grouping ─────────────────────────────

def run_grouped_cv(pipeline, X, y, groups, n_splits=5, random_state=None):
    """
    Run GroupKFold cross-validation and collect per-fold predictions and metrics.

    Parameters
    ----------
    pipeline : Pipeline — sklearn/imblearn pipeline to evaluate.
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    groups : array-like of shape (n_samples,) — Subject IDs for grouping.
    n_splits : int — Number of CV folds.
    random_state : int, optional — Not used by GroupKFold (deterministic),
                   kept for API consistency.

    Returns
    -------
    dict with keys:
        'y_trues'  : list of arrays — true labels per fold
        'y_preds'  : list of arrays — predicted labels per fold
        'y_probs'  : list of arrays — predicted probabilities per fold (or None)
        'fold_metrics' : list of dicts — metrics per fold
        'mean_metrics' : dict — mean metrics across folds
        'std_metrics'  : dict — std of metrics across folds
    """
    gkf = GroupKFold(n_splits=n_splits)

    y_trues, y_preds, y_probs = [], [], []
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # Get probabilities if the classifier supports it
        y_prob = None
        if hasattr(pipeline, 'predict_proba'):
            try:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                pass

        y_trues.append(y_test.values)
        y_preds.append(y_pred)
        y_probs.append(y_prob)

        fold_metrics.append(compute_metrics(y_test, y_pred, y_prob))

    # Aggregate metrics
    metrics_df = {k: [m[k] for m in fold_metrics] for k in fold_metrics[0]}
    mean_metrics = {k: np.mean([v for v in vals if v is not None])
                    for k, vals in metrics_df.items()}
    std_metrics = {k: np.std([v for v in vals if v is not None])
                   for k, vals in metrics_df.items()}

    return {
        'y_trues': y_trues,
        'y_preds': y_preds,
        'y_probs': y_probs,
        'fold_metrics': fold_metrics,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
    }


# ── Nested cross-validation for unbiased tuning ─────────────────────────────

def get_default_param_grids():
    """
    Return default hyperparameter grids for each classifier.
    Keys use the pipeline prefix 'clf__'.

    Returns
    -------
    dict of {model_name: param_grid}
    """
    return {
        'SVM_RBF': {
            'clf__C': [0.1, 1, 10, 100],
            'clf__gamma': ['scale', 'auto', 0.01, 0.1],
        },
        'SVM_Linear': {
            'clf__C': [0.01, 0.1, 1, 10, 100],
        },
        'KNN': {
            'clf__n_neighbors': [3, 5, 7, 9, 11],
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['euclidean', 'manhattan'],
        },
        'Random_Forest': {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 5, 10, 20],
            'clf__min_samples_split': [2, 5, 10],
        },
        'Logistic_L1': {
            'clf__C': [0.01, 0.1, 1, 10, 100],
        },
        'Logistic_L2': {
            'clf__C': [0.01, 0.1, 1, 10, 100],
        },
    }


def run_nested_cv(pipeline, param_grid, X, y, groups,
                  outer_splits=5, inner_splits=5,
                  scoring='balanced_accuracy', n_jobs=-1):
    """
    Run nested cross-validation: outer loop for evaluation, inner loop for tuning.

    Parameters
    ----------
    pipeline : Pipeline — The pipeline to tune and evaluate.
    param_grid : dict — Hyperparameter grid for GridSearchCV.
    X, y, groups : array-like — Features, target, subject IDs.
    outer_splits : int — Number of outer CV folds.
    inner_splits : int — Number of inner CV folds for GridSearchCV.
    scoring : str — Metric to optimize in the inner loop.
    n_jobs : int — Parallel jobs for GridSearchCV.

    Returns
    -------
    dict with same structure as run_grouped_cv, plus:
        'best_params' : list of best params per outer fold
    """
    outer_cv = GroupKFold(n_splits=outer_splits)
    inner_cv = GroupKFold(n_splits=inner_splits)

    y_trues, y_preds, y_probs = [], [], []
    fold_metrics = []
    best_params = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        # Inner loop: hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=inner_cv.split(X_train, y_train, groups_train),
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train, y_train)

        best_params.append(grid_search.best_params_)

        # Evaluate on outer test fold
        y_pred = grid_search.predict(X_test)
        y_prob = None
        if hasattr(grid_search, 'predict_proba'):
            try:
                y_prob = grid_search.predict_proba(X_test)[:, 1]
            except Exception:
                pass

        y_trues.append(y_test.values)
        y_preds.append(y_pred)
        y_probs.append(y_prob)
        fold_metrics.append(compute_metrics(y_test, y_pred, y_prob))

    # Aggregate
    metrics_df = {k: [m[k] for m in fold_metrics] for k in fold_metrics[0]}
    mean_metrics = {k: np.mean([v for v in vals if v is not None])
                    for k, vals in metrics_df.items()}
    std_metrics = {k: np.std([v for v in vals if v is not None])
                   for k, vals in metrics_df.items()}

    return {
        'y_trues': y_trues,
        'y_preds': y_preds,
        'y_probs': y_probs,
        'fold_metrics': fold_metrics,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'best_params': best_params,
    }


# ── Convenience: run all baseline models ─────────────────────────────────────

def run_all_baselines(X, y, groups, use_smote=False, class_weight=None,
                      n_splits=5, use_pca=False, n_components=10):
    """
    Run grouped CV for all default classifiers and return results.

    Returns
    -------
    dict of {model_name: cv_results_dict}
    """
    classifiers = get_classifiers(class_weight=class_weight)
    all_results = {}

    for name, clf in classifiers.items():
        print(f"  Running {name}...")
        pipe = build_pipeline(clf, use_smote=use_smote, use_pca=use_pca,
                              n_components=n_components)
        results = run_grouped_cv(pipe, X, y, groups, n_splits=n_splits)
        all_results[name] = results

    return all_results
