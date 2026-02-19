"""
eval_utils.py — Evaluation metrics, confusion matrix plotting, and results formatting.

Usage:
    from src.eval_utils import compute_metrics, plot_confusion_matrix, results_to_dataframe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, classification_report
)


# ── Metric computation ───────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like — Ground truth labels (0 or 1).
    y_pred : array-like — Predicted labels (0 or 1).
    y_prob : array-like, optional — Predicted probabilities for the positive class.

    Returns
    -------
    dict with keys: accuracy, balanced_accuracy, sensitivity, specificity,
                    precision, f1, mcc, auc_roc (None if y_prob not provided).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,   # recall for PD
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,   # recall for healthy
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob) if y_prob is not None else None,
    }
    return metrics


def compute_cv_metrics(y_trues, y_preds, y_probs=None) -> pd.DataFrame:
    """
    Compute metrics across multiple CV folds and return mean ± std.

    Parameters
    ----------
    y_trues : list of arrays — Ground truth labels per fold.
    y_preds : list of arrays — Predicted labels per fold.
    y_probs : list of arrays, optional — Predicted probabilities per fold.

    Returns
    -------
    pd.DataFrame with columns: metric, mean, std
    """
    all_metrics = []
    for i in range(len(y_trues)):
        prob = y_probs[i] if y_probs is not None else None
        all_metrics.append(compute_metrics(y_trues[i], y_preds[i], prob))

    df = pd.DataFrame(all_metrics)
    summary = pd.DataFrame({
        'metric': df.columns,
        'mean': df.mean().values,
        'std': df.std().values
    })
    summary['mean_std'] = summary.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}" if r['mean'] is not None else "N/A",
        axis=1
    )
    return summary


# ── Confusion matrix plotting ────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix',
                          save_path=None, ax=None):
    """
    Plot a confusion matrix with counts and percentages.

    Parameters
    ----------
    y_true : array-like — Ground truth labels.
    y_pred : array-like — Predicted labels.
    title : str — Plot title.
    save_path : str or Path, optional — If provided, save figure to this path.
    ax : matplotlib Axes, optional — Axes to plot on.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    # Annotations with count and percentage
    total = cm.sum()
    annot = np.array([[f'{val}\n({val/total*100:.1f}%)' for val in row] for row in cm])

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', square=True,
                xticklabels=['Healthy', 'PD'], yticklabels=['Healthy', 'PD'],
                linewidths=1, ax=ax, cbar=False,
                annot_kws={'size': 14, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return ax


def plot_multiple_confusion_matrices(results_dict, save_path=None):
    """
    Plot confusion matrices for multiple models side by side.

    Parameters
    ----------
    results_dict : dict
        {model_name: (y_true, y_pred)} for each model.
    save_path : str or Path, optional
    """
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, (yt, yp)) in zip(axes, results_dict.items()):
        plot_confusion_matrix(yt, yp, title=name, ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── Results comparison table ─────────────────────────────────────────────────

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Convert a dict of {model_name: metrics_dict} into a formatted comparison table.

    Parameters
    ----------
    results : dict
        {model_name: dict of metric_name: value}

    Returns
    -------
    pd.DataFrame — Rows = models, Columns = metrics
    """
    df = pd.DataFrame(results).T
    df.index.name = 'Model'

    # Reorder columns for readability
    col_order = ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity',
                 'precision', 'f1', 'mcc', 'auc_roc']
    cols = [c for c in col_order if c in df.columns]
    return df[cols].round(4)


def print_cv_summary(model_name: str, summary_df: pd.DataFrame):
    """Pretty-print cross-validation results for a model."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Cross-Validation Results")
    print(f"{'='*60}")
    for _, row in summary_df.iterrows():
        print(f"  {row['metric']:>20s}: {row['mean_std']}")
    print(f"{'='*60}\n")
