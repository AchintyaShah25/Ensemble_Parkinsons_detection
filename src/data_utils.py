"""
data_utils.py — Data loading, subject ID extraction, and feature definitions.

Usage:
    from src.data_utils import load_parkinsons, FEATURE_COLS, FEATURE_GROUPS
"""

import pandas as pd
from pathlib import Path

# ── Feature definitions ──────────────────────────────────────────────────────

FEATURE_GROUPS = {
    'Fundamental Frequency': [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)'
    ],
    'Jitter': [
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
        'MDVP:PPQ', 'Jitter:DDP'
    ],
    'Shimmer': [
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
        'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA'
    ],
    'Nonlinear & Noise': [
        'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ],
}

# Flat list of all 22 feature columns
FEATURE_COLS = [f for group in FEATURE_GROUPS.values() for f in group]

# Target column
TARGET_COL = 'status'

# ── Redundant feature pairs (exact linear multiples) ─────────────────────────
# Jitter:DDP = 3 * MDVP:RAP,  Shimmer:DDA = 3 * Shimmer:APQ3
REDUNDANT_FEATURES = ['Jitter:DDP', 'Shimmer:DDA']


def load_parkinsons(data_path: str = None) -> pd.DataFrame:
    """
    Load the UCI Parkinsons dataset and add a subject_id column.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to parkinsons.data file.
        Defaults to '<project_root>/data/parkinsons/parkinsons.data'.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus 'subject_id'.
    """
    if data_path is None:
        # Assumes this file lives in src/, project root is one level up
        data_path = Path(__file__).parent.parent / 'data' / 'parkinsons' / 'parkinsons.data'
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Extract subject ID: 'phon_R01_S01_1' → 'S01'
    df['subject_id'] = df['name'].apply(lambda x: x.split('_')[2])

    return df


def get_X_y_groups(df: pd.DataFrame):
    """
    Extract feature matrix, target vector, and group labels from the DataFrame.

    Returns
    -------
    X : pd.DataFrame of shape (n_samples, 22)
    y : pd.Series of shape (n_samples,)
    groups : pd.Series of shape (n_samples,)  — subject IDs for GroupKFold
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    groups = df['subject_id'].copy()
    return X, y, groups
