"""
preprocess.py
-------------
Reusable preprocessing utilities used across all modeling notebooks.
Keeps normalization consistent between training and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── FEATURE METADATA ──────────────────────────────────────────────────────────
# Source: Baldi et al. (2014), "Searching for Exotic Particles in High-Energy
# Physics with Deep Learning", Nature Communications.

LOW_LEVEL_FEATURES = [
    'lepton_pT', 'lepton_eta', 'lepton_phi',
    'missing_energy_magnitude', 'missing_energy_phi',
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_b-tag',
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_b-tag',
    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_b-tag',
    'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_b-tag',
]

HIGH_LEVEL_FEATURES = [
    'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
]

ALL_FEATURES = LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES


def load_splits(bg_path='data/background_sample.csv',
                test_path='data/test_set.csv',
                labels_path='data/test_labels.npy'):
    """Load pre-saved data splits from download.py"""
    background = pd.read_csv(bg_path)
    test_X     = pd.read_csv(test_path)
    test_y     = np.load(labels_path)

    # Rename columns to meaningful names if dataset uses generic names
    if background.columns[0] != 'lepton_pT':
        background.columns = ALL_FEATURES
        test_X.columns     = ALL_FEATURES

    return background, test_X, test_y


def make_train_val_split(background_df, val_size=0.15, random_state=42):
    """
    Split background-only data into train and validation sets.
    Both sets contain only background — used to monitor training loss.
    """
    train, val = train_test_split(
        background_df,
        test_size=val_size,
        random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def fit_scaler(train_df):
    """Fit StandardScaler on training data. Returns fitted scaler."""
    scaler = StandardScaler()
    scaler.fit(train_df[ALL_FEATURES])
    return scaler


def scale(df, scaler):
    """Apply fitted scaler. Returns numpy array."""
    return scaler.transform(df[ALL_FEATURES]).astype(np.float32)


def get_feature_groups(df):
    """Return low-level and high-level feature subsets."""
    low  = df[[f for f in LOW_LEVEL_FEATURES  if f in df.columns]]
    high = df[[f for f in HIGH_LEVEL_FEATURES if f in df.columns]]
    return low, high
