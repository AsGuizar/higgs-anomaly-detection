# %% [markdown]
# # 01 — Exploratory Data Analysis
# ## Higgs Boson Anomaly Detection Project
#
# **Research context:**
# The HIGGS dataset (Baldi et al., 2014) contains 11 million simulated
# particle collision events. Each event is described by 28 kinematic
# features — 21 low-level detector measurements and 7 high-level features
# derived by physicists from theory.
#
# In this notebook we:
# 1. Load and inspect the dataset
# 2. Understand class imbalance
# 3. Visualize feature distributions and separability
# 4. Identify which features contain the most discriminating information
#
# **Key framing:** Labels are available in this dataset but will be
# withheld from all models during training. We use them here only to
# understand the data — and during final evaluation to measure performance.

# %% [markdown]
# ## 0. Setup

# %%
# Install dependencies (run once in Colab)
# !pip install ucimlrepo seaborn -q

import sys
sys.path.append('..')  # so we can import from src/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import roc_auc_score

from src.preprocess import (
    LOW_LEVEL_FEATURES, HIGH_LEVEL_FEATURES, ALL_FEATURES
)
from src.visualize import (
    plot_class_balance, plot_feature_distributions,
    plot_feature_separability, plot_correlation_matrix
)

plt.style.use('seaborn-v0_8-whitegrid')
print("Imports OK")

# %% [markdown]
# ## 1. Load Dataset

# %%
print("Fetching HIGGS dataset from UCI ML Repository...")
print("First download ~2.6GB, subsequent runs are cached.\n")

higgs = fetch_ucirepo(id=280)
X_raw = higgs.data.features
y_raw = higgs.data.targets.squeeze()

# Rename columns to meaningful physics names
X_raw.columns = ALL_FEATURES

df = pd.concat([X_raw, y_raw.rename('label')], axis=1)

print(f"Shape:             {df.shape}")
print(f"Features:          {len(ALL_FEATURES)}")
print(f"  Low-level:       {len(LOW_LEVEL_FEATURES)}")
print(f"  High-level:      {len(HIGH_LEVEL_FEATURES)}")
print(f"\nSignal events:     {(df.label==1).sum():>10,}  ({(df.label==1).mean()*100:.2f}%)")
print(f"Background events: {(df.label==0).sum():>10,}  ({(df.label==0).mean()*100:.2f}%)")

# %% [markdown]
# ## 2. Class Balance

# %%
plot_class_balance(
    df['label'].values,
    save_path='../results/figures/01_class_balance.png'
)

# %% [markdown]
# **Observation:** The dataset is approximately 47% signal / 53% background.
# Note that this is a *simulated* dataset — real detector data would have a
# far more extreme imbalance. In production searches, signal events may be
# 1 in 10^9 or rarer. Our balanced evaluation set will be constructed to
# assess model behavior at a meaningful signal fraction.

# %% [markdown]
# ## 3. Basic Statistics

# %%
print("── Background events ──")
display(df[df.label==0][ALL_FEATURES].describe().round(3))

# %%
print("── Signal events ──")
display(df[df.label==1][ALL_FEATURES].describe().round(3))

# %% [markdown]
# ## 4. Feature Distributions

# %%
# Sample for visualization speed (full dataset is 11M rows)
viz_sample = df.sample(n=200_000, random_state=42)

plot_feature_distributions(
    viz_sample,
    label_col='label',
    feature_cols=ALL_FEATURES,
    low_level_n=21,
    save_path='../results/figures/01_feature_distributions.png'
)

# %% [markdown]
# **Observation:** High-level features (yellow background) show visually
# cleaner separation between signal and background than low-level detector
# measurements. This is expected — physicists engineered these features
# specifically to capture the kinematic signatures of Higgs decay.
#
# The key question for our anomaly detectors: **can they rediscover this
# structure without being told which features matter?**

# %% [markdown]
# ## 5. Feature Separability

# %%
sep_series = plot_feature_separability(
    viz_sample,
    label_col='label',
    feature_cols=ALL_FEATURES,
    high_level_features=HIGH_LEVEL_FEATURES,
    save_path='../results/figures/01_feature_separability.png'
)

print("\nTop 7 most separating features:")
print(sep_series.tail(7).round(4).to_string())
print("\nBottom 7 least separating features:")
print(sep_series.head(7).round(4).to_string())

# %% [markdown]
# **Finding:** High-level features cluster at the top of the separability
# ranking. This is our first concrete result: **theory-derived features
# encode more discriminating information than raw detector measurements.**
#
# We will return to this finding in Notebook 05, where we ask whether
# the autoencoder's reconstruction error profile matches this ranking —
# i.e., whether the model "learns" to care about the same features that
# physicists care about, without being told to.

# %% [markdown]
# ## 6. Correlation Structure

# %%
bg_sample = df[df.label==0].sample(n=50_000, random_state=42)

plot_correlation_matrix(
    bg_sample,
    feature_cols=ALL_FEATURES,
    low_level_n=21,
    title_suffix='— Background Events',
    save_path='../results/figures/01_correlation_matrix_background.png'
)

# %%
sig_sample = df[df.label==1].sample(n=50_000, random_state=42)

plot_correlation_matrix(
    sig_sample,
    feature_cols=ALL_FEATURES,
    low_level_n=21,
    title_suffix='— Signal Events',
    save_path='../results/figures/01_correlation_matrix_signal.png'
)

# %% [markdown]
# **Observation:** The correlation structure differs between signal and
# background events — particularly in the high-level feature block (bottom
# right quadrant). This means signal events don't just differ in individual
# feature values — they have a *different covariance structure*.
# This is important: it means the autoencoder, which learns the covariance
# structure of background events, should produce high reconstruction errors
# for signal events not just because individual features are off, but
# because their *relationships* are wrong.

# %% [markdown]
# ## 7. Save Processed Splits
#
# We run `download.py` to save the splits used by all subsequent notebooks.
# This only needs to run once — outputs saved to `data/`.

# %%
# Run from repo root:
# !python data/download.py
#
# Or uncomment and run inline:

# import subprocess
# subprocess.run(['python', '../data/download.py'])

# %% [markdown]
# ---
# ## EDA Summary
#
# | Finding | Implication for modeling |
# |---------|--------------------------|
# | High-level features are more separating | AE may concentrate reconstruction error there |
# | Signal has different correlation structure | AE trained on background should fail on signal |
# | ~47% signal in full dataset | We use balanced test set; report PR-AUC not just ROC |
# | No missing values | No imputation needed |
#
# **Next:** `02_isolation_forest.ipynb` — our first anomaly detector.
