# %% [markdown]
# # 02 — Isolation Forest Baseline
# ## Higgs Boson Anomaly Detection Project
#
# **Research question addressed:** RQ1 — Can unsupervised models detect
# rare physics events without ever being told what they look like?
#
# **Model:** Isolation Forest (Liu et al., 2008)
# Random trees partition feature space. Events that are isolated in few
# splits are anomalous. No assumptions about data distribution.
#
# **Unsupervised protocol:** The model is trained ONLY on background events.
# Labels are never used during training. We evaluate on the held-out
# test set (50k signal + 50k background) at the very end.

# %% [markdown]
# ## 0. Setup

# %%
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess import (
    load_splits, make_train_val_split,
    fit_scaler, scale, ALL_FEATURES,
    LOW_LEVEL_FEATURES, HIGH_LEVEL_FEATURES
)
from src.models import IsolationForestDetector
from src.evaluate import evaluate_model, plot_roc_curves, plot_score_distributions

plt.style.use('seaborn-v0_8-whitegrid')
print("Imports OK")

# %% [markdown]
# ## 1. Load Data

# %%
# Load pre-saved splits from data/download.py
background, test_X, test_y = load_splits(
    bg_path='../data/background_sample.csv',
    test_path='../data/test_set.csv',
    labels_path='../data/test_labels.npy'
)

print(f"Background training pool: {len(background):,} events")
print(f"Test set:                 {len(test_X):,} events")
print(f"Test signal fraction:     {test_y.mean()*100:.1f}%")

# %% [markdown]
# ## 2. Preprocessing

# %%
train, val = make_train_val_split(background, val_size=0.15)

scaler   = fit_scaler(train)
X_train  = scale(train,  scaler)
X_val    = scale(val,    scaler)
X_test   = scale(test_X, scaler)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## 3. Train Isolation Forest
#
# Note: We use `contamination='auto'` which assumes no anomalies in the
# training set — consistent with our protocol of training on background only.

# %%
iforest = IsolationForestDetector(
    n_estimators=200,
    contamination='auto',
    random_state=42,
    n_jobs=-1
)

iforest.fit(X_train)

# %% [markdown]
# ## 4. Score the Test Set

# %%
scores_if = iforest.anomaly_score(X_test)

print(f"Score range: [{scores_if.min():.4f}, {scores_if.max():.4f}]")
print(f"Mean score (background): {scores_if[test_y==0].mean():.4f}")
print(f"Mean score (signal):     {scores_if[test_y==1].mean():.4f}")

# %% [markdown]
# ## 5. Evaluation

# %%
result_if = evaluate_model('Isolation Forest', scores_if, test_y)

print(f"ROC-AUC:          {result_if['roc_auc']}")
print(f"PR-AUC:           {result_if['pr_auc']}")
print(f"FPR @ 90% recall: {result_if['fpr_at_90_recall']}")

# %%
plot_roc_curves(
    [result_if],
    save_path='../results/figures/02_roc_isolation_forest.png'
)

# %%
plot_score_distributions(
    [result_if],
    save_path='../results/figures/02_score_dist_isolation_forest.png'
)

# %% [markdown]
# ## 6. Feature Importance via Permutation
#
# Which features contribute most to the anomaly scores?
# We shuffle each feature and measure the drop in AUC.
# Larger drop = feature was more important.

# %%
from sklearn.metrics import roc_auc_score

baseline_auc = result_if['roc_auc']
importances  = {}

print("Computing permutation importance (this takes ~2 min)...")

for i, feature in enumerate(ALL_FEATURES):
    X_permuted = X_test.copy()
    X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
    scores_perm = iforest.anomaly_score(X_permuted)
    perm_auc    = roc_auc_score(test_y, scores_perm)
    importances[feature] = baseline_auc - perm_auc

imp_series = pd.Series(importances).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 10))
colors = ['#E53935' if f in HIGH_LEVEL_FEATURES else '#1E88E5'
          for f in imp_series.index]
ax.barh(imp_series.index, imp_series.values, color=colors, alpha=0.85)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('AUC drop when feature is permuted\n(higher = more important)')
ax.set_title('Isolation Forest — Permutation Feature Importance\n'
             '(red = high-level, blue = low-level)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/02_if_feature_importance.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 5 most important features:")
print(imp_series.tail(5).round(5).to_string())

# %% [markdown]
# ## 7. Save Results for Comparison Notebook

# %%
import pickle
with open('../results/if_result.pkl', 'wb') as f:
    pickle.dump(result_if, f)

print("Saved to results/if_result.pkl")

# %% [markdown]
# ---
# ## Isolation Forest Summary
#
# | Metric | Value |
# |--------|-------|
# | ROC-AUC | (see above) |
# | PR-AUC | (see above) |
# | FPR @ 90% Recall | (see above) |
#
# **Interpretation:** An AUC significantly above 0.5 confirms RQ1's
# hypothesis: even a simple tree-based unsupervised method can detect
# signal events without labeled examples.
#
# **Next:** `03_autoencoder.ipynb` — our main model.
