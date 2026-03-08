# %% [markdown]
# # 03 — Autoencoder Anomaly Detector
# ## Higgs Boson Anomaly Detection Project
#
# **Research question addressed:** RQ1 + RQ3
# - RQ1: Can unsupervised models detect rare physics events without labels?
# - RQ3: Do learned anomaly features correspond to known physics intuitions?
#
# **Model:** Symmetric Autoencoder (28→16→8→4→8→16→28)
# Trained to reconstruct background events. Signal events, which lie
# outside the learned manifold of normal physics, produce high
# reconstruction error — flagging them as anomalous.
#
# **Why autoencoders for physics?** The latent space (dim=4) forces the
# model to learn the most compact representation of normal collision
# kinematics. Signal events, having different kinematic signatures,
# cannot be compressed into this background manifold efficiently.

# %% [markdown]
# ## 0. Setup

# %%
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.preprocess import (
    load_splits, make_train_val_split,
    fit_scaler, scale, ALL_FEATURES,
    LOW_LEVEL_FEATURES, HIGH_LEVEL_FEATURES
)
from src.models import AutoencoderDetector
from src.evaluate import evaluate_model, plot_roc_curves, plot_score_distributions
from src.visualize import plot_per_feature_reconstruction_error

plt.style.use('seaborn-v0_8-whitegrid')
print(f"PyTorch: {torch.__version__}")
print(f"Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")

# %% [markdown]
# ## 1. Load Data

# %%
background, test_X, test_y = load_splits(
    bg_path='../data/background_sample.csv',
    test_path='../data/test_set.csv',
    labels_path='../data/test_labels.npy'
)

train, val = make_train_val_split(background, val_size=0.15)
scaler     = fit_scaler(train)

X_train = scale(train,  scaler)
X_val   = scale(val,    scaler)
X_test  = scale(test_X, scaler)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## 2. Train Autoencoder

# %%
ae = AutoencoderDetector(
    input_dim=28,
    lr=1e-3,
    batch_size=512,
    epochs=50,
    patience=5
)

ae.fit(X_train, X_val)

# %%
ae.plot_training_curves(
    save_path='../results/figures/03_ae_training_curves.png'
)

# %% [markdown]
# **Check:** Loss should decrease smoothly and val loss should track
# train loss closely. Large divergence indicates overfitting to background
# (unlikely with this architecture but worth checking).

# %% [markdown]
# ## 3. Score Test Set

# %%
scores_ae = ae.anomaly_score(X_test)

print(f"Score range: [{scores_ae.min():.6f}, {scores_ae.max():.6f}]")
print(f"Mean score (background): {scores_ae[test_y==0].mean():.6f}")
print(f"Mean score (signal):     {scores_ae[test_y==1].mean():.6f}")
print(f"Ratio signal/background: {scores_ae[test_y==1].mean() / scores_ae[test_y==0].mean():.2f}x")

# %% [markdown]
# ## 4. Evaluation

# %%
result_ae = evaluate_model('Autoencoder', scores_ae, test_y)

print(f"ROC-AUC:          {result_ae['roc_auc']}")
print(f"PR-AUC:           {result_ae['pr_auc']}")
print(f"FPR @ 90% recall: {result_ae['fpr_at_90_recall']}")

# %%
plot_roc_curves(
    [result_ae],
    save_path='../results/figures/03_roc_autoencoder.png'
)

# %%
plot_score_distributions(
    [result_ae],
    save_path='../results/figures/03_score_dist_autoencoder.png'
)

# %% [markdown]
# ## 5. Per-Feature Reconstruction Error (RQ3)
#
# This is where we connect ML back to physics.
# We ask: which features does the autoencoder fail to reconstruct
# for signal events, compared to background?
#
# If the answer matches the separability ranking from EDA (where
# high-level features were most discriminating), that means the model
# has rediscovered the same physical intuition — without being told.

# %%
# Sample for speed
n_analysis = 10_000
bg_idx  = np.where(test_y == 0)[0][:n_analysis]
sig_idx = np.where(test_y == 1)[0][:n_analysis]

errors_bg  = ae.per_feature_reconstruction_error(X_test[bg_idx])
errors_sig = ae.per_feature_reconstruction_error(X_test[sig_idx])

mean_errors_bg  = errors_bg.mean(axis=0)
mean_errors_sig = errors_sig.mean(axis=0)

# %%
plot_per_feature_reconstruction_error(
    mean_errors_bg,
    mean_errors_sig,
    feature_names=ALL_FEATURES,
    save_path='../results/figures/03_per_feature_reconstruction_error.png'
)

# %%
# Which features show the largest signal/background error ratio?
error_ratio = mean_errors_sig / (mean_errors_bg + 1e-10)
ratio_series = pd.Series(dict(zip(ALL_FEATURES, error_ratio))).sort_values(ascending=False)

print("Features where signal reconstruction error >> background")
print("(these are the features the autoencoder 'struggles' with for signal events)")
print()
print(ratio_series.round(3).to_string())

# %% [markdown]
# **Finding:** Features with the highest signal/background reconstruction
# error ratio should overlap substantially with the high-separability
# features from EDA (Notebook 01). If they do, the autoencoder has
# implicitly learned the same discriminating structure that physicists
# derived theoretically. This is a concrete answer to RQ3.

# %% [markdown]
# ## 6. Latent Space Visualization
#
# We project events through the encoder and visualize the 4D latent
# space using PCA. If background and signal cluster separately in latent
# space, the autoencoder has learned a structured representation of
# normal physics that signal events genuinely fall outside of.

# %%
from sklearn.decomposition import PCA as SklearnPCA

# Encode a sample of background and signal
n_viz = 5_000
viz_bg  = X_test[bg_idx[:n_viz]]
viz_sig = X_test[sig_idx[:n_viz]]

import torch
ae.model.eval()
with torch.no_grad():
    z_bg  = ae.model.encode(torch.FloatTensor(viz_bg).to(ae.device)).cpu().numpy()
    z_sig = ae.model.encode(torch.FloatTensor(viz_sig).to(ae.device)).cpu().numpy()

# PCA to 2D for visualization
pca_viz = SklearnPCA(n_components=2)
pca_viz.fit(z_bg)
z_bg_2d  = pca_viz.transform(z_bg)
z_sig_2d = pca_viz.transform(z_sig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(z_bg_2d[:,0],  z_bg_2d[:,1],
           alpha=0.15, s=5, color='#1E88E5', label='Background')
ax.scatter(z_sig_2d[:,0], z_sig_2d[:,1],
           alpha=0.15, s=5, color='#E53935', label='Signal')
ax.set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Autoencoder Latent Space (4D → 2D PCA)\n'
             'Signal vs Background distribution',
             fontsize=13, fontweight='bold')
ax.legend(markerscale=4)
plt.tight_layout()
plt.savefig('../results/figures/03_latent_space.png', dpi=150)
plt.show()

# %% [markdown]
# ## 7. Save Results

# %%
import pickle
with open('../results/ae_result.pkl', 'wb') as f:
    pickle.dump(result_ae, f)

# Also save the error ratios for the final analysis notebook
np.save('../results/ae_error_ratios.npy', error_ratio)
np.save('../results/ae_feature_names.npy', np.array(ALL_FEATURES))

print("Saved results.")

# %% [markdown]
# ---
# ## Autoencoder Summary
#
# | Metric | Value |
# |--------|-------|
# | ROC-AUC | (see above) |
# | PR-AUC | (see above) |
# | FPR @ 90% Recall | (see above) |
#
# **Next:** `04_evaluation.ipynb` — full comparison, cross-model agreement,
# and physics interpretation. This is where all three research questions
# are answered together.
