"""
visualize.py
------------
All plotting functions used across notebooks.
Consistent style, consistent color palette, consistent figure sizes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# ── GLOBAL STYLE ─────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE   = {'background': '#1E88E5', 'signal': '#E53935'}
FIG_DPI   = 150


def plot_class_balance(labels, save_path=None):
    counts = pd.Series(labels).value_counts().sort_index()
    names  = ['Background', 'Signal']
    colors = [PALETTE['background'], PALETTE['signal']]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, counts.values, color=colors, alpha=0.85,
                  edgecolor='white', width=0.5)
    ax.set_title('Class Distribution — HIGGS Dataset',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Event Count')

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + counts.max()*0.01,
                f'{val:,}\n({val/counts.sum()*100:.1f}%)',
                ha='center', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI)
    plt.show()


def plot_feature_distributions(df, label_col, feature_cols,
                                low_level_n=21, save_path=None):
    n_features = len(feature_cols)
    ncols = 7
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*3))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        bg  = df[df[label_col] == 0][col]
        sig = df[df[label_col] == 1][col]

        axes[i].hist(bg,  bins=60, alpha=0.6, density=True,
                     color=PALETTE['background'], label='Background')
        axes[i].hist(sig, bins=60, alpha=0.6, density=True,
                     color=PALETTE['signal'],     label='Signal')
        axes[i].set_title(col, fontsize=8, fontweight='bold')
        axes[i].set_yticks([])

        # Highlight high-level features
        if i >= low_level_n:
            axes[i].set_facecolor('#fffde7')

        if i == 0:
            axes[i].legend(fontsize=7)

    # Hide unused axes
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        'Feature Distributions: Signal vs Background\n'
        '(yellow background = high-level derived physics features)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def plot_feature_separability(df, label_col, feature_cols,
                               high_level_features, save_path=None):
    """
    For each feature, compute single-feature AUC as a separability score.
    Higher = feature alone does better at separating signal from background.
    """
    from sklearn.metrics import roc_auc_score

    sep = {}
    for col in feature_cols:
        auc = roc_auc_score(df[label_col], df[col])
        sep[col] = max(auc, 1 - auc)

    sep_series = pd.Series(sep).sort_values(ascending=True)
    colors = [PALETTE['signal'] if c in high_level_features
              else PALETTE['background']
              for c in sep_series.index]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(sep_series.index, sep_series.values,
            color=colors, alpha=0.85, edgecolor='white')
    ax.axvline(0.5, color='black', linestyle='--',
               alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('Single-Feature AUC', fontsize=11)
    ax.set_title(
        'Feature Separability: Signal vs Background\n'
        '(red = high-level, blue = low-level)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

    return sep_series


def plot_correlation_matrix(df, feature_cols, low_level_n=21,
                             title_suffix='', save_path=None):
    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax,
                linewidths=0.3, cbar_kws={"shrink": 0.8})
    ax.axhline(low_level_n, color='black', linewidth=2)
    ax.axvline(low_level_n, color='black', linewidth=2)
    ax.set_title(
        f'Feature Correlation Matrix {title_suffix}\n'
        '(line separates low-level / high-level features)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI)
    plt.show()


def plot_per_feature_reconstruction_error(mean_errors_bg, mean_errors_sig,
                                           feature_names, save_path=None):
    """
    Compare per-feature reconstruction error for background vs signal events.
    Features where signal >> background are the ones the autoencoder
    'finds hard' — these are the physically interesting ones.
    """
    x     = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width/2, mean_errors_bg,  width, label='Background',
           color=PALETTE['background'], alpha=0.85)
    ax.bar(x + width/2, mean_errors_sig, width, label='Signal',
           color=PALETTE['signal'],     alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Reconstruction Error')
    ax.set_title(
        'Per-Feature Reconstruction Error: Signal vs Background\n'
        'Features where signal > background are "hard" for the autoencoder',
        fontsize=13, fontweight='bold'
    )
    ax.axvline(20.5, color='black', linewidth=1.5, linestyle='--',
               label='Low / High level boundary')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()
