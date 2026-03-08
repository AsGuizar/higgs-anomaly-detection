"""
evaluate.py
-----------
Unified evaluation framework for all anomaly detection models.
All models are evaluated identically — this is what makes the
comparison table scientifically valid.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)


def evaluate_model(model_name, anomaly_scores, true_labels):
    """
    Core evaluation for any anomaly detection model.

    Parameters
    ----------
    model_name    : str   — display name for plots/tables
    anomaly_scores: array — higher = more anomalous
    true_labels   : array — 1=signal, 0=background

    Returns
    -------
    dict with roc_auc, pr_auc, and threshold analysis
    """
    roc_auc = roc_auc_score(true_labels, anomaly_scores)
    pr_auc  = average_precision_score(true_labels, anomaly_scores)

    # Threshold analysis: at 90% signal recall, what is FPR?
    fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
    idx_90 = np.argmin(np.abs(tpr - 0.90))
    fpr_at_90_recall = fpr[idx_90]

    return {
        'model':            model_name,
        'roc_auc':          round(roc_auc, 4),
        'pr_auc':           round(pr_auc, 4),
        'fpr_at_90_recall': round(fpr_at_90_recall, 4),
        # store curves for plotting
        '_fpr': fpr, '_tpr': tpr,
        '_scores': anomaly_scores,
        '_labels': true_labels,
    }


def plot_roc_curves(results_list, save_path=None):
    """
    Overlay ROC curves for all models on one plot.
    results_list: list of dicts from evaluate_model()
    """
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, r in enumerate(results_list):
        ax.plot(r['_fpr'], r['_tpr'],
                color=colors[i % len(colors)],
                linewidth=2.5,
                label=f"{r['model']}  (AUC = {r['roc_auc']:.4f})")

    ax.plot([0,1],[0,1], 'k--', alpha=0.4, label='Random chance (0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Signal Recall)', fontsize=12)
    ax.set_title('ROC Curves — Unsupervised Anomaly Detection\non HIGGS Dataset',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_score_distributions(results_list, save_path=None):
    """
    KDE plot of anomaly scores: signal vs background for each model.
    """
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]

    colors_bg  = '#1E88E5'
    colors_sig = '#E53935'

    for ax, r in zip(axes, results_list):
        scores = r['_scores']
        labels = r['_labels']

        bg_scores  = scores[labels == 0]
        sig_scores = scores[labels == 1]

        ax.hist(bg_scores,  bins=80, density=True, alpha=0.6,
                color=colors_bg,  label='Background')
        ax.hist(sig_scores, bins=80, density=True, alpha=0.6,
                color=colors_sig, label='Signal')

        ax.set_title(f'{r["model"]}\nAUC = {r["roc_auc"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.legend()

    plt.suptitle('Anomaly Score Distributions: Signal vs Background',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def comparison_table(results_list):
    """
    Print and return a clean comparison DataFrame.
    This is the main results table in the README.
    """
    rows = []
    for r in results_list:
        rows.append({
            'Model':              r['model'],
            'ROC-AUC':            r['roc_auc'],
            'PR-AUC':             r['pr_auc'],
            'FPR @ 90% Recall':   r['fpr_at_90_recall'],
        })

    df = pd.DataFrame(rows).set_index('Model')
    print("\n── Results Comparison ──────────────────────────────")
    print(df.to_string())
    print("────────────────────────────────────────────────────\n")
    return df


def cross_model_agreement(results_list, top_percentile=0.01, save_path=None):
    """
    Analyze which events are flagged as anomalous by each model.
    Events in top X% of anomaly scores are considered 'flagged'.
    Returns overlap analysis and Venn-style summary.
    """
    n_events = len(results_list[0]['_scores'])
    cutoff   = int(n_events * top_percentile)

    flagged = {}
    for r in results_list:
        top_indices = set(
            np.argsort(r['_scores'])[-cutoff:]
        )
        flagged[r['model']] = top_indices

    models = list(flagged.keys())
    print(f"\n── Cross-Model Agreement (top {top_percentile*100:.0f}% flagged) ──")

    # Pairwise overlap
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            a, b = flagged[models[i]], flagged[models[j]]
            overlap = len(a & b)
            pct = overlap / cutoff * 100
            print(f"  {models[i]} ∩ {models[j]}: {overlap:,} events ({pct:.1f}%)")

    # All models agree
    all_agree = set.intersection(*flagged.values())
    print(f"\n  All models agree: {len(all_agree):,} events")

    # Signal rate in high-confidence anomalies
    labels = results_list[0]['_labels']
    if labels is not None:
        sig_rate = np.mean([labels[i] for i in all_agree])
        print(f"  Signal rate in consensus anomalies: {sig_rate*100:.1f}%")
        print(f"  (Baseline signal rate in test set: "
              f"{np.mean(labels)*100:.1f}%)")

    return flagged
