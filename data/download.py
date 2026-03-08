"""
download.py
-----------
Downloads the HIGGS dataset from UCI ML Repository and saves
background/signal splits to disk for use across all notebooks.

Usage:
    python data/download.py

Outputs:
    data/background_sample.csv   — 500k background events (training pool)
    data/test_set.csv            — 100k balanced test set (signal + background)
    data/full_labels.npy         — labels for test set

Notes:
    Full dataset is 11M rows. We sample strategically to stay within
    Colab free tier RAM limits (~12GB). All modeling uses these splits.
"""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

RANDOM_SEED = 42
BACKGROUND_SAMPLE_SIZE = 500_000
TEST_SIZE = 100_000  # 50k signal + 50k background

def main():
    print("Fetching HIGGS dataset from UCI ML Repository...")
    print("(This may take a few minutes — 2.6GB download)")

    higgs = fetch_ucirepo(id=280)
    X = higgs.data.features
    y = higgs.data.targets.squeeze()

    print(f"Full dataset loaded: {X.shape[0]:,} events, {X.shape[1]} features")
    print(f"Signal events:     {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    print(f"Background events: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")

    df = pd.concat([X, y.rename('label')], axis=1)

    # ── SPLIT BACKGROUND AND SIGNAL ──────────────────────────────────
    background = df[df['label'] == 0].copy()
    signal     = df[df['label'] == 1].copy()

    # ── TRAINING POOL: background only ───────────────────────────────
    # Models are trained ONLY on background — this is the core
    # unsupervised framing. Labels are never seen during training.
    bg_sample = background.sample(
        n=BACKGROUND_SAMPLE_SIZE,
        random_state=RANDOM_SEED
    )

    # ── TEST SET: balanced signal + background ────────────────────────
    # Used only for evaluation. Labels withheld from models.
    half = TEST_SIZE // 2
    test_bg  = background.drop(bg_sample.index).sample(
        n=half, random_state=RANDOM_SEED
    )
    test_sig = signal.sample(n=half, random_state=RANDOM_SEED)
    test_set = pd.concat([test_bg, test_sig]).sample(
        frac=1, random_state=RANDOM_SEED  # shuffle
    ).reset_index(drop=True)

    # ── SAVE ──────────────────────────────────────────────────────────
    bg_path   = 'data/background_sample.csv'
    test_path = 'data/test_set.csv'

    bg_sample.drop(columns='label').to_csv(bg_path, index=False)
    test_set.drop(columns='label').to_csv(test_path, index=False)
    np.save('data/test_labels.npy', test_set['label'].values)

    print(f"\nSaved:")
    print(f"  {bg_path}  — {len(bg_sample):,} background events (training pool)")
    print(f"  {test_path} — {len(test_set):,} events (50/50 signal/background)")
    print(f"  data/test_labels.npy — labels for evaluation only")
    print(f"\nDone. You are ready to open 01_EDA.ipynb.")


if __name__ == '__main__':
    main()
