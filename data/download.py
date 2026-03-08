# ══════════════════════════════════════════════════════════════════
# download.py
#
# Downloads the HIGGS dataset and saves background/signal splits
# to disk for use across all notebooks.
#
# Usage:
#   python data/download.py
#
# Outputs:
#   data/background_sample.csv  — 500k background events (training pool)
#   data/test_set.csv           — 100k balanced test set
#   data/test_labels.npy        — labels for test set (evaluation only)
#
# Note: labels are NEVER passed to models during training.
# They are saved here only for final evaluation.
# ══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

RANDOM_SEED = 42
BACKGROUND_SAMPLE_SIZE = 500_000
TEST_SIZE = 100_000  # 50k signal + 50k background

LOW_LEVEL_FEATURES = [
    'lepton_pT', 'lepton_eta', 'lepton_phi',
    'missing_energy_magnitude', 'missing_energy_phi',
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_b-tag',
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_b-tag',
    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_b-tag',
    'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_b-tag',
]
HIGH_LEVEL_FEATURES = ['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
ALL_FEATURES = LOW_LEVEL_FEATURES + HIGH_LEVEL_FEATURES


def main():
    data_dir  = Path('data')
    data_path = data_dir / 'HIGGS.csv.gz'

    # ── DOWNLOAD IF NEEDED ────────────────────────────────────────
    if not data_path.exists():
        print("Downloading HIGGS dataset (~2.6GB)...")
        subprocess.run([
            'wget', '-O', str(data_path),
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
        ], check=True)
        print("Download complete.")

    # ── LOAD ──────────────────────────────────────────────────────
    print("Loading dataset...")
    col_names = ['label'] + ALL_FEATURES
    df = pd.read_csv(data_path, header=None, names=col_names)
    df['label'] = df['label'].astype(int)

    print(f"Loaded: {df.shape[0]:,} events, {df.shape[1]-1} features")
    print(f"Signal:     {(df.label==1).sum():,} ({(df.label==1).mean()*100:.1f}%)")
    print(f"Background: {(df.label==0).sum():,} ({(df.label==0).mean()*100:.1f}%)")

    # ── SPLIT ─────────────────────────────────────────────────────
    background = df[df['label'] == 0].copy()
    signal     = df[df['label'] == 1].copy()

    # Training pool — background only, labels never seen by models
    bg_sample = background.sample(n=BACKGROUND_SAMPLE_SIZE, random_state=RANDOM_SEED)

    # Test set — balanced signal + background, labels saved separately
    half     = TEST_SIZE // 2
    test_bg  = background.drop(bg_sample.index).sample(n=half, random_state=RANDOM_SEED)
    test_sig = signal.sample(n=half, random_state=RANDOM_SEED)
    test_set = pd.concat([test_bg, test_sig]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # ── SAVE ──────────────────────────────────────────────────────
    bg_sample.drop(columns='label').to_csv(data_dir / 'background_sample.csv', index=False)
    test_set.drop(columns='label').to_csv(data_dir / 'test_set.csv', index=False)
    np.save(data_dir / 'test_labels.npy', test_set['label'].values)

    print(f"\nSaved:")
    print(f"  data/background_sample.csv — {len(bg_sample):,} background events")
    print(f"  data/test_set.csv          — {len(test_set):,} events (50/50 split)")
    print(f"  data/test_labels.npy       — labels for evaluation only")
    print(f"\nDone. Ready for notebook 02.")


if __name__ == '__main__':
    main()