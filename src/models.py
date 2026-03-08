"""
models.py
---------
All anomaly detection model definitions in one place.
Each model exposes a consistent interface:
    .fit(X_train)
    .anomaly_score(X)  → higher = more anomalous

This unified interface is what makes the comparison in evaluate.py valid.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ── BASELINE 1: ISOLATION FOREST ─────────────────────────────────────────────

class IsolationForestDetector:
    """
    Isolation Forest anomaly detector.
    Fast, non-parametric, no GPU needed.
    Serves as the primary non-neural baseline.
    """

    def __init__(self, n_estimators=200, contamination='auto',
                 random_state=42, n_jobs=-1):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X_train):
        print("Training Isolation Forest...")
        self.model.fit(X_train)
        print("Done.")
        return self

    def anomaly_score(self, X):
        """
        Returns scores where HIGHER = more anomalous.
        sklearn's decision_function returns negative outlier scores,
        so we negate it.
        """
        return -self.model.decision_function(X)


# ── BASELINE 2: PCA RECONSTRUCTION ERROR ─────────────────────────────────────

class PCADetector:
    """
    PCA-based anomaly detector.
    Fits PCA on background, flags events with high reconstruction error.
    Simplest possible linear baseline.
    """

    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.pca = None
        self.n_components_ = None

    def fit(self, X_train):
        print(f"Fitting PCA (retaining {self.variance_threshold*100:.0f}% variance)...")
        self.pca = PCA(n_components=self.variance_threshold, random_state=42)
        self.pca.fit(X_train)
        self.n_components_ = self.pca.n_components_
        print(f"Using {self.n_components_} components.")
        return self

    def anomaly_score(self, X):
        """Reconstruction error = anomaly score."""
        X_reduced = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_reduced)
        return np.mean((X - X_reconstructed) ** 2, axis=1)


# ── MAIN MODEL: AUTOENCODER ───────────────────────────────────────────────────

class AutoencoderNet(nn.Module):
    """
    Symmetric autoencoder: 28 → 16 → 8 → 4 → 8 → 16 → 28
    Trained to reconstruct normal (background) events.
    Signal events reconstruct poorly → high reconstruction error → anomaly.
    """

    def __init__(self, input_dim=28):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            # Linear output — input is StandardScaled so values are unbounded
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector:
    """
    Wrapper for AutoencoderNet with fit/anomaly_score interface.
    """

    def __init__(self, input_dim=28, lr=1e-3, batch_size=512,
                 epochs=50, patience=5, device=None):
        self.input_dim  = input_dim
        self.lr         = lr
        self.batch_size = batch_size
        self.epochs     = epochs
        self.patience   = patience
        self.device     = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model      = None
        self.train_losses = []
        self.val_losses   = []

    def fit(self, X_train, X_val=None):
        print(f"Training Autoencoder on {self.device}...")
        print(f"  Architecture: {self.input_dim}→16→8→4→8→16→{self.input_dim}")
        print(f"  Epochs: {self.epochs}, Batch: {self.batch_size}, LR: {self.lr}")

        self.model = AutoencoderNet(self.input_dim).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion  = nn.MSELoss()

        # DataLoaders
        t_tensor = torch.FloatTensor(X_train)
        train_loader = DataLoader(
            TensorDataset(t_tensor, t_tensor),
            batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None:
            v_tensor = torch.FloatTensor(X_val)
            val_loader = DataLoader(
                TensorDataset(v_tensor, v_tensor),
                batch_size=self.batch_size
            )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, _ in train_loader:
                X_batch = X_batch.to(self.device)
                recon   = self.model(X_batch)
                loss    = criterion(recon, X_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(X_train)
            self.train_losses.append(train_loss)

            # Validate
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, _ in val_loader:
                        X_batch = X_batch.to(self.device)
                        recon   = self.model(X_batch)
                        val_loss += criterion(recon, X_batch).item() * len(X_batch)
                val_loss /= len(X_val)
                self.val_losses.append(val_loss)

                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d} | "
                          f"train: {train_loss:.5f} | val: {val_loss:.5f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    self._best_weights = {
                        k: v.clone()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d} | train: {train_loss:.5f}")

        # Restore best weights
        if hasattr(self, '_best_weights'):
            self.model.load_state_dict(self._best_weights)

        print("Done.")
        return self

    def anomaly_score(self, X):
        """Per-event reconstruction error = anomaly score."""
        self.model.eval()
        tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            recon = self.model(tensor)

        errors = torch.mean((tensor - recon) ** 2, dim=1)
        return errors.cpu().numpy()

    def per_feature_reconstruction_error(self, X):
        """
        Returns per-feature reconstruction error for a set of events.
        Used in feature analysis notebook to understand which features
        drive anomaly scores.
        """
        self.model.eval()
        tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            recon = self.model(tensor)

        per_feature = (tensor - recon) ** 2
        return per_feature.cpu().numpy()

    def plot_training_curves(self, save_path=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.train_losses, label='Train loss', color='#1E88E5')
        if self.val_losses:
            ax.plot(self.val_losses, label='Val loss',
                    color='#E53935', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Autoencoder Training Curves', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
