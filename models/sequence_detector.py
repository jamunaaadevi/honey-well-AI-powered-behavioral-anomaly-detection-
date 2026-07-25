"""Sequence-aware anomaly detection: an LSTM autoencoder over per-entity event windows.

Trained only on normal-labeled training-split sequences (never sees attack labels, same
philosophy as the Isolation Forest). Reconstruction error on a held-out window is the
sequence anomaly score -- high error means "this run of the entity's last N actions
doesn't look like the sequences the model learned to reconstruct."
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from features.feature_engineering import FEATURE_COLUMNS

DEFAULT_CONFIG = {
    "window_size": 10,
    "hidden_size": 64,
    "num_layers": 1,
    "epochs": 12,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "seed": 42,
    "artifact_path": "models/artifacts/sequence_detector.pt",
}


def _build_entity_windows(feature_matrix, window_size):
    """feature_matrix: (n, F) for one entity's chronologically-sorted events.
    Returns (n, window_size, F) -- window i ends at event i, left zero-padded for i < window_size-1."""
    n, f = feature_matrix.shape
    padded = np.vstack([
        np.zeros((window_size - 1, f), dtype=np.float32),
        feature_matrix.astype(np.float32),
    ])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_size, axis=0)  # (n, f, window_size)
    return windows.transpose(0, 2, 1).copy()  # (n, window_size, f)


def build_sequences(df, feature_cols, window_size):
    """df must have user_id, timestamp, event_id + feature_cols. Returns (event_ids, windows)."""
    df = df.sort_values(["user_id", "timestamp"])
    event_id_chunks, window_chunks = [], []
    for _, group in df.groupby("user_id", sort=False):
        mat = group[feature_cols].to_numpy(dtype=np.float32)
        window_chunks.append(_build_entity_windows(mat, window_size))
        event_id_chunks.append(group["event_id"].to_numpy())
    event_ids = np.concatenate(event_id_chunks)
    windows = np.concatenate(window_chunks, axis=0)
    return event_ids, windows


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        last_hidden = h_n[-1]
        seq_len = x.size(1)
        repeated = last_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)


class SequenceAnomalyDetector:
    """Wraps the LSTM autoencoder: per-feature scaling + windowing + train/score/persist."""

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.feature_names_ = list(FEATURE_COLUMNS)
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None

    def _scale(self, windows):
        return (windows - self.scaler_mean) / self.scaler_scale

    def fit(self, train_df, verbose=True):
        cfg = self.config
        torch.manual_seed(cfg["seed"])

        normal_train = train_df[train_df["label"] == "normal"]
        raw = normal_train[self.feature_names_].to_numpy(dtype=np.float32)
        self.scaler_mean = raw.mean(axis=0)
        scale = raw.std(axis=0)
        self.scaler_scale = np.where(scale < 1e-6, 1.0, scale)

        _, windows = build_sequences(normal_train, self.feature_names_, cfg["window_size"])
        scaled = self._scale(windows)

        self.model = LSTMAutoencoder(len(self.feature_names_), cfg["hidden_size"], cfg["num_layers"])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(scaled))
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

        self.model.train()
        for epoch in range(cfg["epochs"]):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)
            avg_loss = total_loss / len(dataset)
            if verbose:
                print(f"  [sequence_detector] epoch {epoch + 1}/{cfg['epochs']}  recon_mse={avg_loss:.5f}")
        return self

    def score(self, df):
        """Returns a pandas Series of reconstruction-error anomaly scores, indexed by event_id."""
        event_ids, windows = build_sequences(df, self.feature_names_, self.config["window_size"])
        scaled = self._scale(windows)

        self.model.eval()
        errors = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(scaled), batch_size):
                batch = torch.from_numpy(scaled[i:i + batch_size])
                recon = self.model(batch)
                mse = ((recon - batch) ** 2).mean(dim=(1, 2))
                errors.append(mse.numpy())
        errors = np.concatenate(errors) if errors else np.array([])
        return pd.Series(errors, index=event_ids)

    def save(self, path=None):
        path = Path(path or self.config["artifact_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "feature_names": self.feature_names_,
            "config": self.config,
        }, path)
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or DEFAULT_CONFIG["artifact_path"])
        checkpoint = torch.load(path, weights_only=False)
        detector = cls(checkpoint["config"])
        detector.feature_names_ = checkpoint["feature_names"]
        detector.scaler_mean = checkpoint["scaler_mean"]
        detector.scaler_scale = checkpoint["scaler_scale"]
        detector.model = LSTMAutoencoder(
            len(detector.feature_names_), detector.config["hidden_size"], detector.config["num_layers"]
        )
        detector.model.load_state_dict(checkpoint["model_state"])
        detector.model.eval()
        return detector
