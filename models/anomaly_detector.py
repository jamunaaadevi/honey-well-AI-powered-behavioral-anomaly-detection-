"""Unsupervised anomaly scoring via Isolation Forest, trained only on normal behavior."""

from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest

DEFAULT_CONFIG = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": 0.025,
    "random_state": 42,
    "artifact_path": "models/artifacts/anomaly_detector.joblib",
}


class AnomalyDetector:
    """Wraps sklearn's IsolationForest; fit only on rows known to be normal."""

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = IsolationForest(
            n_estimators=self.config["n_estimators"],
            max_samples=self.config["max_samples"],
            contamination=self.config["contamination"],
            random_state=self.config["random_state"],
        )
        self.feature_names_ = None

    def fit(self, X):
        self.feature_names_ = list(X.columns)
        self.model.fit(X[self.feature_names_])
        return self

    def score(self, X):
        """Higher score = more anomalous (sign-flipped decision_function)."""
        return -self.model.decision_function(X[self.feature_names_])

    def save(self, path=None):
        path = Path(path or self.config["artifact_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or DEFAULT_CONFIG["artifact_path"])
        return joblib.load(path)
