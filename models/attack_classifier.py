"""Supervised attack-type classification via a balanced Random Forest."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

DEFAULT_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "class_weight": "balanced",
    "random_state": 42,
    "artifact_path": "models/artifacts/attack_classifier.joblib",
}


class AttackClassifier:
    """Wraps sklearn's RandomForestClassifier over normal + the 7 attack labels."""

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = RandomForestClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            class_weight=self.config["class_weight"],
            random_state=self.config["random_state"],
        )
        self.feature_names_ = None

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        self.model.fit(X[self.feature_names_], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.feature_names_])

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.feature_names_])

    def save(self, path=None):
        path = Path(path or self.config["artifact_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or DEFAULT_CONFIG["artifact_path"])
        return joblib.load(path)
