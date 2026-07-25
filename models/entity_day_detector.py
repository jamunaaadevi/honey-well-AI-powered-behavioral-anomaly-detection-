"""Entity-day aggregated detector for slow-building patterns (e.g. low_and_slow_exfiltration).

No single event -- and no 30-min or even 7-day per-event rolling window -- fully captures a
pattern that is deliberately spread thin across 1-2 weeks. This aggregates one row per
(entity, day) instead of one row per event, trains a binary "was this entity's day part of a
suspicious slow-accumulation pattern" classifier, and at inference boosts the combined_risk of
that entity's events on flagged days -- a second, coarser-grained opinion layered on top of the
per-event pipeline rather than a replacement for it.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DEFAULT_CONFIG = {
    "n_estimators": 200,
    "max_depth": 8,
    "class_weight": "balanced",
    "random_state": 42,
    "artifact_path": "models/artifacts/entity_day_detector.joblib",
    "positive_label": "low_and_slow_exfiltration",
    "suspicious_threshold": 0.5,
    "risk_boost": 0.2,  # additive, in normalized [0,1] combined_risk space
}

ENTITY_DAY_FEATURE_COLUMNS = [
    "new_resources_today", "off_hours_count", "sensitive_access_count",
    "total_events", "distinct_resources", "mean_session_duration_z",
    "cumulative_new_resource_count_7d", "off_hours_access_count_7d", "sensitive_access_count_7d",
]


def build_entity_day_table(features_df, logs_df, positive_label):
    """One row per (user_id, day). features_df needs the standard FEATURE_COLUMNS + label;
    logs_df supplies the raw `resource` column (features.csv doesn't retain it) for a true
    distinct-resources-per-day count."""
    merged = features_df.merge(logs_df[["event_id", "resource"]], on="event_id", how="left")
    merged["day"] = merged["timestamp"].dt.normalize()

    grouped = merged.groupby(["user_id", "day"], sort=False)
    agg = grouped.agg(
        new_resources_today=("is_new_resource", "sum"),
        off_hours_count=("is_outside_usual_hours", "sum"),
        sensitive_access_count=("is_sensitive_resource", "sum"),
        total_events=("event_id", "count"),
        distinct_resources=("resource", "nunique"),
        mean_session_duration_z=("session_duration_zscore", "mean"),
        cumulative_new_resource_count_7d=("cumulative_new_resource_count_7d", "max"),
        off_hours_access_count_7d=("off_hours_access_count_7d", "max"),
        sensitive_access_count_7d=("sensitive_access_count_7d", "max"),
    ).reset_index()

    is_suspicious = grouped["label"].apply(lambda s: (s == positive_label).any())
    agg = agg.merge(is_suspicious.rename("is_suspicious_day").reset_index(), on=["user_id", "day"])
    return agg


class EntityDayDetector:
    """Wraps a binary RandomForestClassifier over entity-day aggregates."""

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = RandomForestClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            class_weight=self.config["class_weight"],
            random_state=self.config["random_state"],
        )
        self.feature_names_ = list(ENTITY_DAY_FEATURE_COLUMNS)

    def fit(self, entity_day_df):
        self.model.fit(entity_day_df[self.feature_names_], entity_day_df["is_suspicious_day"])
        return self

    def predict_suspicion(self, entity_day_df):
        """P(is_suspicious_day) per row, aligned with entity_day_df's index."""
        classes = list(self.model.classes_)
        pos_idx = classes.index(True)
        return self.model.predict_proba(entity_day_df[self.feature_names_])[:, pos_idx]

    def save(self, path=None):
        path = Path(path or self.config["artifact_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or DEFAULT_CONFIG["artifact_path"])
        return joblib.load(path)


def compute_risk_boost(test_df, entity_day_df, suspicion_scores, config):
    """Per-test-event additive risk boost: `risk_boost` if that event's (entity, day) was
    flagged suspicious, else 0. Returns a numpy array aligned with test_df's row order."""
    threshold = config["suspicious_threshold"]
    flagged = entity_day_df.loc[suspicion_scores > threshold, ["user_id", "day"]].copy()
    flagged["is_flagged"] = True

    lookup = test_df[["user_id", "timestamp"]].copy()
    lookup["day"] = lookup["timestamp"].dt.normalize()
    lookup = lookup.merge(flagged, on=["user_id", "day"], how="left")
    return np.where(lookup["is_flagged"].fillna(False).to_numpy(), config["risk_boost"], 0.0)
