"""SHAP explanations for alerted events, using the trained attack classifier."""

import numpy as np
import pandas as pd
import shap

from features.feature_engineering import FEATURE_COLUMNS
from models.attack_classifier import AttackClassifier
from models.attack_classifier import DEFAULT_CONFIG as CLASSIFIER_CONFIG

DEFAULT_CONFIG = {
    "predictions_path": "data/predictions.csv",
    "features_path": "data/features.csv",
    "classifier_artifact_path": CLASSIFIER_CONFIG["artifact_path"],
}


def _bool_template(true_text, false_text):
    return lambda v: true_text if bool(v) else false_text


FEATURE_TEMPLATES = {
    "has_baseline": _bool_template(
        "user has an established behavioral baseline",
        "user has little to no behavioral history yet (cold start)",
    ),
    "hour_of_day": lambda v: f"event occurred at {int(v):02d}:00",
    "is_weekend": _bool_template("activity occurred on a weekend", "activity occurred on a weekday"),
    "is_outside_usual_hours": _bool_template(
        "activity at an hour unusual for this user", "activity during this user's usual hours"
    ),
    "is_new_country": _bool_template(
        "login from a country never seen for this user", "login from a previously seen country"
    ),
    "is_new_device": _bool_template(
        "login from a device never seen for this user", "login from a known, previously seen device"
    ),
    "is_new_resource": _bool_template(
        "access to a resource never used by this user before", "access to a resource this user has used before"
    ),
    "geo_velocity_kmh": lambda v: f"location implies impossible travel speed of {v:.0f} km/h",
    "minutes_since_last_event": lambda v: f"only {v:.1f} minutes since this user's previous event",
    "failed_login_count_last_30min": lambda v: f"{int(v)} failed logins in the last 30 minutes",
    "total_event_count_last_30min": lambda v: f"{int(v)} events from this user in the last 30 minutes",
    "distinct_resources_last_30min": lambda v: f"accessed {int(v)} distinct resources in the last 30 minutes",
    "is_sensitive_resource": _bool_template(
        "accessed a sensitive/high-value resource", "resource accessed is not flagged as sensitive"
    ),
    "login_fail": _bool_template("this event was a failed login", "this event was not a failed login"),
    "user_typical_daily_event_count": lambda v: f"user's typical baseline is {v:.1f} events/day",
    "is_new_auth_method": _bool_template(
        "authenticated with a method never used by this entity before", "authentication method matches this entity's usual one"
    ),
    "session_duration_zscore": lambda v: (
        f"session duration is {v:.1f} standard deviations {'above' if v >= 0 else 'below'} this entity's usual length"
    ),
    "command_sequence_length": lambda v: f"ran a {int(v)}-command privileged session",
    "is_rare_command": _bool_template(
        "privileged session included a command never seen from this entity before",
        "privileged session commands all match this entity's history",
    ),
    "is_user": _bool_template("entity is a human user account", "entity is not a human user account"),
    "is_service_account": _bool_template(
        "entity is a service account (machine identity)", "entity is not a service account"
    ),
    "is_edge_device": _bool_template("entity is an edge/IoT device", "entity is not an edge device"),
    "distinct_entities_from_this_ip_30min": lambda v: (
        f"this source IP touched {int(v)} different accounts in 30 minutes" if v > 3
        else "activity from a single-account source IP"
    ),
    "failed_logins_from_this_ip_30min": lambda v: f"{int(v)} failed logins from this source IP in the last 30 minutes",
    "fail_rate_from_this_ip_30min": lambda v: f"{v * 100:.0f}% of this source IP's recent attempts failed",
    "cumulative_new_resource_count_7d": lambda v: f"accessed {int(v)} never-before-used resources over the past week",
    "off_hours_access_count_7d": lambda v: f"{int(v)} off-hours accesses in the past week",
    "sensitive_access_count_7d": lambda v: f"{int(v)} sensitive-resource accesses in the past week",
}


def _to_per_class_array(raw_shap_values, n_classes):
    """Normalize whatever shape shap.TreeExplainer returns to (n_samples, n_features, n_classes)."""
    if isinstance(raw_shap_values, list):
        return np.stack(raw_shap_values, axis=-1)
    arr = np.asarray(raw_shap_values)
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unexpected SHAP output shape {arr.shape} for a {n_classes}-class model")


class AlertExplainer:
    """Computes SHAP values once for the alerted subset, then answers explain_event() lookups."""

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.classifier = None
        self.class_list = None
        self.data = None  # alerted rows (predictions + features), indexed by event_id
        self.X = None  # feature matrix, same index
        self.shap_array = None  # (n_alerted, n_features, n_classes)
        self._row_pos = None  # event_id -> row position

    def fit(self):
        predictions = pd.read_csv(self.config["predictions_path"])
        alerted = predictions[predictions["alerted"]].reset_index(drop=True)
        features = pd.read_csv(self.config["features_path"], parse_dates=["timestamp"])
        merged = alerted.merge(features, on="event_id", how="left", suffixes=("", "_feat"))

        self.classifier = AttackClassifier.load(self.config["classifier_artifact_path"])
        self.class_list = list(self.classifier.model.classes_)

        X = merged[FEATURE_COLUMNS].astype(float)
        explainer = shap.TreeExplainer(self.classifier.model)
        # exact SHAP is impractical here: unbounded max_depth gives trees ~30 deep,
        # which blows up the exact tree-path algorithm; approximate (Saabas) is ~300x
        # faster and still ranks contributing features sensibly for our explanations.
        raw_shap_values = explainer.shap_values(X, approximate=True)
        self.shap_array = _to_per_class_array(raw_shap_values, len(self.class_list))

        merged = merged.set_index("event_id", drop=False)
        self.data = merged
        self.X = X.set_index(merged.index)
        self._row_pos = {event_id: i for i, event_id in enumerate(merged.index)}
        return self

    def explain_event(self, event_id, top_n=5):
        if self.data is None:
            raise RuntimeError("call .fit() before .explain_event()")
        if event_id not in self._row_pos:
            raise KeyError(f"{event_id} has no computed SHAP values (not an alerted event)")

        pos = self._row_pos[event_id]
        predicted_class = self.data.loc[event_id, "predicted_label"]
        if predicted_class == "unknown_anomaly":
            # Classifier itself said "normal" -- explain via what pushed AWAY from that
            # verdict, i.e. what the unsupervised IF safety net picked up on instead.
            class_idx = self.class_list.index("normal")
            sort_reverse = False
        else:
            class_idx = self.class_list.index(predicted_class)
            sort_reverse = True
        shap_row = self.shap_array[pos, :, class_idx]
        feature_values = self.X.loc[event_id]

        contributions = sorted(zip(FEATURE_COLUMNS, shap_row, feature_values), key=lambda t: t[1], reverse=sort_reverse)
        results = []
        for feature, sv, value in contributions[:top_n]:
            template = FEATURE_TEMPLATES.get(feature, lambda v, f=feature: f"{f} = {v}")
            results.append({
                "feature": feature,
                "shap_value": float(sv),
                "value": float(value),
                "sentence": template(value),
            })
        return results


_default_explainer = None


def explain_event(event_id, top_n=5, config=None):
    """Module-level convenience wrapper; lazily fits and caches a shared AlertExplainer."""
    global _default_explainer
    if _default_explainer is None:
        _default_explainer = AlertExplainer(config).fit()
    return _default_explainer.explain_event(event_id, top_n=top_n)


def main():
    explainer = AlertExplainer().fit()
    print(f"Computed SHAP values for {len(explainer.data)} alerted events.\n")

    sample_event_id = explainer.data.index[0]
    print(f"Example explanation for {sample_event_id} (predicted: {explainer.data.loc[sample_event_id, 'predicted_label']}):")
    for item in explainer.explain_event(sample_event_id, top_n=5):
        print(f"  [{item['shap_value']:+.4f}] {item['feature']}: {item['sentence']}")


if __name__ == "__main__":
    main()
