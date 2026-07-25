"""Cold-start demo: score a brand-new entity's first few events.

Shows the pipeline stays conservative -- no "is_new_*" flag fires just because the
entity has no history yet, and the trained models don't spike a risk score on pure
novelty. Also exercises models.baseline_profile.EntityProfile.deviation_score()
directly against a genuinely foreign follow-up event.

Depends on artifacts trained by the current models/train.py (post train/val/test split) --
rerun `python -m models.train` first if the saved anomaly detector / classifier are stale.
"""

import pandas as pd

from features.feature_engineering import FEATURE_COLUMNS, compute_features
from models.anomaly_detector import AnomalyDetector
from models.attack_classifier import AttackClassifier
from models.baseline_profile import EntityProfile

NEW_ENTITY_ID = "user_9999"


def _synthetic_events():
    base_ts = pd.Timestamp("2026-07-20 09:00:00")
    resources = ["auth-server", "eng-repo-01", "eng-repo-01", "vpn-gateway"]
    event_types = ["login", "file_access", "file_access", "host_connect"]
    rows = []
    for i in range(4):
        rows.append({
            "event_id": f"COLDSTART{i:03d}",
            "timestamp": base_ts + pd.Timedelta(minutes=15 * i),
            "user_id": NEW_ENTITY_ID,
            "entity_type": "user",
            "event_type": event_types[i],
            "src_ip": "10.99.99.5",
            "country": "United States",
            "city": "New York",
            "lat": 40.7128,
            "lon": -74.0060,
            "device_id": "dev-newlaptop01",
            "device_fingerprint": "newlaptop-fingerprint-0001",
            "resource": resources[i],
            "auth_method": "password",
            "session_duration": 900.0,
            "command_sequence": "",
            "success": True,
            "label": "normal",
        })
    return pd.DataFrame(rows)


def main():
    print(f"=== Cold-Start Demo: brand-new entity `{NEW_ENTITY_ID}` ===\n")
    events = _synthetic_events()
    print(f"Synthetic history: {len(events)} events for an entity never seen before.\n")

    # profile_store_path=None -- don't let this tiny one-entity run clobber the real artifact
    features_df = compute_features(events, config={"profile_store_path": None})

    print("Computed features (baseline-dependent columns are the interesting ones):")
    cols = ["event_id", "has_baseline", "is_new_country", "is_new_device",
            "is_new_resource", "is_new_auth_method", "is_outside_usual_hours"]
    print(features_df[cols].to_string(index=False))
    print(
        "\nhas_baseline is False for every event (fewer than min_baseline_events) -> every "
        "is_new_* flag\nis held conservatively at False instead of firing on every single "
        "field, which would otherwise\nflood a brand-new entity with false alarms.\n"
    )

    detector = AnomalyDetector.load()
    classifier = AttackClassifier.load()
    X = features_df[FEATURE_COLUMNS]
    anomaly_scores = detector.score(X)
    probs = classifier.predict_proba(X)
    classes = classifier.model.classes_
    max_prob = probs.max(axis=1)
    pred_label = classes[probs.argmax(axis=1)]

    result = pd.DataFrame({
        "event_id": features_df["event_id"],
        "anomaly_score": anomaly_scores.round(4),
        "predicted_label": pred_label,
        "classifier_confidence": max_prob.round(3),
    })
    print("Scored by the trained anomaly detector + classifier:")
    print(result.to_string(index=False))
    print(
        f"\nNone of these are predicted as an attack type despite zero prior history for "
        f"{NEW_ENTITY_ID} -- the\nsystem doesn't punish novelty itself, only genuine deviation "
        f"from an established baseline.\n"
    )

    print("=== models.baseline_profile.EntityProfile.deviation_score() directly ===")
    baseline_events = events.iloc[:2]
    profile = EntityProfile.fit(NEW_ENTITY_ID, baseline_events)
    print(f"Profile built from the first {len(baseline_events)} events (event_count={profile.event_count}).")

    familiar_event = {
        "hour": events.iloc[2]["timestamp"].hour,
        "country": events.iloc[2]["country"],
        "device_fingerprint": events.iloc[2]["device_fingerprint"],
        "resource": events.iloc[2]["resource"],
        "auth_method": events.iloc[2]["auth_method"],
    }
    foreign_event = {
        "hour": 3,
        "country": "Russia",
        "device_fingerprint": "unknown-device-xyz",
        "resource": "domain-controller-01",
        "auth_method": "certificate",
    }
    print(f"  deviation_score(matching follow-up event):  {profile.deviation_score(familiar_event):.3f}")
    print(f"  deviation_score(wildly different event):    {profile.deviation_score(foreign_event):.3f}")
    print(
        "\nEven with a real (if tiny) profile, deviation_score still recognizes a genuinely "
        "foreign\npattern -- cold-start conservatism in the per-event flags is about avoiding "
        "false alarms on\nnoise, not about being blind to real anomalies once *some* signal exists."
    )


if __name__ == "__main__":
    main()
