"""Composite 0-100 risk score per alerted event, with SHAP-backed plain-English reasons."""

from pathlib import Path

import numpy as np
import pandas as pd

from explain.shap_explainer import AlertExplainer

DEFAULT_CONFIG = {
    "predictions_path": "data/predictions.csv",
    "features_path": "data/features.csv",
    "alerts_path": "data/alerts.csv",
    "top_reasons": 3,
}

RISK_TIERS = [(0, 40, "low"), (40, 70, "medium"), (70, 90, "high"), (90, 101, "critical")]


def risk_tier(scores):
    conditions = [scores < 40, scores < 70, scores < 90]
    choices = ["low", "medium", "high"]
    return np.select(conditions, choices, default="critical")


def _normalize(values, population_min, population_max):
    span = population_max - population_min
    if span <= 0:
        return np.zeros(len(values))
    return ((values - population_min) / span).clip(0, 1)


def compute_risk_scores(config=None):
    """risk_score is 100x the pipeline's own validated combined_risk (classifier p_attack +
    Isolation Forest + LSTM sequence detector, weighted and selected on validation by
    models/train.py) -- the same number that drives the hybrid alert rule and the top-1%
    budget, not a separately recomputed 2-signal blend. That keeps the analyst-facing risk
    score consistent with what actually triggered the alert.

    Checked empirically: for unknown_anomaly events (classifier itself predicted "normal", so
    combined_risk's classifier term -- 70% of the weight -- is near-zero), this dragged every
    such event into the "low" tier even when the IF or sequence safety net fired hard, which is
    backwards for exactly the case that safety net exists to catch. Those rows are overridden
    with a pure IF+sequence blend (no classifier term) instead.
    """
    config = {**DEFAULT_CONFIG, **(config or {})}

    predictions = pd.read_csv(config["predictions_path"])
    alerted = predictions[predictions["alerted"]].reset_index(drop=True)
    features = pd.read_csv(config["features_path"], parse_dates=["timestamp"])
    merged = alerted.merge(features, on="event_id", how="left", suffixes=("", "_feat"))

    risk_score = 100 * merged["combined_risk"].to_numpy()

    is_unknown = (merged["predicted_label"] == "unknown_anomaly").to_numpy()
    if is_unknown.any():
        normalized_anomaly = _normalize(
            merged["anomaly_score"].to_numpy(), predictions["anomaly_score"].min(), predictions["anomaly_score"].max()
        )
        normalized_sequence = _normalize(
            merged["sequence_score"].to_numpy(), predictions["sequence_score"].min(), predictions["sequence_score"].max()
        )
        unknown_score = 100 * (normalized_anomaly + normalized_sequence) / 2
        risk_score = np.where(is_unknown, unknown_score, risk_score)

    result = merged[["event_id", "user_id", "timestamp", "predicted_label", "true_label"]].copy()
    result["combined_risk"] = merged["combined_risk"].to_numpy()
    result["risk_score"] = risk_score.round(2)
    result["risk_tier"] = risk_tier(risk_score)
    return result


def attach_explanations(alerts_df, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    explainer = AlertExplainer(config).fit()

    explanations = []
    for event_id in alerts_df["event_id"]:
        reasons = explainer.explain_event(event_id, top_n=config["top_reasons"])
        explanations.append("; ".join(r["sentence"] for r in reasons))
    alerts_df = alerts_df.copy()
    alerts_df["explanation"] = explanations
    return alerts_df


def main():
    config = DEFAULT_CONFIG
    alerts_df = compute_risk_scores(config)
    alerts_df = attach_explanations(alerts_df, config)

    out_cols = ["event_id", "user_id", "timestamp", "predicted_label", "true_label", "risk_score", "risk_tier", "explanation"]
    out_df = alerts_df[out_cols].sort_values("risk_score", ascending=False).reset_index(drop=True)

    out_path = Path(config["alerts_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} alerts to {out_path}\n")

    print("Alerts per risk tier:")
    print(out_df["risk_tier"].value_counts().reindex(["critical", "high", "medium", "low"]).fillna(0).astype(int))
    print()

    for attack_type in ["brute_force", "impossible_travel", "credential_misuse"]:
        example = out_df[out_df["predicted_label"] == attack_type].head(1)
        if example.empty:
            print(f"No alerted example found for {attack_type}\n")
            continue
        row = example.iloc[0]
        print(f"--- Example: {attack_type} ---")
        print(f"event_id:      {row['event_id']}")
        print(f"user_id:       {row['user_id']}")
        print(f"timestamp:     {row['timestamp']}")
        print(f"true_label:    {row['true_label']}")
        print(f"risk_score:    {row['risk_score']}  ({row['risk_tier']})")
        print(f"explanation:   {row['explanation']}\n")


if __name__ == "__main__":
    main()
