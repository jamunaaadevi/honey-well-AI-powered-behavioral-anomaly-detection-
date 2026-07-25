"""Trains the anomaly detector and attack classifier, evaluates the combined pipeline.

Chronological train/validation/test split: every tunable decision (IF threshold, safety-net
percentiles, combined_risk weighting, entity-day boost accept/revert) is selected using
validation data only, then frozen and applied once to the test set for final reporting.
The test set's labels never influence any decision -- only the one final scoring pass.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score

from features.feature_engineering import FEATURE_COLUMNS
from models.anomaly_detector import AnomalyDetector
from models.attack_classifier import AttackClassifier
from models.entity_day_detector import EntityDayDetector, build_entity_day_table, compute_risk_boost
from models.sequence_detector import SequenceAnomalyDetector

DEFAULT_CONFIG = {
    "features_path": "data/features.csv",
    "logs_path": "data/logs.csv",
    "predictions_path": "data/predictions.csv",
    "train_fraction": 0.6,
    "val_fraction": 0.2,
    "test_fraction": 0.2,
    "target_attack_catch_rate": 0.90,
    "classifier_prob_cutoff": 0.5,
    "if_high_confidence_percentile": 99.5,  # top 0.5% most anomalous (default reference point)
    "sequence_high_confidence_percentile": 99.5,  # top 0.5% most anomalous (default reference point)
    "operating_table_cutoffs": [0.3, 0.5, 0.7],
    "incident_gap_minutes": 180,
    # FIX 3: safety-net threshold sweep. Percentiles, not "top %" -- e.g. 99.7 = top 0.3%.
    "safety_net_sweep": {
        "if_percentiles": [99.7, 99.5],  # top 0.3%, 0.5%
        "sequence_percentiles": [99.7, 99.5, 99.3],  # top 0.3%, 0.5%, 0.7%
        "detection_tolerance": 0.01,
    },
    "alert_budget_fraction": 0.01,  # top-1% of test events by combined risk
    # combined_risk weighting candidates, tried side by side each run. classifier_signal picks
    # which classifier-derived array feeds the "classifier" weight slot: max_prob (confidence in
    # whichever class was predicted -- turned out to be biased toward confidently-normal events,
    # see the printed "combined_risk weighting experiment" section) or p_attack (1 - P(normal),
    # the classifier's actual probability mass on "this is some kind of attack").
    "combined_risk_weight_variants": {
        "equal-weight (max_prob baseline)": {
            "classifier_signal": "max_prob", "weights": {"classifier": 1 / 3, "if": 1 / 3, "sequence": 1 / 3},
        },
        "p_attack alone": {
            "classifier_signal": "p_attack", "weights": {"classifier": 1.0, "if": 0.0, "sequence": 0.0},
        },
        "p_attack-heavy (70/10/20)": {
            "classifier_signal": "p_attack", "weights": {"classifier": 0.70, "if": 0.10, "sequence": 0.20},
        },
    },
}


def load_splits(config):
    """Chronological train/validation/test split -- train is earliest, test is latest.
    A UEBA system should always be evaluated on time strictly after what it was tuned on."""
    df = pd.read_csv(config["features_path"], parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_cutoff = int(n * config["train_fraction"])
    val_cutoff = int(n * (config["train_fraction"] + config["val_fraction"]))
    train_df = df.iloc[:train_cutoff].reset_index(drop=True)
    val_df = df.iloc[train_cutoff:val_cutoff].reset_index(drop=True)
    test_df = df.iloc[val_cutoff:].reset_index(drop=True)
    return train_df, val_df, test_df


def train_anomaly_detector(train_df):
    normal_train = train_df[train_df["label"] == "normal"]
    return AnomalyDetector().fit(normal_train[FEATURE_COLUMNS])


def train_attack_classifier(train_df):
    return AttackClassifier().fit(train_df[FEATURE_COLUMNS], train_df["label"])


def compute_signals(df, detector, classifier, seq_score_lookup):
    """anomaly_scores, max_prob, y_pred, p_attack, sequence_scores, is_attack for one split.

    seq_score_lookup: sequence reconstruction-error scores already computed once over the
    full chronological history (train+val+test), indexed by event_id -- sliced down here so
    every split's windows see each entity's real preceding events instead of being zero-padded
    as if cold-start.
    """
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    is_attack = (y != "normal").to_numpy()

    anomaly_scores = detector.score(X)

    probs = classifier.predict_proba(X)
    classes = classifier.model.classes_
    max_idx = probs.argmax(axis=1)
    max_prob = probs[np.arange(len(probs)), max_idx]
    y_pred = classes[max_idx]
    p_attack = 1.0 - probs[:, list(classes).index("normal")]  # P(any attack class), unlike max_prob

    sequence_scores = seq_score_lookup.loc[df["event_id"]].to_numpy()

    return {
        "y": y, "is_attack": is_attack, "anomaly_scores": anomaly_scores,
        "max_prob": max_prob, "y_pred": y_pred, "p_attack": p_attack,
        "sequence_scores": sequence_scores,
    }


def _select_threshold(anomaly_scores, is_attack, target_recall):
    """Threshold s.t. `target_recall` fraction of attack events score at or above it."""
    attack_scores = anomaly_scores[is_attack]
    return np.percentile(attack_scores, 100 * (1 - target_recall))


def evaluate_anomaly_detector(anomaly_scores, is_attack, threshold):
    alerted = anomaly_scores >= threshold
    tp = int((alerted & is_attack).sum())
    fp = int((alerted & ~is_attack).sum())
    fn = int((~alerted & is_attack).sum())
    tn = int((~alerted & ~is_attack).sum())
    return {
        "roc_auc": roc_auc_score(is_attack, anomaly_scores),
        "threshold": threshold,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "alerted": alerted,
    }


def evaluate_classifier(y_true, y_pred):
    labels = sorted(y_true.unique())
    report = classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    return report, cm_df


def _assign_incident_ids(df, gap_minutes):
    """Cluster a user's events into incidents: a new incident starts after a gap > gap_minutes."""
    d = df.sort_values(["user_id", "timestamp"])
    gap = d.groupby("user_id")["timestamp"].diff().dt.total_seconds() / 60
    is_new_incident = gap.isna() | (gap > gap_minutes)
    incident_seq = is_new_incident.groupby(d["user_id"]).cumsum()
    return d["user_id"].astype(str) + "_" + incident_seq.astype(str)


def evaluate_combined_pipeline(test_df, alerted, y_pred, is_attack, incident_gap_minutes):
    """Per-attack-type detection.

    impossible_travel is incident-aware: by construction only the second of its two
    events (the far-away login) looks anomalous, so an incident counts as detected if
    EITHER of its two events is alerted and correctly classified.
    """
    eval_df = test_df.copy()
    eval_df["alerted"] = alerted
    eval_df["pred_label"] = y_pred
    eval_df["event_detected"] = eval_df["alerted"] & (eval_df["pred_label"] == eval_df["label"])

    attack_types = sorted(l for l in eval_df["label"].unique() if l != "normal")
    rows = []
    for attack_type in attack_types:
        subset = eval_df[eval_df["label"] == attack_type]
        if attack_type == "impossible_travel":
            incident_id = _assign_incident_ids(subset, incident_gap_minutes)
            detected_by_incident = subset["event_detected"].groupby(incident_id).any()
            unit, n, detection_rate = "incident", len(detected_by_incident), detected_by_incident.mean()
        else:
            unit, n, detection_rate = "event", len(subset), subset["event_detected"].mean()
        rows.append({"attack_type": attack_type, "unit": unit, "n": n, "detection_rate": detection_rate})
    summary = pd.DataFrame(rows)

    tp = int((alerted & is_attack).sum())
    fp = int((alerted & ~is_attack).sum())
    tn = int((~alerted & ~is_attack).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return summary, precision, fp_rate


def _confusion_rates(alerted, is_attack):
    alerted = np.asarray(alerted)
    is_attack = np.asarray(is_attack)
    tp = int((alerted & is_attack).sum())
    fp = int((alerted & ~is_attack).sum())
    fn = int((~alerted & is_attack).sum())
    tn = int((~alerted & ~is_attack).sum())
    return {
        "detection_rate": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def _minmax(values):
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _combined_risk(classifier_signal, anomaly_scores, sequence_scores, weights):
    return (
        weights["classifier"] * _minmax(classifier_signal)
        + weights["if"] * _minmax(anomaly_scores)
        + weights["sequence"] * _minmax(sequence_scores)
    )


def _budget_mask(combined_risk, budget_fraction):
    n = len(combined_risk)
    budget_n = max(1, round(budget_fraction * n))
    threshold = np.sort(combined_risk)[-budget_n]
    return combined_risk >= threshold


def evaluate_alert_budget(combined_risk, is_attack, budget_fraction):
    """Rank all events by combined_risk; alert only the top `budget_fraction`."""
    alerted = _budget_mask(combined_risk, budget_fraction)
    metrics = _confusion_rates(alerted, is_attack)
    metrics["budget_n"] = int(alerted.sum())
    metrics["threshold"] = float(np.sort(combined_risk)[-metrics["budget_n"]])
    return metrics


def _class_budget_detection(mask, true_label, target_label):
    is_target = (true_label == target_label).to_numpy()
    return float((mask & is_target).sum() / is_target.sum()) if is_target.sum() else 0.0


def build_operating_table(y_pred, max_prob, if_high_conf_alert, is_attack, cutoffs):
    rows = []
    for cutoff in cutoffs:
        alert = ((y_pred != "normal") & (max_prob > cutoff)) | if_high_conf_alert
        tp = int((alert & is_attack).sum())
        fp = int((alert & ~is_attack).sum())
        fn = int((~alert & is_attack).sum())
        tn = int((~alert & ~is_attack).sum())
        rows.append({
            "classifier_cutoff": cutoff,
            "detection_rate": tp / (tp + fn) if (tp + fn) else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        })
    return pd.DataFrame(rows)


def sweep_safety_net_thresholds(anomaly_scores, sequence_scores, classifier_alert, is_attack, if_percentiles, seq_percentiles):
    """Grid over (IF percentile, sequence percentile); classifier_alert stays fixed."""
    rows = []
    thresholds = {}
    for if_pct in if_percentiles:
        if_thr = np.percentile(anomaly_scores, if_pct)
        if_alert = anomaly_scores >= if_thr
        for seq_pct in seq_percentiles:
            seq_thr = np.percentile(sequence_scores, seq_pct)
            seq_alert = sequence_scores >= seq_thr
            rates = _confusion_rates(classifier_alert | if_alert | seq_alert, is_attack)
            rows.append({
                "if_top_pct": round(100 - if_pct, 2), "seq_top_pct": round(100 - seq_pct, 2),
                "detection_rate": rates["detection_rate"], "precision": rates["precision"],
                "false_positive_rate": rates["false_positive_rate"],
            })
            thresholds[(if_pct, seq_pct)] = (if_thr, seq_thr)
    return pd.DataFrame(rows), thresholds


def main():
    config = DEFAULT_CONFIG
    train_df, val_df, test_df = load_splits(config)

    print(f"Train events: {len(train_df)}  ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Val events:   {len(val_df)}  ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"Test events:  {len(test_df)}  ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})\n")

    detector = train_anomaly_detector(train_df)
    detector.save()
    classifier = train_attack_classifier(train_df)
    classifier.save()

    print("=== Sequence Detector (LSTM autoencoder, normal-only training) ===")
    seq_detector = SequenceAnomalyDetector().fit(train_df)
    seq_detector.save()
    # score the full train+val+test history once so every split's windows see each entity's
    # real preceding events instead of being zero-padded as if cold-start
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    seq_score_lookup = seq_detector.score(full_df)
    print("  trained on train split only; windows scored over the full train+val+test history\n")

    # ================================================================================
    # VALIDATION-BASED TUNING -- every threshold, percentile, and weighting choice below
    # is selected using val_df only. test_df's labels influence nothing in this section.
    # ================================================================================
    print("################################################################")
    print("# VALIDATION-BASED TUNING (test set not used for any decision below)")
    print("################################################################\n")

    val_sig = compute_signals(val_df, detector, classifier, seq_score_lookup)

    threshold = _select_threshold(val_sig["anomaly_scores"], val_sig["is_attack"], config["target_attack_catch_rate"])
    anomaly_metrics_val = evaluate_anomaly_detector(val_sig["anomaly_scores"], val_sig["is_attack"], threshold)
    print("=== Anomaly Detector (Isolation Forest) -- threshold selected on validation ===")
    print(f"ROC-AUC (validation, normal vs any attack): {anomaly_metrics_val['roc_auc']:.4f}")
    print(f"Threshold @ {config['target_attack_catch_rate']:.0%} attack catch rate (frozen from validation): {threshold:.4f}")
    print(f"  precision (validation): {anomaly_metrics_val['precision']:.4f}")
    print(f"  recall (validation):    {anomaly_metrics_val['recall']:.4f}")
    print(f"  FPR (validation):       {anomaly_metrics_val['false_positive_rate']:.4f}\n")

    classifier_alert_val = (val_sig["y_pred"] != "normal") & (val_sig["max_prob"] > config["classifier_prob_cutoff"])

    print("=== FIX 3: Safety-Net Threshold Sweep (selected on validation) ===")
    sweep_cfg = config["safety_net_sweep"]
    sweep_df, sweep_thresholds = sweep_safety_net_thresholds(
        val_sig["anomaly_scores"], val_sig["sequence_scores"], classifier_alert_val, val_sig["is_attack"],
        sweep_cfg["if_percentiles"], sweep_cfg["sequence_percentiles"],
    )
    print(sweep_df.round(4).to_string(index=False))

    if_alert_default_val = val_sig["anomaly_scores"] >= np.percentile(val_sig["anomaly_scores"], config["if_high_confidence_percentile"])
    seq_alert_default_val = val_sig["sequence_scores"] >= np.percentile(val_sig["sequence_scores"], config["sequence_high_confidence_percentile"])
    hybrid_alerted_default_val = classifier_alert_val | if_alert_default_val | seq_alert_default_val
    current_detection = _confusion_rates(hybrid_alerted_default_val, val_sig["is_attack"])["detection_rate"]

    eligible = sweep_df[sweep_df["detection_rate"] >= current_detection - sweep_cfg["detection_tolerance"]]
    if eligible.empty:
        eligible = sweep_df
    best_row = eligible.loc[eligible["precision"].idxmax()]
    chosen_if_pct = 100 - best_row["if_top_pct"]
    chosen_seq_pct = 100 - best_row["seq_top_pct"]
    print(
        f"\nCurrent default (validation): IF top 0.5% / sequence top 0.5%, detection={current_detection:.4f}\n"
        f"Selected (validation): IF top {best_row['if_top_pct']:.1f}% / sequence top {best_row['seq_top_pct']:.1f}% -- "
        f"best precision ({best_row['precision']:.4f}) among combos with detection >= "
        f"{current_detection - sweep_cfg['detection_tolerance']:.4f} (current - {sweep_cfg['detection_tolerance']:.0%})\n"
    )
    if_high_conf_threshold, seq_high_conf_threshold = sweep_thresholds[(chosen_if_pct, chosen_seq_pct)]
    if (chosen_if_pct, chosen_seq_pct) != (99.5, 99.5):
        print(f"FIX 3 (frozen from validation): switched safety nets to IF top {100 - chosen_if_pct:.1f}% / "
              f"sequence top {100 - chosen_seq_pct:.1f}% -- will be applied to test.\n")
    else:
        print("FIX 3 (frozen from validation): the 0.5%/0.5% default was already the best precision "
              "point within tolerance -- no change, will be applied to test.\n")

    print("=== combined_risk weighting experiment (selected on validation) ===")
    mean_max_prob_normal = float(val_sig["max_prob"][~val_sig["is_attack"]].mean())
    mean_max_prob_attack = float(val_sig["max_prob"][val_sig["is_attack"]].mean())
    mean_p_attack_normal = float(val_sig["p_attack"][~val_sig["is_attack"]].mean())
    mean_p_attack_attack = float(val_sig["p_attack"][val_sig["is_attack"]].mean())
    print("Classifier signal calibration check (mean value on normal vs attack events, validation):")
    print(
        f"  max_prob:  normal={mean_max_prob_normal:.4f}  attack={mean_max_prob_attack:.4f}  "
        f"({'BIASED toward normal' if mean_max_prob_normal > mean_max_prob_attack else 'separates correctly'})"
    )
    print(
        f"  p_attack:  normal={mean_p_attack_normal:.4f}  attack={mean_p_attack_attack:.4f}  "
        f"({'separates correctly' if mean_p_attack_attack > mean_p_attack_normal else 'BIASED toward normal'})\n"
    )

    classifier_signals_val = {"max_prob": val_sig["max_prob"], "p_attack": val_sig["p_attack"]}
    variant_rows = []
    for name, spec in config["combined_risk_weight_variants"].items():
        risk = _combined_risk(classifier_signals_val[spec["classifier_signal"]], val_sig["anomaly_scores"], val_sig["sequence_scores"], spec["weights"])
        budget = evaluate_alert_budget(risk, val_sig["is_attack"], config["alert_budget_fraction"])
        variant_rows.append({
            "weighting": name,
            "pr_auc": average_precision_score(val_sig["is_attack"], risk),
            "budget_detection_rate": budget["detection_rate"],
            "budget_precision": budget["precision"],
            "budget_fpr": budget["false_positive_rate"],
        })
    variant_df = pd.DataFrame(variant_rows)
    print(variant_df.round(4).to_string(index=False))

    ranked = variant_df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best_name, best_pr_auc = ranked.loc[0, "weighting"], ranked.loc[0, "pr_auc"]
    baseline_pr_auc = variant_df.loc[variant_df["weighting"] == "equal-weight (max_prob baseline)", "pr_auc"].iloc[0]
    ranking_str = " > ".join(f"{r.weighting} ({r.pr_auc:.4f})" for r in ranked.itertuples())
    print(
        f"\nFinding (validation): '{best_name}' wins on PR-AUC. Full ranking: {ranking_str}\n"
        f"p_attack is correctly calibrated where max_prob wasn't (see the check above), which is why "
        f"{'the p_attack variants beat the old max_prob baseline' if best_pr_auc > baseline_pr_auc else 'it still does not overtake the max_prob baseline overall'} "
        f"({best_pr_auc:.4f} vs {baseline_pr_auc:.4f}). Frozen weighting '{best_name}' -- will be applied to test.\n"
    )
    best_weights_spec = config["combined_risk_weight_variants"][best_name]

    print("=== FIX 1: Entity-Day Detector (low_and_slow_exfiltration) -- accept/revert decided on validation ===")
    logs_df = pd.read_csv(config["logs_path"], usecols=["event_id", "resource"])
    entity_day_detector = EntityDayDetector()
    train_entity_day = build_entity_day_table(train_df, logs_df, entity_day_detector.config["positive_label"])
    val_entity_day = build_entity_day_table(val_df, logs_df, entity_day_detector.config["positive_label"])
    print(f"  suspicious entity-days: {int(train_entity_day['is_suspicious_day'].sum())} of "
          f"{len(train_entity_day)} in train, {int(val_entity_day['is_suspicious_day'].sum())} of "
          f"{len(val_entity_day)} in validation")

    entity_day_detector.fit(train_entity_day)
    entity_day_detector.save()
    suspicion_scores_val = entity_day_detector.predict_suspicion(val_entity_day)
    n_flagged_val = int((suspicion_scores_val > entity_day_detector.config["suspicious_threshold"]).sum())
    print(f"  flagged {n_flagged_val} validation entity-days as suspicious "
          f"(threshold {entity_day_detector.config['suspicious_threshold']})")

    classifier_signal_val = classifier_signals_val[best_weights_spec["classifier_signal"]]
    combined_risk_val = _combined_risk(classifier_signal_val, val_sig["anomaly_scores"], val_sig["sequence_scores"], best_weights_spec["weights"])
    risk_boost_val = compute_risk_boost(val_df, val_entity_day, suspicion_scores_val, entity_day_detector.config)
    combined_risk_val_boosted = np.clip(combined_risk_val + risk_boost_val, 0.0, 1.0)
    print(f"  boosted {int((risk_boost_val > 0).sum())} validation events' combined_risk by "
          f"+{entity_day_detector.config['risk_boost']}\n")

    mask_before = _budget_mask(combined_risk_val, config["alert_budget_fraction"])
    mask_after = _budget_mask(combined_risk_val_boosted, config["alert_budget_fraction"])
    low_slow_before = _class_budget_detection(mask_before, val_sig["y"], "low_and_slow_exfiltration")
    low_slow_after = _class_budget_detection(mask_after, val_sig["y"], "low_and_slow_exfiltration")
    pr_auc_before_fix1 = average_precision_score(val_sig["is_attack"], combined_risk_val)
    pr_auc_after_fix1 = average_precision_score(val_sig["is_attack"], combined_risk_val_boosted)
    budget_before_fix1 = evaluate_alert_budget(combined_risk_val, val_sig["is_attack"], config["alert_budget_fraction"])
    budget_after_fix1 = evaluate_alert_budget(combined_risk_val_boosted, val_sig["is_attack"], config["alert_budget_fraction"])

    fix1_comparison = pd.DataFrame([
        {"metric": "low_and_slow budget detection", "before": low_slow_before, "after": low_slow_after},
        {"metric": "overall PR-AUC", "before": pr_auc_before_fix1, "after": pr_auc_after_fix1},
        {"metric": "overall budget detection_rate", "before": budget_before_fix1["detection_rate"], "after": budget_after_fix1["detection_rate"]},
        {"metric": "overall budget precision", "before": budget_before_fix1["precision"], "after": budget_after_fix1["precision"]},
        {"metric": "overall budget FPR", "before": budget_before_fix1["false_positive_rate"], "after": budget_after_fix1["false_positive_rate"]},
    ])
    print(fix1_comparison.round(4).to_string(index=False))

    fix1_regression = (
        pr_auc_after_fix1 < pr_auc_before_fix1 - 1e-9
        or budget_after_fix1["detection_rate"] < budget_before_fix1["detection_rate"] - 1e-9
        or budget_after_fix1["precision"] < budget_before_fix1["precision"] - 1e-9
    )
    use_entity_day_boost = not fix1_regression
    if fix1_regression:
        print("\nFIX 1 REVERTED (decided on validation): regressed PR-AUC, budget detection, or budget "
              "precision -- entity-day boost will NOT be applied to test.\n")
    else:
        print(f"\nFIX 1 KEPT (decided on validation): low_and_slow budget detection {low_slow_before:.4f} -> "
              f"{low_slow_after:.4f}, no regression -- boost WILL be applied to test.\n")

    # ================================================================================
    # FINAL TEST-SET EVALUATION -- every choice above is now frozen. This is the only
    # place test_df's labels are used, and only for reporting, never for selection.
    # Numbers below may differ from earlier tuning runs against a single test split --
    # that's expected: this is now a genuinely held-out evaluation.
    # ================================================================================
    print("################################################################")
    print("# FINAL TEST-SET EVALUATION (frozen thresholds/weights from validation)")
    print("################################################################\n")

    test_sig = compute_signals(test_df, detector, classifier, seq_score_lookup)
    y_test, is_attack = test_sig["y"], test_sig["is_attack"]
    anomaly_scores, sequence_scores = test_sig["anomaly_scores"], test_sig["sequence_scores"]
    max_prob, y_pred, p_attack = test_sig["max_prob"], test_sig["y_pred"], test_sig["p_attack"]

    report, cm_df = evaluate_classifier(y_test, y_pred)
    print("=== Attack Classifier (Random Forest, class_weight=balanced) -- test set ===")
    print(report)
    print("Confusion matrix (test):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(cm_df)
    print()

    anomaly_metrics_test = evaluate_anomaly_detector(anomaly_scores, is_attack, threshold)
    print("=== Anomaly Detector (Isolation Forest) -- frozen validation threshold applied to test ===")
    print(f"ROC-AUC (test, normal vs any attack): {anomaly_metrics_test['roc_auc']:.4f}")
    print(f"Threshold (frozen from validation @ {config['target_attack_catch_rate']:.0%} catch rate): {threshold:.4f}")
    print(f"  precision:            {anomaly_metrics_test['precision']:.4f}")
    print(f"  recall:               {anomaly_metrics_test['recall']:.4f}")
    print(f"  false positive rate:  {anomaly_metrics_test['false_positive_rate']:.4f}\n")

    seq_roc_auc_test = roc_auc_score(is_attack, sequence_scores)
    print(f"Sequence Detector ROC-AUC (test, reconstruction error vs any attack): {seq_roc_auc_test:.4f}\n")

    if_high_conf_alert = anomaly_scores >= if_high_conf_threshold  # frozen from validation (FIX 3)
    seq_high_conf_alert = sequence_scores >= seq_high_conf_threshold  # frozen from validation (FIX 3)
    classifier_alert = (y_pred != "normal") & (max_prob > config["classifier_prob_cutoff"])
    hybrid_alerted = classifier_alert | if_high_conf_alert | seq_high_conf_alert

    print("=== Hybrid Alert Rule (3 signals, frozen thresholds applied to test) ===")
    print(f"Classifier probability cutoff: {config['classifier_prob_cutoff']}")
    print(f"IF safety-net threshold (frozen from validation): {if_high_conf_threshold:.4f}")
    print(f"Sequence safety-net threshold (frozen from validation): {seq_high_conf_threshold:.4f}")
    print(f"  alerted via classifier: {int(classifier_alert.sum())}")
    print(f"  alerted via IF safety net: {int(if_high_conf_alert.sum())}")
    print(f"  alerted via sequence safety net: {int(seq_high_conf_alert.sum())}")
    print(f"  alerted by sequence signal ONLY (missed by classifier & IF): {int((seq_high_conf_alert & ~classifier_alert & ~if_high_conf_alert).sum())}")
    print(f"  total alerted (3-signal): {int(hybrid_alerted.sum())}\n")

    combined_summary, combined_precision, combined_fpr = evaluate_combined_pipeline(
        test_df, hybrid_alerted, y_pred, is_attack, config["incident_gap_minutes"]
    )
    print("=== Combined Pipeline (test set, frozen hybrid alert rule) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(combined_summary.to_string(index=False))
    print(f"Overall precision:           {combined_precision:.4f}")
    print(f"Overall false positive rate: {combined_fpr:.4f}\n")

    op_table = build_operating_table(y_pred, max_prob, if_high_conf_alert, is_attack, config["operating_table_cutoffs"])
    print("=== Operating Table (test set; classifier cutoff varies, IF safety net fixed) ===")
    print(op_table.to_string(index=False))
    print()

    classifier_signals_test = {"max_prob": max_prob, "p_attack": p_attack}
    classifier_signal_test = classifier_signals_test[best_weights_spec["classifier_signal"]]
    combined_risk = _combined_risk(classifier_signal_test, anomaly_scores, sequence_scores, best_weights_spec["weights"])
    print(f"combined_risk (test) uses the validation-selected weighting: '{best_name}'\n")

    if use_entity_day_boost:
        test_entity_day = build_entity_day_table(test_df, logs_df, entity_day_detector.config["positive_label"])
        suspicion_scores_test = entity_day_detector.predict_suspicion(test_entity_day)
        risk_boost_test = compute_risk_boost(test_df, test_entity_day, suspicion_scores_test, entity_day_detector.config)
        combined_risk = np.clip(combined_risk + risk_boost_test, 0.0, 1.0)
        print(f"FIX 1 (frozen KEPT from validation): boosted {int((risk_boost_test > 0).sum())} test events' "
              f"combined_risk by +{entity_day_detector.config['risk_boost']}\n")
    else:
        print("FIX 1 (frozen REVERTED from validation): entity-day boost not applied to test.\n")

    budget_metrics = evaluate_alert_budget(combined_risk, is_attack, config["alert_budget_fraction"])
    n_attack_events = int(is_attack.sum())
    theoretical_max_detection = min(budget_metrics["budget_n"], n_attack_events) / n_attack_events
    print(f"=== Top-{config['alert_budget_fraction']:.0%} Alert Budget (test set, ranked by combined_risk: {best_name}) ===")
    print(f"Budget: {budget_metrics['budget_n']} events (of {len(test_df)} test events, {n_attack_events} are attacks)")
    print(f"  detection_rate:       {budget_metrics['detection_rate']:.4f}")
    print(f"  theoretical max:      {theoretical_max_detection:.4f}  (if every budget slot were a true positive: "
          f"min(budget_n, n_attack_events)/n_attack_events)")
    print(f"  ceiling reached:      {budget_metrics['detection_rate'] / theoretical_max_detection:.1%}")
    print(f"  precision:            {budget_metrics['precision']:.4f}")
    print(f"  false_positive_rate:  {budget_metrics['false_positive_rate']:.4f}\n")

    pr_auc_if_only = average_precision_score(is_attack, anomaly_scores)
    pr_auc_combined = average_precision_score(is_attack, combined_risk)
    if_alert_default_test = anomaly_scores >= np.percentile(anomaly_scores, config["if_high_confidence_percentile"])
    hybrid_alerted_2signal = classifier_alert | if_alert_default_test  # fixed historical reference rule, pre-sequence-detector
    old_rates = _confusion_rates(hybrid_alerted_2signal, is_attack)
    new_rates = _confusion_rates(hybrid_alerted, is_attack)
    comparison = pd.DataFrame([
        {"metric": "detection_rate", "old (classifier+IF)": old_rates["detection_rate"], "new (+sequence, 3-signal)": new_rates["detection_rate"]},
        {"metric": "precision", "old (classifier+IF)": old_rates["precision"], "new (+sequence, 3-signal)": new_rates["precision"]},
        {"metric": "false_positive_rate", "old (classifier+IF)": old_rates["false_positive_rate"], "new (+sequence, 3-signal)": new_rates["false_positive_rate"]},
        {"metric": "PR-AUC", "old (classifier+IF)": pr_auc_if_only, "new (+sequence, 3-signal)": pr_auc_combined},
    ])
    print("=== Metrics: Old (2-signal hybrid) vs New (3-signal hybrid + sequence) -- test set ===")
    print(comparison.round(4).to_string(index=False))
    print()

    # Events alerted only by a safety net (classifier itself says "normal") are
    # novel/unmodeled anomalies -- surface them distinctly rather than mislabeling "normal".
    predicted_label = np.where(
        ~hybrid_alerted, "normal", np.where(y_pred == "normal", "unknown_anomaly", y_pred)
    )
    predictions_df = pd.DataFrame({
        "event_id": test_df["event_id"],
        "anomaly_score": anomaly_scores,
        "sequence_score": sequence_scores,
        "combined_risk": combined_risk,
        "alerted": hybrid_alerted,
        "predicted_label": predicted_label,
        "true_label": y_test.to_numpy(),
    })
    out_path = Path(config["predictions_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(out_path, index=False)
    print(f"Wrote {len(predictions_df)} predictions to {out_path} (test set, frozen thresholds/weights only)")


if __name__ == "__main__":
    main()
