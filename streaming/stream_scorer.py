"""Real-time streaming scorer: scores test-period events from data/logs.csv.

Proves real-time feasibility rather than claiming it. The batch features/feature_engineering.py
pipeline computes all 28 features via groupby+rolling over the *entire* dataset at once -- that
is fundamentally a batch operation. This module instead maintains small, bounded per-entity and
per-src_ip windows (deques, pruned as they age out) that are updated incrementally as each event
arrives, and computes the exact same 28 features from that live state, with no full-matrix
recompute anywhere in the per-event path.

Two scoring modes, both available:
- single-event: score each event through IF/classifier/LSTM the instant it arrives.
- micro-batch (FIX 5): buffer up to `micro_batch_size` events (or `micro_batch_max_wait_ms`,
  whichever comes first), then make ONE model call per detector for the whole batch. Feature
  computation and state updates stay strictly per-event and sequential (correctness depends on
  it); only the three model calls are batched, amortizing their fixed per-call overhead.

Alert thresholds and risk-score normalization bounds are calibrated ONCE at startup from the
last batch run's data/predictions.csv (mirroring how a real system periodically recalibrates
offline, then applies fixed thresholds live) -- that calibration step is explicitly excluded
from the per-event latency measurement below, as is the training-period warm-up replay.
"""

import copy
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch

from features.feature_engineering import DEFAULT_CONFIG as FEATURES_CONFIG
from features.feature_engineering import FEATURE_COLUMNS
from models.anomaly_detector import AnomalyDetector
from models.attack_classifier import AttackClassifier
from models.baseline_profile import EntityProfile, EntityProfileStore
from models.sequence_detector import SequenceAnomalyDetector

DEFAULT_CONFIG = {
    "logs_path": "data/logs.csv",
    "features_path": FEATURES_CONFIG["output_path"],  # only its timestamp/event_id columns, to find the split boundary
    "predictions_path": "data/predictions.csv",  # last batch run's scores, for one-time threshold calibration
    "profile_store_path": FEATURES_CONFIG["profile_store_path"],
    "train_fraction": 0.7,  # must match models/train.py's split
    "min_baseline_events": FEATURES_CONFIG["min_baseline_events"],
    "sensitive_resources": FEATURES_CONFIG["sensitive_resources"],
    "geo_velocity_cap_kmh": FEATURES_CONFIG["geo_velocity_cap_kmh"],
    "rolling_window_minutes": 30,
    "cumulative_window_days": 7,
    "classifier_prob_cutoff": 0.5,
    "safety_net_percentile": 99.5,
    "combined_risk_weights": {"classifier": 0.70, "if": 0.10, "sequence": 0.20},
    "n_events_to_stream": 5000,
    "micro_batch_size": 32,
    "micro_batch_max_wait_ms": 50,
}

LOG_COLUMNS = [
    "event_id", "timestamp", "user_id", "entity_type", "event_type", "src_ip",
    "country", "lat", "lon", "device_fingerprint", "resource", "auth_method",
    "session_duration", "command_sequence", "success", "label",
]


class EntityState:
    """Per-entity incremental state: last event pointer + two bounded rolling windows."""

    __slots__ = ("last_lat", "last_lon", "last_ts", "win30", "win7d", "seq_window")

    def __init__(self):
        self.last_lat = None
        self.last_lon = None
        self.last_ts = None
        self.win30 = deque()  # (ts, resource, login_fail)
        self.win7d = deque()  # (ts, is_new_resource, is_outside_usual_hours, is_sensitive_resource)
        self.seq_window = deque(maxlen=10)  # last-10 raw feature vectors, oldest first


class IPState:
    """Per-src_ip incremental state: one bounded rolling window across all entities sharing it."""

    __slots__ = ("win30",)

    def __init__(self):
        self.win30 = deque()  # (ts, entity_id, login_fail)


def _prune(window, current_ts, max_age):
    while window and (current_ts - window[0][0]) > max_age:
        window.popleft()


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _update_and_compute_features(row, profile, entity_state, ip_state, config, seq_feature_names):
    """Mutates entity_state/ip_state (append current event, prune stale ones) and returns
    this event's 28-feature dict -- the incremental equivalent of one row of features.csv."""
    ts = row.timestamp
    has_baseline = profile.event_count >= config["min_baseline_events"]

    hour = ts.hour
    is_weekend = ts.weekday() >= 5
    is_outside_usual_hours = has_baseline and profile.hour_histogram[hour] == 0
    is_new_country = has_baseline and row.country not in profile.countries
    is_new_device = has_baseline and row.device_fingerprint not in profile.devices
    is_new_resource = has_baseline and row.resource not in profile.resources
    is_new_auth_method = has_baseline and row.auth_method not in profile.auth_methods
    is_sensitive_resource = row.resource in config["sensitive_resources"]
    login_fail = (row.event_type == "login") and (not row.success)

    if entity_state.last_ts is None:
        geo_velocity_kmh = 0.0
        minutes_since_last_event = -1.0
    else:
        minutes_gap = (ts - entity_state.last_ts).total_seconds() / 60
        hours_gap = max(minutes_gap / 60, 1 / 3600)
        distance_km = _haversine_km(row.lat, row.lon, entity_state.last_lat, entity_state.last_lon)
        geo_velocity_kmh = min(distance_km / hours_gap, config["geo_velocity_cap_kmh"])
        minutes_since_last_event = minutes_gap

    std = profile.session_duration_std
    if std and std > 0:
        z = (row.session_duration - profile.session_duration_mean) / std
        session_duration_zscore = float(z) if np.isfinite(z) else 0.0
    else:
        session_duration_zscore = 0.0

    command_sequence = row.command_sequence if isinstance(row.command_sequence, str) else ""
    if command_sequence:
        commands = command_sequence.split(";")
        command_sequence_length = len(commands)
        is_rare_command = has_baseline and any(c not in profile.known_commands for c in commands)
    else:
        command_sequence_length = 0
        is_rare_command = False

    max_age_30min = pd.Timedelta(minutes=config["rolling_window_minutes"])
    max_age_7d = pd.Timedelta(days=config["cumulative_window_days"])

    _prune(entity_state.win30, ts, max_age_30min)
    entity_state.win30.append((ts, row.resource, login_fail))
    _prune(ip_state.win30, ts, max_age_30min)
    ip_state.win30.append((ts, row.user_id, login_fail))
    _prune(entity_state.win7d, ts, max_age_7d)
    entity_state.win7d.append((ts, is_new_resource, is_outside_usual_hours, is_sensitive_resource))

    total_30 = len(entity_state.win30)
    failed_30 = sum(1 for _, _, f in entity_state.win30 if f)
    distinct_resources_30 = len({r for _, r, _ in entity_state.win30})

    ip_total = len(ip_state.win30)
    ip_failed = sum(1 for _, _, f in ip_state.win30 if f)
    ip_distinct_entities = len({e for _, e, _ in ip_state.win30})
    ip_fail_rate = ip_failed / ip_total if ip_total else 0.0

    cum_new_resource_7d = sum(1 for _, nr, _, _ in entity_state.win7d if nr)
    off_hours_7d = sum(1 for _, _, oh, _ in entity_state.win7d if oh)
    sensitive_7d = sum(1 for _, _, _, sr in entity_state.win7d if sr)

    features = {
        "has_baseline": float(has_baseline),
        "hour_of_day": float(hour),
        "is_weekend": float(is_weekend),
        "is_outside_usual_hours": float(is_outside_usual_hours),
        "is_new_country": float(is_new_country),
        "is_new_device": float(is_new_device),
        "is_new_resource": float(is_new_resource),
        "is_new_auth_method": float(is_new_auth_method),
        "geo_velocity_kmh": float(geo_velocity_kmh),
        "minutes_since_last_event": float(minutes_since_last_event),
        "failed_login_count_last_30min": float(failed_30),
        "total_event_count_last_30min": float(total_30),
        "distinct_resources_last_30min": float(distinct_resources_30),
        "is_sensitive_resource": float(is_sensitive_resource),
        "login_fail": float(login_fail),
        "user_typical_daily_event_count": float(profile.daily_event_rate),
        "session_duration_zscore": float(session_duration_zscore),
        "command_sequence_length": float(command_sequence_length),
        "is_rare_command": float(is_rare_command),
        "is_user": float(row.entity_type == "user"),
        "is_service_account": float(row.entity_type == "service_account"),
        "is_edge_device": float(row.entity_type == "edge_device"),
        "distinct_entities_from_this_ip_30min": float(ip_distinct_entities),
        "failed_logins_from_this_ip_30min": float(ip_failed),
        "fail_rate_from_this_ip_30min": float(ip_fail_rate),
        "cumulative_new_resource_count_7d": float(cum_new_resource_7d),
        "off_hours_access_count_7d": float(off_hours_7d),
        "sensitive_access_count_7d": float(sensitive_7d),
    }

    entity_state.last_lat, entity_state.last_lon, entity_state.last_ts = row.lat, row.lon, ts
    entity_state.seq_window.append(np.array([features[c] for c in seq_feature_names], dtype=np.float32))

    return features


def _build_window(seq_detector, entity_state):
    window_size = seq_detector.config["window_size"]
    n_features = len(seq_detector.feature_names_)
    vectors = list(entity_state.seq_window)
    if len(vectors) < window_size:
        vectors = [np.zeros(n_features, dtype=np.float32)] * (window_size - len(vectors)) + vectors
    return np.stack(vectors, axis=0)


def _score_sequence(seq_detector, entity_state):
    window = _build_window(seq_detector, entity_state)
    scaled = ((window - seq_detector.scaler_mean) / seq_detector.scaler_scale).astype(np.float32)
    batch = torch.from_numpy(scaled[np.newaxis, ...])
    with torch.no_grad():
        recon = seq_detector.model(batch)
        mse = ((recon - batch) ** 2).mean().item()
    return mse


def _score_sequence_batch(seq_detector, windows):
    stacked = np.stack(windows, axis=0)
    scaled = ((stacked - seq_detector.scaler_mean) / seq_detector.scaler_scale).astype(np.float32)
    batch = torch.from_numpy(scaled)
    with torch.no_grad():
        recon = seq_detector.model(batch)
        mse = ((recon - batch) ** 2).mean(dim=(1, 2)).numpy()
    return mse


def _norm(value, lo, hi):
    if hi - lo < 1e-12:
        return 0.0
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def _decide(pred_label, max_prob, p_attack, anomaly_score, sequence_score, calib, config):
    """Hybrid alert rule + p_attack-heavy combined risk for one already-scored event."""
    w = config["combined_risk_weights"]
    classifier_alert = (pred_label != "normal") and (max_prob > config["classifier_prob_cutoff"])
    if_alert = anomaly_score >= calib["if_threshold"]
    seq_alert = sequence_score >= calib["seq_threshold"]
    alerted = classifier_alert or if_alert or seq_alert

    combined_risk = 100 * (
        w["classifier"] * min(max(p_attack, 0.0), 1.0)
        + w["if"] * _norm(anomaly_score, calib["if_min"], calib["if_max"])
        + w["sequence"] * _norm(sequence_score, calib["seq_min"], calib["seq_max"])
    )
    predicted_label = "normal"
    if alerted:
        predicted_label = "unknown_anomaly" if pred_label == "normal" else pred_label
    return alerted, predicted_label, combined_risk


def _calibrate(config):
    """One-time, offline: derive fixed alert thresholds + normalization bounds from the last
    batch run. A live system recalibrates this periodically -- never per event."""
    predictions = pd.read_csv(config["predictions_path"])
    anomaly = predictions["anomaly_score"].to_numpy()
    sequence = predictions["sequence_score"].to_numpy()
    return {
        "if_threshold": float(np.percentile(anomaly, config["safety_net_percentile"])),
        "seq_threshold": float(np.percentile(sequence, config["safety_net_percentile"])),
        "if_min": float(anomaly.min()), "if_max": float(anomaly.max()),
        "seq_min": float(sequence.min()), "seq_max": float(sequence.max()),
    }


def _load_split_boundary(config):
    """The exact event_id set models/train.py treated as the test split, derived from
    features.csv so this module's split is guaranteed identical to the trained pipeline's."""
    features = pd.read_csv(config["features_path"], usecols=["event_id", "timestamp"], parse_dates=["timestamp"])
    features = features.sort_values("timestamp").reset_index(drop=True)
    cutoff = int(len(features) * config["train_fraction"])
    boundary_ts = features.loc[cutoff, "timestamp"]
    test_event_ids = set(features.loc[cutoff:, "event_id"])
    return boundary_ts, test_event_ids


def _setup(config):
    """Load artifacts, calibrate thresholds, find the split boundary, load logs. Shared by
    both scoring modes so this I/O + calibration only happens once per process."""
    store = EntityProfileStore.load_json(config["profile_store_path"])
    detector = AnomalyDetector.load()
    classifier = AttackClassifier.load()
    seq_detector = SequenceAnomalyDetector.load()
    normal_idx = list(classifier.model.classes_).index("normal")
    calib = _calibrate(config)
    boundary_ts, test_event_ids = _load_split_boundary(config)
    logs = pd.read_csv(config["logs_path"], usecols=LOG_COLUMNS, parse_dates=["timestamp"])
    logs = logs.sort_values("timestamp").reset_index(drop=True)
    return {
        "store": store, "detector": detector, "classifier": classifier, "seq_detector": seq_detector,
        "normal_idx": normal_idx, "calib": calib, "boundary_ts": boundary_ts,
        "test_event_ids": test_event_ids, "logs": logs,
    }


def _warm_up(ctx, config):
    """Replay training-period history through the incremental state updater only (no model
    inference) so entity/IP state reflects real history at the test boundary."""
    store, logs, test_event_ids = ctx["store"], ctx["logs"], ctx["test_event_ids"]
    seq_feature_names = ctx["seq_detector"].feature_names_
    entity_states, ip_states = defaultdict(EntityState), defaultdict(IPState)

    warm_start = time.perf_counter()
    n_warmed = 0
    for row in logs.itertuples(index=False):
        if row.event_id in test_event_ids:
            break  # logs sorted by timestamp -- first test event reached, warm-up done
        profile = store.get(row.user_id) or EntityProfile(row.user_id)
        _update_and_compute_features(row, profile, entity_states[row.user_id], ip_states[row.src_ip], config, seq_feature_names)
        n_warmed += 1
    warm_elapsed = time.perf_counter() - warm_start
    return entity_states, ip_states, n_warmed, warm_elapsed


def run_single_event_mode(ctx, config, entity_states, ip_states):
    """Score each test event through IF/classifier/LSTM the instant it arrives."""
    store, logs, test_event_ids = ctx["store"], ctx["logs"], ctx["test_event_ids"]
    detector, classifier, seq_detector = ctx["detector"], ctx["classifier"], ctx["seq_detector"]
    normal_idx, calib = ctx["normal_idx"], ctx["calib"]
    seq_feature_names = seq_detector.feature_names_

    latencies_ms, alert_log = [], []
    n_scored = n_alerts = 0
    stream_start = time.perf_counter()

    for row in logs.itertuples(index=False):
        if row.event_id not in test_event_ids:
            continue

        t0 = time.perf_counter()
        profile = store.get(row.user_id) or EntityProfile(row.user_id)
        entity_state = entity_states[row.user_id]
        features = _update_and_compute_features(row, profile, entity_state, ip_states[row.src_ip], config, seq_feature_names)

        X_row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        anomaly_score = float(detector.score(X_row)[0])
        probs = classifier.predict_proba(X_row)[0]
        max_idx = int(np.argmax(probs))
        pred_label = classifier.model.classes_[max_idx]
        max_prob = float(probs[max_idx])
        p_attack = 1.0 - float(probs[normal_idx])
        sequence_score = _score_sequence(seq_detector, entity_state)

        alerted, predicted_label, combined_risk = _decide(pred_label, max_prob, p_attack, anomaly_score, sequence_score, calib, config)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        n_scored += 1
        if alerted:
            n_alerts += 1
            alert_log.append({
                "event_id": row.event_id, "user_id": row.user_id, "timestamp": row.timestamp,
                "predicted_label": predicted_label, "true_label": row.label,
                "combined_risk": round(combined_risk, 2), "latency_ms": round(elapsed_ms, 3),
            })

        if n_scored >= config["n_events_to_stream"]:
            break

    stream_elapsed = time.perf_counter() - stream_start
    return {
        "mode": "single-event", "n_scored": n_scored, "n_alerts": n_alerts, "elapsed": stream_elapsed,
        "latencies_ms": np.array(latencies_ms), "alert_log": alert_log,
    }


def run_micro_batch_mode(ctx, config, entity_states, ip_states):
    """Buffer events (size or time trigger, whichever first); one model call per batch.
    Feature computation + state updates stay per-event and sequential; only the three model
    calls (IF, classifier, LSTM) are batched. Latency below includes the buffering wait."""
    store, logs, test_event_ids = ctx["store"], ctx["logs"], ctx["test_event_ids"]
    detector, classifier, seq_detector = ctx["detector"], ctx["classifier"], ctx["seq_detector"]
    normal_idx, calib = ctx["normal_idx"], ctx["calib"]
    seq_feature_names = seq_detector.feature_names_
    batch_size = config["micro_batch_size"]
    max_wait_s = config["micro_batch_max_wait_ms"] / 1000.0

    latencies_ms, alert_log = [], []
    counters = {"n_scored": 0, "n_alerts": 0}
    n_flushes_by_size, n_flushes_by_timeout = 0, 0

    def _flush(buffer):
        nonlocal n_flushes_by_size
        if not buffer:
            return
        X_batch = pd.DataFrame([b["features"] for b in buffer], columns=FEATURE_COLUMNS)
        anomaly_scores = detector.score(X_batch)
        probs = classifier.predict_proba(X_batch)
        max_idx = np.argmax(probs, axis=1)
        pred_labels = classifier.model.classes_[max_idx]
        max_probs = probs[np.arange(len(probs)), max_idx]
        p_attacks = 1.0 - probs[:, normal_idx]
        sequence_scores = _score_sequence_batch(seq_detector, [b["window"] for b in buffer])

        now = time.perf_counter()
        for i, b in enumerate(buffer):
            alerted, predicted_label, combined_risk = _decide(
                pred_labels[i], float(max_probs[i]), float(p_attacks[i]),
                float(anomaly_scores[i]), float(sequence_scores[i]), calib, config,
            )
            elapsed_ms = (now - b["enqueue_time"]) * 1000
            latencies_ms.append(elapsed_ms)
            counters["n_scored"] += 1
            if alerted:
                counters["n_alerts"] += 1
                row = b["row"]
                alert_log.append({
                    "event_id": row.event_id, "user_id": row.user_id, "timestamp": row.timestamp,
                    "predicted_label": predicted_label, "true_label": row.label,
                    "combined_risk": round(combined_risk, 2), "latency_ms": round(elapsed_ms, 3),
                })

    buffer = []
    batch_open_time = None
    stream_start = time.perf_counter()

    for row in logs.itertuples(index=False):
        if row.event_id not in test_event_ids:
            continue

        enqueue_time = time.perf_counter()
        profile = store.get(row.user_id) or EntityProfile(row.user_id)
        entity_state = entity_states[row.user_id]
        features = _update_and_compute_features(row, profile, entity_state, ip_states[row.src_ip], config, seq_feature_names)
        window = _build_window(seq_detector, entity_state)
        buffer.append({"row": row, "features": features, "window": window, "enqueue_time": enqueue_time})
        if batch_open_time is None:
            batch_open_time = enqueue_time

        size_trigger = len(buffer) >= batch_size
        timeout_trigger = (time.perf_counter() - batch_open_time) >= max_wait_s
        if size_trigger or timeout_trigger:
            n_flushes_by_size += int(size_trigger)
            n_flushes_by_timeout += int(timeout_trigger and not size_trigger)
            _flush(buffer)
            buffer = []
            batch_open_time = None

        if counters["n_scored"] >= config["n_events_to_stream"]:
            break

    if buffer and counters["n_scored"] < config["n_events_to_stream"]:
        _flush(buffer)  # final partial flush at end of stream, below either trigger

    stream_elapsed = time.perf_counter() - stream_start
    return {
        "mode": "micro-batch", "n_scored": counters["n_scored"], "n_alerts": counters["n_alerts"],
        "elapsed": stream_elapsed, "latencies_ms": np.array(latencies_ms), "alert_log": alert_log,
        "n_flushes_by_size": n_flushes_by_size, "n_flushes_by_timeout": n_flushes_by_timeout,
    }


def _print_result(result, config):
    n_scored, latencies = result["n_scored"], result["latencies_ms"]
    print(f"--- {result['mode']} mode ---")
    print("Sample of alerts as they were emitted (first 5):")
    sample_cols = ["event_id", "user_id", "timestamp", "predicted_label", "true_label", "combined_risk", "latency_ms"]
    if result["alert_log"]:
        print(pd.DataFrame(result["alert_log"][:5])[sample_cols].to_string(index=False))
    else:
        print("  (none)")
    if result["mode"] == "micro-batch":
        print(f"\nBatch flushes: {result['n_flushes_by_size']} by size ({config['micro_batch_size']} events), "
              f"{result['n_flushes_by_timeout']} by timeout ({config['micro_batch_max_wait_ms']}ms)")
    print(f"\nTotal events scored:        {n_scored}")
    print(f"Alerts emitted:             {result['n_alerts']}  ({result['n_alerts'] / n_scored:.1%} of scored events)")
    print(f"Wall-clock time:            {result['elapsed']:.2f}s")
    print(f"Throughput:                 {n_scored / result['elapsed']:.1f} events/sec")
    print(f"Mean per-event latency:     {latencies.mean():.2f} ms")
    print(f"p95 per-event latency:      {np.percentile(latencies, 95):.2f} ms")
    print(f"p99 per-event latency:      {np.percentile(latencies, 99):.2f} ms")
    print(f"Max per-event latency:      {latencies.max():.2f} ms\n")


def main():
    config = DEFAULT_CONFIG
    print("=== Real-Time Streaming Scorer ===\n")

    print("Loading trained artifacts once (profiles, Isolation Forest, classifier, LSTM autoencoder)...")
    ctx = _setup(config)
    print(f"  {len(ctx['store'].profiles)} entity profiles, IF, Random Forest classifier, LSTM autoencoder ready.\n")

    print("Calibrating alert thresholds + risk normalization bounds from the last batch run "
          "(data/predictions.csv)\n-- one-time and offline, NOT part of per-event latency below...")
    calib = ctx["calib"]
    pct_top = 100 - config["safety_net_percentile"]
    print(f"  IF safety-net threshold (top {pct_top:.1f}% most anomalous): {calib['if_threshold']:.4f}")
    print(f"  sequence safety-net threshold (top {pct_top:.1f}% most anomalous): {calib['seq_threshold']:.4f}")
    print(f"  IF score normalization range:       [{calib['if_min']:.4f}, {calib['if_max']:.4f}]")
    print(f"  sequence score normalization range: [{calib['seq_min']:.4f}, {calib['seq_max']:.4f}]\n")
    print(f"Test period starts at {ctx['boundary_ts']}, {len(ctx['test_event_ids'])} candidate test events\n")

    print("Warming up incremental state by replaying training-period history ONCE (state updates "
          "only,\nno model inference, NOT part of per-event latency below) -- both modes start "
          "from an\nidentical, independently-copied snapshot of this warmed-up state...")
    entity_states_warm, ip_states_warm, n_warmed, warm_elapsed = _warm_up(ctx, config)
    print(f"  warmed up state from {n_warmed} historical events in {warm_elapsed:.2f}s\n")

    print(f"=== Streaming {config['n_events_to_stream']} test-period events: single-event mode ===\n")
    single_result = run_single_event_mode(ctx, config, copy.deepcopy(entity_states_warm), copy.deepcopy(ip_states_warm))
    _print_result(single_result, config)

    print(f"=== Streaming {config['n_events_to_stream']} test-period events: micro-batch mode "
          f"(batch={config['micro_batch_size']}, max_wait={config['micro_batch_max_wait_ms']}ms) ===\n")
    batch_result = run_micro_batch_mode(ctx, config, copy.deepcopy(entity_states_warm), copy.deepcopy(ip_states_warm))
    _print_result(batch_result, config)

    print("=== Mode Comparison ===")
    comparison = pd.DataFrame([
        {
            "mode": r["mode"], "events_scored": r["n_scored"],
            "throughput_ev_per_sec": r["n_scored"] / r["elapsed"],
            "mean_latency_ms": r["latencies_ms"].mean(),
            "p95_latency_ms": np.percentile(r["latencies_ms"], 95),
        }
        for r in (single_result, batch_result)
    ])
    print(comparison.round(2).to_string(index=False))
    speedup = (batch_result["n_scored"] / batch_result["elapsed"]) / (single_result["n_scored"] / single_result["elapsed"])
    print(
        f"\n=== Real-time feasibility: micro-batching (batch={config['micro_batch_size']}) delivers "
        f"{speedup:.1f}x the throughput of\nsingle-event scoring ({batch_result['n_scored'] / batch_result['elapsed']:.0f} "
        f"vs {single_result['n_scored'] / single_result['elapsed']:.0f} events/sec) by amortizing the fixed "
        f"per-call overhead of the IF,\nRandom Forest, and LSTM model calls across a batch, while incremental "
        f"feature computation and state\nupdates stay strictly per-event and sequential throughout -- both "
        f"modes remain available; micro-batching\ntrades a few-events'-worth of buffering latency for "
        f"materially higher sustained throughput. ==="
    )


if __name__ == "__main__":
    main()
