# UEBA — AI-Powered Behavioral Anomaly Detection

User and Entity Behavior Analytics (UEBA) system for cybersecurity, built for the
Honeywell Hackathon. Generates synthetic access logs, engineers behavioral features,
detects anomalies, classifies attack types, explains predictions with SHAP, and
surfaces everything in an interactive Streamlit dashboard — plus a real-time
streaming scorer and cold-start/insider-drift demos.

## Project layout

- `data_gen/log_generator.py` — synthetic log generator: ~124k events for 200
  entities (users/service accounts/edge devices) over 30 days, deterministic
  (seed=42). Per-entity persistent profiles (home city, work hours, devices, usual
  resources, auth method, session-duration distribution). Injects 7 labeled attack
  types (brute_force, impossible_travel, lateral_movement, device_spoofing,
  credential_misuse, credential_stuffing, low_and_slow_exfiltration) plus
  insider_drift (legitimate footprint expansion, labeled normal, tracked separately
  for the drift demo).
- `features/feature_engineering.py` — builds per-entity baselines from each
  entity's own *normal-labeled* history, then computes 28 features per event:
  baseline-deviation flags, geo-velocity, 30-min rolling counts, cross-entity
  source-IP features, and 7-day cumulative features. Fully vectorized.
- `models/baseline_profile.py` — `EntityProfile`/`EntityProfileStore`: the formal
  per-entity baseline (hour histogram, known countries/devices/resources/auth
  methods/commands, session-duration stats), JSON-persisted.
- `models/anomaly_detector.py`, `attack_classifier.py`, `sequence_detector.py` —
  Isolation Forest (normal-only, unsupervised), Random Forest (all 8 labels,
  balanced), LSTM autoencoder (PyTorch, normal-only, 10-event windows). Three
  independent signals combined into a hybrid alert rule.
- `models/entity_day_detector.py` — a second RandomForest over (entity, day)
  aggregates to catch slow-accumulation patterns, wired into `train.py` with an
  accept/revert gate on validation performance.
- `models/train.py` — orchestrator. Chronological train/validation/test split;
  all thresholds, safety-net percentiles, and combined-risk weighting are selected
  on validation and evaluated once, frozen, on test. combined_risk uses
  p_attack = 1 - P(normal), weighted 70% classifier / 10% IF / 20% sequence.
  Also applies incident-aware linking as a supplementary output-only enrichment:
  once one event of an `impossible_travel` incident is alerted, the entity's other
  event in that incident is retroactively linked too (mirroring a SOC analyst
  pulling recent history), tagged `linked_via_incident`/`linked_from_event_id` in
  `predictions.csv`/`alerts.csv`. This never changes any metric computed before
  it -- raw per-event recall stays honestly reported alongside the linked number.
- `explain/shap_explainer.py`, `risk_scoring.py` — SHAP explanations (approximate
  mode) with plain-English templates for every feature; risk_scoring.py computes
  the analyst-facing 0-100 risk score/tier directly from the pipeline's validated
  combined_risk and writes `data/alerts.csv`. `impossible_travel` alerts that were
  backfilled by incident-linking (see below) get a plain-English linking sentence
  instead of a misleading SHAP explanation on their by-design-normal-looking features.
- `dashboard/app.py` — Streamlit UI: KPIs, live-replay mode, filterable alerts
  table with SHAP detail view + analyst feedback buttons, analytics charts,
  model-info tab. Alerts produced by incident-linking show a distinct "Linked
  detection" badge instead of looking identical to an independently-caught alert.
- `demo/cold_start_demo.py`, `drift_demo.py` — prove the system doesn't punish
  novelty (new entities) or legitimate slow footprint expansion (drift entities)
  with false alarms. `drift_demo.py` selects label-pure drift entities (100% true
  label `normal` in their drift window) by default so its printed false-positive
  rate isn't confounded by a coincidental, independently-injected real attack
  landing in the same date range -- only 1 of 12 tracked drift entities is fully
  clean; confounded entities are shown only as a clearly labeled secondary example.
- `streaming/stream_scorer.py` — proves real-time feasibility: incremental
  per-entity/per-IP state (no full-matrix recompute), single-event mode and
  micro-batch mode, with calibration and warm-up excluded from latency
  measurement.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Reproduce from scratch

```bash
python -m data_gen.log_generator
python -m features.feature_engineering
python -m models.train
python -m explain.risk_scoring
streamlit run dashboard/app.py
```

## Known limitations

- `unknown_anomaly` SHAP explanations use the classifier's attribution toward
  "not normal" as a proxy; they are not a direct attribution from the Isolation
  Forest or LSTM autoencoder, since tree-based SHAP doesn't apply to those model
  types directly.
- Only `impossible_travel` is evaluated incident-aware (either of its two events
  counts as detection); other multi-event attack types are scored per-event, which
  is a conservative lower bound on real-world SOC detection rate.
- `true_label` is shown in the dashboard's alert detail view for demo/evaluation
  purposes only — it would not be available at real inference time.
