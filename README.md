# AI-Powered Behavioral Anomaly Detection for Cybersecurity

A UEBA (User and Entity Behavior Analytics) system built for the Honeywell Hackathon.
It learns what normal activity looks like for each user, service account, and device,
flags anything unusual, explains why in plain English, and shows everything in a
live dashboard. It also includes a real-time streaming scorer and demos for
cold-start entities and behavior drift.

## Project layout

- `data_gen/log_generator.py` — synthetic login/access log generator
  - 200 entities (users, service accounts, edge devices) over 30 days, ~124k events
  - Same output every run (seed=42)
  - Each entity gets a persistent profile: home city, work hours, devices, usual resources, auth method
  - Injects 7 labeled attack types: brute force, impossible travel, lateral movement, device spoofing, credential misuse, credential stuffing, low-and-slow exfiltration
  - Plus a legitimate "insider drift" pattern, used only for the drift demo

- `features/feature_engineering.py` — turns raw logs into model-ready features
  - Builds each entity's baseline from that entity's own normal-labeled history only
  - Computes 28 behavioral features per event: baseline deviations, geo-velocity, rolling activity counts, and more

- `models/baseline_profile.py` — the formal per-entity baseline profile
  - Usual hours, countries, devices, resources, auth methods
  - Saved as JSON

- `models/anomaly_detector.py`, `attack_classifier.py`, `sequence_detector.py` — the three detection models
  - Isolation Forest — unsupervised anomaly score
  - Random Forest — names the likely attack type
  - LSTM Autoencoder — watches sequences of events, not single events alone
  - Their outputs combine into one hybrid alert rule

- `models/entity_day_detector.py` — a second, experimental detector
  - Looks at daily activity totals per entity
  - Only kept if it actually improves results on validation data — right now it doesn't, so it's switched off

- `models/train.py` — runs the whole training pipeline
  - Splits data chronologically into train / validation / test
  - Tunes every threshold and weight on validation only
  - Checks everything once on test data it never touched
  - Also links related events of a two-part `impossible_travel` attack together, the way a real analyst would, without changing any of the honestly-reported metrics

- `explain/shap_explainer.py`, `risk_scoring.py` — explains and scores every alert
  - Plain-English reason for every alert, using SHAP
  - A 0–100 risk score per alert

- `dashboard/app.py` — the analyst-facing dashboard
  - Ranked alerts table
  - Detail view with the reason for each flag
  - Analytics charts
  - A distinct badge for alerts caught through incident-linking

- `demo/cold_start_demo.py`, `drift_demo.py` — prove the system doesn't cry wolf
  - Cold-start: a brand-new entity with no history
  - Drift: an entity whose legitimate behavior changes over time

- `streaming/stream_scorer.py` — real-time scoring
  - Scores events one at a time or in small batches
  - Never recomputes everything from scratch

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

- For alerts caught only by the safety-net models (not the classifier itself), the SHAP explanation is based on the classifier's general attribution, not a direct explanation from the Isolation Forest or LSTM.
- Only `impossible_travel` is evaluated as a two-event incident. Other attack types are scored one event at a time, which is a stricter way to measure them.
- The dashboard shows each event's true label for demo purposes only — that wouldn't be available in a real deployment.
