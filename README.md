# UEBA — AI-Powered Behavioral Anomaly Detection

User and Entity Behavior Analytics (UEBA) system for cybersecurity. Generates
synthetic access logs, engineers behavioral features, detects anomalies,
classifies attack types, explains predictions with SHAP, and surfaces
everything in an interactive Streamlit dashboard.

## Project layout

- `data_gen/` — synthetic access log generator
- `features/` — feature engineering from raw logs
- `models/` — anomaly detection and attack classification models
- `explain/` — SHAP explanations and composite risk scoring
- `dashboard/` — Streamlit app for analysts

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Status

Skeleton only — no logic implemented yet.
