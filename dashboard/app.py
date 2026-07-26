"""Streamlit analyst dashboard: alerts triage, analytics, and model internals."""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from explain.shap_explainer import AlertExplainer
from features.feature_engineering import FEATURE_COLUMNS
from models.anomaly_detector import AnomalyDetector
from models.attack_classifier import AttackClassifier
from models.train import _assign_incident_ids

DATA_DIR = ROOT_DIR / "data"
PREDICTIONS_PATH = DATA_DIR / "predictions.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
ALERTS_PATH = DATA_DIR / "alerts.csv"
LOGS_PATH = DATA_DIR / "logs.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"

TIER_ORDER = ["critical", "high", "medium", "low"]
TIER_COLORS = {"critical": "#d03b3b", "high": "#ec835a", "medium": "#fab219", "low": "#0ca30c"}

ATTACK_TYPES = [
    "brute_force", "credential_misuse", "credential_stuffing", "device_spoofing",
    "impossible_travel", "lateral_movement", "low_and_slow_exfiltration",
]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
ATTACK_COLORS = dict(zip(ATTACK_TYPES, CATEGORICAL_DARK))

SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"


# --------------------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------------------

@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])


@st.cache_data
def load_alerts():
    df = pd.read_csv(ALERTS_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def load_logs():
    return pd.read_csv(LOGS_PATH, parse_dates=["timestamp"])


@st.cache_resource
def load_models():
    return AnomalyDetector.load(), AttackClassifier.load()


@st.cache_resource
def load_explainer():
    return AlertExplainer().fit()


@st.cache_data
def build_annotated_predictions():
    """predictions.csv + user_id/timestamp, for anything that needs the saved pipeline decision."""
    predictions = load_predictions()
    features = load_features()
    return predictions.merge(features[["event_id", "user_id", "timestamp"]], on="event_id", how="left")


@st.cache_data
def build_scored_test_set():
    """Full test set with a freshly-computed classifier probability, for the interactive slider."""
    predictions = load_predictions()
    features = load_features()
    _, classifier = load_models()
    merged = predictions.merge(
        features[["event_id", "user_id", "timestamp"] + FEATURE_COLUMNS], on="event_id", how="left"
    )
    X = merged[FEATURE_COLUMNS].astype(float)
    probs = classifier.predict_proba(X)
    classes = classifier.model.classes_
    max_idx = probs.argmax(axis=1)
    merged["classifier_pred"] = classes[max_idx]
    merged["classifier_max_prob"] = probs[np.arange(len(probs)), max_idx]
    merged["is_attack"] = merged["true_label"] != "normal"
    return merged


# --------------------------------------------------------------------------------------
# Metrics helpers
# --------------------------------------------------------------------------------------

def alerts_at_cutoff(scored, classifier_cutoff, if_percentile=99.5):
    if_threshold = np.percentile(scored["anomaly_score"], if_percentile)
    classifier_alert = (scored["classifier_pred"] != "normal") & (scored["classifier_max_prob"] > classifier_cutoff)
    if_alert = scored["anomaly_score"] >= if_threshold
    return (classifier_alert | if_alert).to_numpy()


def rate_metrics(alerted_mask, is_attack):
    alerted_mask = np.asarray(alerted_mask)
    is_attack = np.asarray(is_attack)
    tp = int((alerted_mask & is_attack).sum())
    fp = int((alerted_mask & ~is_attack).sum())
    fn = int((~alerted_mask & is_attack).sum())
    tn = int((~alerted_mask & ~is_attack).sum())
    return {
        "detection_rate": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def per_type_detection_rates(annotated_predictions, incident_gap_minutes=180):
    eval_df = annotated_predictions.copy()
    eval_df["event_detected"] = eval_df["alerted"] & (eval_df["predicted_label"] == eval_df["true_label"])
    rows = []
    for attack_type in ATTACK_TYPES:
        subset = eval_df[eval_df["true_label"] == attack_type]
        if subset.empty:
            continue
        if attack_type == "impossible_travel":
            incident_id = _assign_incident_ids(subset, incident_gap_minutes)
            rate = subset["event_detected"].groupby(incident_id).any().mean()
        else:
            rate = subset["event_detected"].mean()
        rows.append({"attack_type": attack_type, "detection_rate": rate})
    return pd.DataFrame(rows)


def record_feedback(event_id, verdict, alert_row):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = pd.DataFrame([{
        "event_id": event_id,
        "user_id": alert_row["user_id"],
        "predicted_label": alert_row["predicted_label"],
        "risk_score": alert_row["risk_score"],
        "verdict": verdict,
        "recorded_at": pd.Timestamp.now(),
    }])
    header = not FEEDBACK_PATH.exists()
    entry.to_csv(FEEDBACK_PATH, mode="a", header=header, index=False)


def _extract_selected_rows(event):
    if event is None:
        return []
    try:
        return list(event["selection"]["rows"])
    except (TypeError, KeyError):
        pass
    try:
        return list(event.selection.rows)
    except AttributeError:
        return []


# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------

def badge_html(text, color):
    return (
        f'<span style="background-color:{color}22;color:{color};border:1px solid {color};'
        f'padding:2px 10px;border-radius:12px;font-size:0.82em;font-weight:600;white-space:nowrap;">{text}</span>'
    )


def tier_badge(tier):
    return badge_html(str(tier).upper(), TIER_COLORS.get(tier, INK_MUTED))


def _style_dark_axes(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.grid(color=GRIDLINE, linewidth=0.6, alpha=0.6)


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------

def render_kpi_row(predictions, alerts):
    total_events = len(predictions)
    total_alerts = int(predictions["alerted"].sum())
    is_attack = (predictions["true_label"] != "normal").to_numpy()
    alerted = predictions["alerted"].to_numpy()
    m = rate_metrics(alerted, is_attack)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events monitored", f"{total_events:,}")
    c2.metric("Alerts raised", f"{total_alerts:,}")
    c3.metric("Detection rate", f"{m['detection_rate']:.1%}")
    c4.metric("False positive rate", f"{m['false_positive_rate']:.2%}")

    tier_counts = alerts["risk_tier"].value_counts().reindex(TIER_ORDER).fillna(0).astype(int)
    st.markdown(
        "&nbsp;&nbsp;&nbsp;&nbsp;".join(f"{tier_badge(t)}&nbsp;<b>{tier_counts[t]}</b>" for t in TIER_ORDER),
        unsafe_allow_html=True,
    )
    st.caption("Detection / FPR above are any-alert-on-an-attack-event (see Analytics for the incident-aware, per-type breakdown).")
    st.divider()


def render_live_mode(alerts):
    st.subheader("🔴 Live Replay")
    col1, col2, col3 = st.columns([2, 2, 1])
    n_replay = col1.slider("Events to replay", 10, len(alerts), min(150, len(alerts)), key="live_n")
    speed = col2.slider("Playback speed (sec/event)", 0.02, 0.5, 0.06, step=0.02, key="live_speed")
    start = col3.button("▶ Start", type="primary", width="stretch")

    feed_area = st.empty()
    if start:
        subset = alerts.tail(n_replay).reset_index(drop=True)
        progress = st.progress(0.0)
        shown = []
        for i, row in subset.iterrows():
            shown.insert(0, row)
            shown = shown[:12]
            with feed_area.container():
                for r in shown:
                    ts = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(
                        f"{tier_badge(r['risk_tier'])}&nbsp;&nbsp;`{ts}`&nbsp;&nbsp;"
                        f"**{r['user_id']}**&nbsp;&nbsp;→&nbsp;&nbsp;**{r['predicted_label']}**"
                        f"&nbsp;&nbsp;risk={r['risk_score']:.1f}",
                        unsafe_allow_html=True,
                    )
                    st.caption(r["explanation"])
            progress.progress((i + 1) / len(subset))
            time.sleep(speed)
        st.success(f"Replay complete — streamed {len(subset)} alerts in timestamp order.")
    else:
        feed_area.info("Configure playback above and press ▶ Start to replay alerts as a live feed.")
    st.divider()


def render_alert_detail(event_id, alerts, logs, explainer):
    row = alerts.loc[alerts["event_id"] == event_id].iloc[0]
    raw = logs.loc[logs["event_id"] == event_id]
    entity_type = raw.iloc[0]["entity_type"] if not raw.empty else "unknown"
    st.markdown(f"#### Alert detail — `{event_id}`")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk score", f"{row['risk_score']:.1f}")
    c2.markdown(f"**Tier**<br>{tier_badge(row['risk_tier'])}", unsafe_allow_html=True)
    c3.metric("Predicted", row["predicted_label"])
    c4.metric("True label", row["true_label"])
    c5.metric("Entity type", entity_type)

    if bool(row.get("linked_via_incident", False)):
        st.markdown(badge_html("LINKED DETECTION", "#9085e9"), unsafe_allow_html=True)
        st.caption(
            f"This event was not independently detected — it was linked via incident "
            f"correlation with event `{row['linked_from_event_id']}`, which was."
        )

    st.markdown("**Top SHAP reasons**")
    try:
        for r in explainer.explain_event(event_id, top_n=5):
            if r["is_linked"]:
                st.markdown(f"- {r['sentence']}")
            else:
                st.markdown(f"- `{r['shap_value']:+.3f}`  **{r['feature']}** — {r['sentence']}")
    except KeyError:
        st.info("No SHAP explanation available for this event.")

    st.markdown("**Recent activity timeline (same user)**")
    user_events = logs[logs["user_id"] == row["user_id"]].copy()
    user_events["gap_minutes"] = (user_events["timestamp"] - row["timestamp"]).dt.total_seconds() / 60
    nearby = user_events.reindex(user_events["gap_minutes"].abs().sort_values().index).head(15)
    nearby = nearby.sort_values("timestamp").copy()
    nearby["is_this_event"] = nearby["event_id"] == event_id
    st.dataframe(
        nearby[["timestamp", "event_type", "resource", "country", "city", "device_fingerprint", "success", "label", "is_this_event"]],
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Raw event fields**")
    if not raw.empty:
        st.dataframe(raw.T.rename(columns={raw.index[0]: "value"}), width="stretch")

    st.markdown("**Analyst verdict**")
    fb1, fb2, _ = st.columns([1, 1, 3])
    if fb1.button("✅ Confirm threat", key=f"confirm_{event_id}", width="stretch"):
        record_feedback(event_id, "confirmed_threat", row)
        st.success("Recorded: confirmed threat.")
    if fb2.button("❌ False positive", key=f"fp_{event_id}", width="stretch"):
        record_feedback(event_id, "false_positive", row)
        st.success("Recorded: false positive.")


def render_alerts_tab(alerts, logs, explainer):
    if st.session_state.get("live_mode"):
        render_live_mode(alerts)

    st.subheader("Alerts")
    f1, f2, f3 = st.columns(3)
    tiers = f1.multiselect("Risk tier", TIER_ORDER, default=TIER_ORDER)
    types = sorted(alerts["predicted_label"].unique())
    attack_filter = f2.multiselect("Attack type", types, default=types)
    user_query = f3.text_input("User ID contains", "")

    filtered = alerts[alerts["risk_tier"].isin(tiers) & alerts["predicted_label"].isin(attack_filter)]
    if user_query:
        filtered = filtered[filtered["user_id"].str.contains(user_query, case=False, na=False)]
    filtered = filtered.sort_values("risk_score", ascending=False).reset_index(drop=True)

    st.caption(f"{len(filtered)} of {len(alerts)} alerts shown")

    display_df = filtered[["timestamp", "user_id", "predicted_label", "risk_score", "risk_tier", "explanation"]].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    event = st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=380,
        on_select="rerun",
        selection_mode="single-row",
        key="alerts_table",
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
        },
    )

    selected_rows = _extract_selected_rows(event)
    st.divider()
    if selected_rows:
        selected_event_id = filtered.iloc[selected_rows[0]]["event_id"]
        render_alert_detail(selected_event_id, alerts, logs, explainer)
    else:
        st.info("Select a row in the table above to see the full detail view.")


def render_analytics_tab(annotated_predictions, alerts, scored):
    st.subheader("Detection rate by attack type")
    per_type = per_type_detection_rates(annotated_predictions)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = [ATTACK_COLORS[t] for t in per_type["attack_type"]]
    bars = ax.bar(per_type["attack_type"], per_type["detection_rate"], color=colors, width=0.55)
    for b, v in zip(bars, per_type["detection_rate"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", color=INK_PRIMARY, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Detection rate")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _style_dark_axes(ax, fig)
    st.pyplot(fig, clear_figure=True)
    st.caption("impossible_travel is counted per-incident (either of its 2 events counts); other types are per-event.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Alerts over time")
        ts = alerts.set_index("timestamp").resample("6h").size().rename("alerts")
        st.area_chart(ts, color=CATEGORICAL_DARK[0])
    with col2:
        st.subheader("Risk tier distribution")
        tier_counts = alerts["risk_tier"].value_counts().reindex(TIER_ORDER).fillna(0).astype(int)
        fig2, ax2 = plt.subplots(figsize=(4.2, 3.2))
        ax2.bar(TIER_ORDER, tier_counts.values, color=[TIER_COLORS[t] for t in TIER_ORDER], width=0.55)
        ax2.set_ylabel("Alerts")
        _style_dark_axes(ax2, fig2)
        st.pyplot(fig2, clear_figure=True)

    st.divider()
    st.subheader("Operating trade-off: classifier probability cutoff")
    st.caption("IF safety net held fixed at the top 0.5% most anomalous; only the classifier confidence cutoff varies.")
    cutoff = st.slider("Classifier probability cutoff", 0.0, 1.0, 0.5, step=0.05, key="cutoff_slider")

    grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    is_attack = scored["is_attack"].to_numpy()
    curve = pd.DataFrame([rate_metrics(alerts_at_cutoff(scored, c), is_attack) for c in grid], index=grid)

    fig3, ax3 = plt.subplots(figsize=(8, 3.2))
    ax3.plot(grid, curve["detection_rate"], color=CATEGORICAL_DARK[0], linewidth=2, label="Detection rate")
    ax3.plot(grid, curve["false_positive_rate"], color=CATEGORICAL_DARK[7], linewidth=2, label="False positive rate")
    ax3.axvline(cutoff, color=INK_MUTED, linestyle="--", linewidth=1)
    ax3.set_xlabel("Classifier probability cutoff")
    ax3.set_ylim(0, 1.05)
    legend = ax3.legend(facecolor=SURFACE, edgecolor=GRIDLINE)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)
    _style_dark_axes(ax3, fig3)
    st.pyplot(fig3, clear_figure=True)

    m = rate_metrics(alerts_at_cutoff(scored, cutoff), is_attack)
    c1, c2, c3 = st.columns(3)
    c1.metric("Detection rate", f"{m['detection_rate']:.1%}")
    c2.metric("Precision", f"{m['precision']:.1%}")
    c3.metric("False positive rate", f"{m['false_positive_rate']:.2%}")

    st.caption("Fixed reference points:")
    ref_rows = [{"classifier_cutoff": c, **rate_metrics(alerts_at_cutoff(scored, c), is_attack)} for c in [0.3, 0.5, 0.7]]
    ref_table = pd.DataFrame(ref_rows)
    st.dataframe(
        ref_table.style.format({"detection_rate": "{:.1%}", "precision": "{:.1%}", "false_positive_rate": "{:.2%}"}),
        hide_index=True,
        width="stretch",
    )


def render_model_tab(predictions, scored):
    st.subheader("Architecture")
    st.markdown(
        """
**Three-signal design:**
1. **Isolation Forest** (unsupervised) trained only on events labeled `normal` in the training
   period — it never sees attack labels, it just learns what "normal" looks like and scores
   deviation from that.
2. **Random Forest classifier** (`class_weight="balanced"`), trained on all 8 labels
   (7 attack types + normal), assigns each event a specific attack type with a confidence.
3. **LSTM autoencoder** (sequence detector) trained only on normal training-split sequences —
   reconstructs each entity's trailing 10-event window; high reconstruction error means the
   *sequence* of recent actions doesn't look like a normal run, even if no single event does.

**Hybrid alert rule:** an event is alerted if the classifier is >50% confident it's an attack
type, OR the Isolation Forest score lands in the top 0.5% most anomalous, OR the sequence
reconstruction error lands in the top 0.5% — a safety net (in either unsupervised direction)
for attacks that don't match a known per-event pattern, surfaced as `unknown_anomaly`.

**Risk scoring:** the analyst-facing risk score is 100x the pipeline's own validated
`combined_risk` — the same 3-signal blend (classifier p_attack + Isolation Forest + LSTM
sequence detector, weighted and selected on a held-out validation split) that drives the
hybrid alert rule and the top-1% alert budget, not a separately recomputed blend. Mapped to
low / medium / high / critical. `unknown_anomaly` events (classifier itself said "normal")
use a pure IF+sequence blend instead, since the classifier has no attack-type opinion to
weight in there.

**Explainability:** a SHAP `TreeExplainer` on the Random Forest ranks which features pushed the
prediction toward its predicted class (or away from "normal", for `unknown_anomaly` events),
converted into plain-English sentences via per-feature templates.
        """
    )

    st.divider()
    st.subheader("Metrics")
    is_attack = scored["is_attack"].to_numpy()
    roc_auc = roc_auc_score(is_attack, scored["anomaly_score"])
    m = rate_metrics(predictions["alerted"].to_numpy(), is_attack)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IF ROC-AUC", f"{roc_auc:.3f}")
    c2.metric("Pipeline precision", f"{m['precision']:.1%}")
    c3.metric("Pipeline detection rate", f"{m['detection_rate']:.1%}")
    c4.metric("Pipeline FPR", f"{m['false_positive_rate']:.2%}")

    st.divider()
    st.subheader("Classifier confusion matrix (test set)")
    labels = sorted(scored["true_label"].unique())
    cm = confusion_matrix(scored["true_label"], scored["classifier_pred"], labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", color=INK_SECONDARY)
    ax.set_yticklabels(labels, color=INK_SECONDARY)
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("True", color=INK_SECONDARY)
    vmax = cm.max()
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val > vmax / 2 else INK_SECONDARY, fontsize=8)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig, clear_figure=True)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="UEBA Anomaly Detection", page_icon="🛡️", layout="wide")

    predictions = load_predictions()
    annotated_predictions = build_annotated_predictions()
    alerts = load_alerts()
    logs = load_logs()
    explainer = load_explainer()
    scored = build_scored_test_set()

    st.title("🛡️ UEBA — Behavioral Anomaly Detection")
    st.caption("Two-stage anomaly detection over synthetic corporate access logs (test period).")

    render_kpi_row(predictions, alerts)

    st.sidebar.header("Controls")
    st.sidebar.toggle("🔴 Live mode", value=False, key="live_mode")
    st.sidebar.caption("When on, the Alerts tab replays alerts as a live feed instead of a static table.")

    tab_alerts, tab_analytics, tab_model = st.tabs(["🚨 Alerts", "📊 Analytics", "🧠 Model Info"])
    with tab_alerts:
        render_alerts_tab(alerts, logs, explainer)
    with tab_analytics:
        render_analytics_tab(annotated_predictions, alerts, scored)
    with tab_model:
        render_model_tab(predictions, scored)


if __name__ == "__main__":
    main()
