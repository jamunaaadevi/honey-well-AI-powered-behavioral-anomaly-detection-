"""Per-event behavioral feature engineering for UEBA anomaly detection."""

from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_profile import EntityProfileStore

DEFAULT_CONFIG = {
    "input_path": "data/logs.csv",
    "output_path": "data/features.csv",
    "baseline_fraction": 0.6,
    "min_baseline_events": 5,
    "rolling_window": "30min",
    "geo_velocity_cap_kmh": 3000,
    "profile_store_path": "models/artifacts/entity_profiles.json",
    "sensitive_resources": [
        "payroll-db", "employee-records-db", "domain-controller-01",
        "tax-reporting-app", "legal-vault", "contracts-db", "admin-console",
    ],
}

FEATURE_COLUMNS = [
    "has_baseline",
    "hour_of_day", "is_weekend", "is_outside_usual_hours",
    "is_new_country", "is_new_device", "is_new_resource", "is_new_auth_method",
    "geo_velocity_kmh", "minutes_since_last_event",
    "failed_login_count_last_30min", "total_event_count_last_30min",
    "distinct_resources_last_30min",
    "is_sensitive_resource", "login_fail",
    "user_typical_daily_event_count",
    "session_duration_zscore", "command_sequence_length", "is_rare_command",
    "is_user", "is_service_account", "is_edge_device",
    # cross-entity source-IP signals (brute_force vs credential_stuffing): brute force is one
    # entity failing many times from an IP, stuffing is many entities failing from an IP.
    "distinct_entities_from_this_ip_30min", "failed_logins_from_this_ip_30min", "fail_rate_from_this_ip_30min",
    # trailing 7-day cumulative signals: catches slow-building patterns (e.g. low_and_slow
    # exfiltration) that build gradually well outside any 30-minute window.
    "cumulative_new_resource_count_7d", "off_hours_access_count_7d", "sensitive_access_count_7d",
]


def _coerce_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False}).astype(bool)


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_user_baselines(df, config):
    """Per user: first `baseline_fraction` of that entity's NORMAL-labeled events only.

    Attack events must never be used to define "normal" for a compromised entity --
    df is sorted chronologically but attacks are scattered randomly across the 30-day
    period, so without this filter an attack that happens to land in an entity's early
    history would get absorbed into that entity's own baseline (its resources/hours/etc.
    would look "usual"), silently weakening detection of that exact pattern later. So we
    filter to label == "normal" *before* taking the chronological first
    `baseline_fraction` -- the 60% is 60% of each entity's normal history, not 60% of
    all its events regardless of label.

    Fits one models.baseline_profile.EntityProfile per entity (persisted to JSON as the
    official baseline artifact) and derives the merge-pair lookup tables straight from
    each profile's own sets/stats -- the profiles are the source of truth here, this just
    flattens them into the long-format tables the fast vectorized merges below need.
    """
    normal_df = df[df["label"] == "normal"]
    excluded = len(df) - len(normal_df)
    print(f"  build_user_baselines: excluded {excluded} non-normal events from baseline "
          f"consideration ({excluded / len(df):.2%} of all events)")

    rank = normal_df.groupby("user_id").cumcount()
    user_n = normal_df.groupby("user_id")["user_id"].transform("size")
    cutoff = (user_n * config["baseline_fraction"]).astype(int)
    baseline_df = normal_df[rank < cutoff]

    store = EntityProfileStore.fit(baseline_df, entity_col="user_id")
    if config.get("profile_store_path"):
        store.save_json(config["profile_store_path"])

    counts_rows, country_rows, device_rows, resource_rows = [], [], [], []
    hour_rows, auth_rows, daily_rows, session_rows, command_rows = [], [], [], [], []

    for entity_id, profile in store.profiles.items():
        counts_rows.append({"user_id": entity_id, "baseline_event_count": profile.event_count})
        daily_rows.append({"user_id": entity_id, "user_typical_daily_event_count": profile.daily_event_rate})
        session_rows.append({
            "user_id": entity_id,
            "session_duration_mean": profile.session_duration_mean,
            "session_duration_std": profile.session_duration_std,
        })
        country_rows.extend({"user_id": entity_id, "country": c, "is_known_country": True} for c in profile.countries)
        device_rows.extend(
            {"user_id": entity_id, "device_fingerprint": d, "is_known_device": True} for d in profile.devices
        )
        resource_rows.extend({"user_id": entity_id, "resource": r, "is_known_resource": True} for r in profile.resources)
        auth_rows.extend({"user_id": entity_id, "auth_method": a, "is_known_auth": True} for a in profile.auth_methods)
        command_rows.extend(
            {"user_id": entity_id, "command": cmd, "is_known_command": True} for cmd in profile.known_commands
        )
        hour_rows.extend(
            {"user_id": entity_id, "hour_of_day": h, "is_known_hour": True}
            for h, count in enumerate(profile.hour_histogram) if count > 0
        )

    def _frame(rows, cols):
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    return {
        "store": store,
        "counts": _frame(counts_rows, ["user_id", "baseline_event_count"]),
        "country_pairs": _frame(country_rows, ["user_id", "country", "is_known_country"]),
        "device_pairs": _frame(device_rows, ["user_id", "device_fingerprint", "is_known_device"]),
        "resource_pairs": _frame(resource_rows, ["user_id", "resource", "is_known_resource"]),
        "hour_pairs": _frame(hour_rows, ["user_id", "hour_of_day", "is_known_hour"]),
        "auth_pairs": _frame(auth_rows, ["user_id", "auth_method", "is_known_auth"]),
        "daily_counts": _frame(daily_rows, ["user_id", "user_typical_daily_event_count"]),
        "session_stats": _frame(session_rows, ["user_id", "session_duration_mean", "session_duration_std"]),
        "known_command_pairs": _frame(command_rows, ["user_id", "command", "is_known_command"]),
    }


def _merge_baseline_flags(df, baselines, config):
    df = df.merge(baselines["counts"], on="user_id", how="left")
    df["baseline_event_count"] = df["baseline_event_count"].fillna(0).astype(int)
    df["has_baseline"] = df["baseline_event_count"] >= config["min_baseline_events"]

    df = df.merge(baselines["country_pairs"], on=["user_id", "country"], how="left")
    df["is_new_country"] = df["is_known_country"].isna() & df["has_baseline"]

    df = df.merge(baselines["device_pairs"], on=["user_id", "device_fingerprint"], how="left")
    df["is_new_device"] = df["is_known_device"].isna() & df["has_baseline"]

    df = df.merge(baselines["resource_pairs"], on=["user_id", "resource"], how="left")
    df["is_new_resource"] = df["is_known_resource"].isna() & df["has_baseline"]

    df = df.merge(baselines["hour_pairs"], on=["user_id", "hour_of_day"], how="left")
    df["is_outside_usual_hours"] = df["is_known_hour"].isna() & df["has_baseline"]

    df = df.merge(baselines["auth_pairs"], on=["user_id", "auth_method"], how="left")
    df["is_new_auth_method"] = df["is_known_auth"].isna() & df["has_baseline"]

    df = df.merge(baselines["daily_counts"], on="user_id", how="left")
    df["user_typical_daily_event_count"] = df["user_typical_daily_event_count"].fillna(0)

    df = df.merge(baselines["session_stats"], on="user_id", how="left")
    std_safe = df["session_duration_std"].replace(0, np.nan)
    zscore = (df["session_duration"] - df["session_duration_mean"]) / std_safe
    df["session_duration_zscore"] = zscore.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df.drop(columns=[
        "is_known_country", "is_known_device", "is_known_resource", "is_known_hour", "is_known_auth",
        "session_duration_mean", "session_duration_std",
    ])


def _add_geo_velocity(df, config):
    grp = df.groupby("user_id")
    prev_lat = grp["lat"].shift(1)
    prev_lon = grp["lon"].shift(1)
    prev_ts = grp["timestamp"].shift(1)

    minutes_gap = (df["timestamp"] - prev_ts).dt.total_seconds() / 60
    distance_km = _haversine_km(df["lat"].to_numpy(), df["lon"].to_numpy(), prev_lat.to_numpy(), prev_lon.to_numpy())
    hours_gap = (minutes_gap / 60).clip(lower=1 / 3600)  # floor at 1 second to avoid divide-by-zero blowups

    velocity = pd.Series(distance_km, index=df.index) / hours_gap
    df["geo_velocity_kmh"] = velocity.fillna(0).clip(upper=config["geo_velocity_cap_kmh"])
    df["minutes_since_last_event"] = minutes_gap.fillna(-1)
    return df


def _add_command_features(df, baselines):
    has_commands = df["command_sequence"].fillna("") != ""
    df["command_sequence_length"] = np.where(
        has_commands, df["command_sequence"].fillna("").str.count(";") + 1, 0
    )

    cmd_events = df.loc[has_commands, ["event_id", "user_id", "command_sequence"]].copy()
    cmd_events["command"] = cmd_events["command_sequence"].str.split(";")
    exploded = cmd_events.explode("command")[["event_id", "user_id", "command"]]
    exploded = exploded.merge(baselines["known_command_pairs"], on=["user_id", "command"], how="left")
    per_event_unknown = exploded.groupby("event_id")["is_known_command"].apply(lambda s: s.isna().any())

    df["is_rare_command"] = df["event_id"].map(per_event_unknown).fillna(False) & df["has_baseline"]
    return df


def _add_rolling_window_features(df, config):
    window = config["rolling_window"]
    pieces = []
    for _, g in df.groupby("user_id", sort=False):
        g = g.sort_values("timestamp").set_index("timestamp")
        g["total_event_count_last_30min"] = g["event_id"].rolling(window).count()
        g["failed_login_count_last_30min"] = g["login_fail"].rolling(window).sum()
        resource_codes = pd.Series(pd.factorize(g["resource"])[0].astype(float), index=g.index)
        g["distinct_resources_last_30min"] = resource_codes.rolling(window).apply(
            lambda x: np.unique(x).size, raw=True
        )
        pieces.append(g.reset_index())
    return pd.concat(pieces, ignore_index=True)


def _add_cross_ip_features(df, config):
    """Per src_ip, rolling 30-min window across ALL entities sharing that IP.

    Grouped-rolling (not a per-group Python loop) so this stays fast with tens of
    thousands of distinct src_ip values: pre-sort by [src_ip, timestamp] so a sort=False
    groupby iterates groups in that same order, then pandas' own grouped-rolling
    implementation reassembles results in the identical row order -- verified positionally
    against the sorted frame's index before trusting a plain `.to_numpy()` assignment. The
    entity count is what separates brute_force (one entity, many fails, one IP) from
    credential_stuffing (many entities, few IPs, high fail rate).
    """
    window = config["rolling_window"]
    d = df[["event_id", "timestamp", "src_ip", "user_id", "login_fail"]].copy()
    d = d.sort_values(["src_ip", "timestamp"]).reset_index(drop=True)
    d = d.set_index("timestamp")
    d["_entity_code"] = pd.factorize(d["user_id"])[0].astype(float)

    grouped = d.groupby("src_ip", sort=False)
    total = grouped["event_id"].rolling(window).count()
    failed = grouped["login_fail"].rolling(window).sum()
    distinct_entities = grouped["_entity_code"].rolling(window).apply(lambda x: np.unique(x).size, raw=True)

    assert (total.index.get_level_values(1).to_numpy() == d.index.to_numpy()).all(), (
        "grouped-rolling result order diverged from the sorted frame -- do not trust positional assignment"
    )

    d["total_events_from_ip_30min"] = total.to_numpy()
    d["failed_logins_from_this_ip_30min"] = failed.to_numpy()
    d["distinct_entities_from_this_ip_30min"] = distinct_entities.to_numpy()
    d["fail_rate_from_this_ip_30min"] = (
        d["failed_logins_from_this_ip_30min"] / d["total_events_from_ip_30min"].replace(0, np.nan)
    ).fillna(0)

    result = d.reset_index()[[
        "event_id", "distinct_entities_from_this_ip_30min",
        "failed_logins_from_this_ip_30min", "fail_rate_from_this_ip_30min",
    ]]
    return df.merge(result, on="event_id", how="left")


def _add_cumulative_features(df, config):
    """Per entity, trailing 7-day cumulative counts.

    A 30-minute window can't see a pattern that deliberately builds up slowly (e.g.
    low_and_slow_exfiltration accessing a few more resources than usual each day, off-hours,
    over 1-2 weeks) -- these look for that at a week-long timescale instead. Only ~200
    entities, so a per-entity loop (matching the existing 30-min rolling code) is plenty fast.
    """
    window = "7D"
    pieces = []
    for _, g in df.groupby("user_id", sort=False):
        g = g.sort_values("timestamp").set_index("timestamp")
        g["cumulative_new_resource_count_7d"] = g["is_new_resource"].rolling(window).sum()
        g["off_hours_access_count_7d"] = g["is_outside_usual_hours"].rolling(window).sum()
        g["sensitive_access_count_7d"] = g["is_sensitive_resource"].rolling(window).sum()
        pieces.append(g.reset_index())
    return pd.concat(pieces, ignore_index=True)


def compute_features(df, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    df = df.copy()
    df["success"] = _coerce_bool(df["success"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df["login_fail"] = (df["event_type"] == "login") & (~df["success"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    df["is_sensitive_resource"] = df["resource"].isin(config["sensitive_resources"])
    df["is_user"] = df["entity_type"] == "user"
    df["is_service_account"] = df["entity_type"] == "service_account"
    df["is_edge_device"] = df["entity_type"] == "edge_device"

    baselines = build_user_baselines(df, config)
    df = _merge_baseline_flags(df, baselines, config)
    print(f"  has_baseline: {int(df['has_baseline'].sum())} of {len(df)} events "
          f"({df['has_baseline'].mean():.1%})")
    df = _add_command_features(df, baselines)
    df = _add_geo_velocity(df, config)
    df = _add_rolling_window_features(df, config)
    df = _add_cross_ip_features(df, config)
    df = _add_cumulative_features(df, config)

    output_cols = ["event_id", "user_id", "timestamp", "label"] + FEATURE_COLUMNS
    return df[output_cols].sort_values("event_id").reset_index(drop=True)


def main():
    config = DEFAULT_CONFIG
    df = pd.read_csv(config["input_path"], parse_dates=["timestamp"])
    features_df = compute_features(df, config)

    out_path = Path(config["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_path, index=False)

    print(f"Feature matrix shape: {features_df.shape}")
    print(f"Wrote features to {out_path}\n")

    print("Mean feature value by label:")
    summary = features_df.groupby("label")[FEATURE_COLUMNS].mean().round(3)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.T.to_string())


if __name__ == "__main__":
    main()
