"""Per-entity statistical behavior profile -- the formal baseline representation.

Built once from each entity's early ("baseline") event history, this is the source of
truth feature engineering derives its per-event new-country/new-device/... flags from.
It also stands alone: deviation_score(event) scores a single hypothetical or live event
against the profile without needing the rest of the pipeline, and the whole store
persists to JSON for reuse (dashboard, demos) without retraining.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ARTIFACT_PATH = "models/artifacts/entity_profiles.json"


class EntityProfile:
    """Usual hours, countries, devices, resources, auth methods, commands, session length."""

    def __init__(self, entity_id):
        self.entity_id = entity_id
        self.event_count = 0
        self.hour_histogram = [0] * 24
        self.countries = set()
        self.devices = set()
        self.resources = set()
        self.auth_methods = set()
        self.known_commands = set()
        self.session_duration_mean = 0.0
        self.session_duration_std = 0.0
        self.daily_event_rate = 0.0
        self.first_seen = None
        self.last_seen = None

    @classmethod
    def fit(cls, entity_id, events):
        """events: a DataFrame of this entity's baseline events."""
        profile = cls(entity_id)
        profile.event_count = len(events)
        if profile.event_count == 0:
            return profile

        timestamps = pd.to_datetime(events["timestamp"])
        for hour, count in timestamps.dt.hour.value_counts().items():
            profile.hour_histogram[int(hour)] = int(count)

        profile.countries = set(events["country"].dropna().unique())
        profile.devices = set(events["device_fingerprint"].dropna().unique())
        profile.resources = set(events["resource"].dropna().unique())

        if "auth_method" in events.columns:
            profile.auth_methods = set(events["auth_method"].dropna().unique())

        if "command_sequence" in events.columns:
            commands = events["command_sequence"].fillna("")
            tokens = commands[commands != ""].str.split(";").explode()
            profile.known_commands = set(tokens.dropna().unique())

        if "session_duration" in events.columns:
            sd = events["session_duration"].dropna()
            if len(sd) > 0:
                mean_val = sd.mean()
                std_val = sd.std()
                profile.session_duration_mean = float(mean_val) if pd.notna(mean_val) else 0.0
                profile.session_duration_std = float(std_val) if pd.notna(std_val) else 0.0

        profile.first_seen = timestamps.min()
        profile.last_seen = timestamps.max()
        span_days = max(1, (profile.last_seen - profile.first_seen).days + 1)
        profile.daily_event_rate = profile.event_count / span_days
        return profile

    def _hour_is_usual(self, hour):
        if hour is None:
            return True
        return self.hour_histogram[int(hour)] > 0

    def deviation_score(self, event):
        """event: dict-like with hour/hour_of_day, country, device_fingerprint, resource,
        auth_method, session_duration (any subset -- missing keys are simply skipped).
        Returns a 0-1 composite: 0 fits the profile perfectly, 1 is maximally novel.
        No baseline yet -> 0.0 (conservative: can't claim deviation without a baseline)."""
        if self.event_count == 0:
            return 0.0

        signals = []
        hour = event.get("hour", event.get("hour_of_day"))
        if hour is not None:
            signals.append(0.0 if self._hour_is_usual(hour) else 1.0)
        if event.get("country") is not None:
            signals.append(0.0 if event["country"] in self.countries else 1.0)
        if event.get("device_fingerprint") is not None:
            signals.append(0.0 if event["device_fingerprint"] in self.devices else 1.0)
        if event.get("resource") is not None:
            signals.append(0.0 if event["resource"] in self.resources else 1.0)
        if event.get("auth_method") is not None and self.auth_methods:
            signals.append(0.0 if event["auth_method"] in self.auth_methods else 1.0)

        session_duration = event.get("session_duration")
        if session_duration is not None and self.session_duration_std > 0:
            z = abs(session_duration - self.session_duration_mean) / self.session_duration_std
            signals.append(min(1.0, z / 4.0))

        return float(np.mean(signals)) if signals else 0.0

    def to_dict(self):
        return {
            "entity_id": self.entity_id,
            "event_count": self.event_count,
            "hour_histogram": self.hour_histogram,
            "countries": sorted(self.countries),
            "devices": sorted(self.devices),
            "resources": sorted(self.resources),
            "auth_methods": sorted(self.auth_methods),
            "known_commands": sorted(self.known_commands),
            "session_duration_mean": self.session_duration_mean,
            "session_duration_std": self.session_duration_std,
            "daily_event_rate": self.daily_event_rate,
            "first_seen": str(self.first_seen) if self.first_seen is not None else None,
            "last_seen": str(self.last_seen) if self.last_seen is not None else None,
        }

    @classmethod
    def from_dict(cls, d):
        profile = cls(d["entity_id"])
        profile.event_count = d["event_count"]
        profile.hour_histogram = d["hour_histogram"]
        profile.countries = set(d["countries"])
        profile.devices = set(d["devices"])
        profile.resources = set(d["resources"])
        profile.auth_methods = set(d["auth_methods"])
        profile.known_commands = set(d.get("known_commands", []))
        profile.session_duration_mean = d["session_duration_mean"]
        profile.session_duration_std = d["session_duration_std"]
        profile.daily_event_rate = d["daily_event_rate"]
        profile.first_seen = pd.Timestamp(d["first_seen"]) if d.get("first_seen") else None
        profile.last_seen = pd.Timestamp(d["last_seen"]) if d.get("last_seen") else None
        return profile


class EntityProfileStore:
    """A dict of entity_id -> EntityProfile, buildable in one pass and JSON-persistable."""

    def __init__(self, profiles=None):
        self.profiles = profiles or {}

    @classmethod
    def fit(cls, baseline_df, entity_col="user_id"):
        profiles = {
            entity_id: EntityProfile.fit(entity_id, group)
            for entity_id, group in baseline_df.groupby(entity_col, sort=False)
        }
        return cls(profiles)

    def get(self, entity_id):
        return self.profiles.get(entity_id)

    def save_json(self, path=DEFAULT_ARTIFACT_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({eid: p.to_dict() for eid, p in self.profiles.items()}, f, indent=2)
        return path

    @classmethod
    def load_json(cls, path=DEFAULT_ARTIFACT_PATH):
        with open(path) as f:
            raw = json.load(f)
        return cls({eid: EntityProfile.from_dict(d) for eid, d in raw.items()})
