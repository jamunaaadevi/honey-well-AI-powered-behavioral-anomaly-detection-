"""Synthetic UEBA access log generator with labeled attack injection."""

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

DEFAULT_CONFIG = {
    "seed": 42,
    "num_users": 200,
    "num_days": 30,
    "start_date": "2026-06-24",
    "total_events": 100_000,
    "attack_ratio": 0.03,
    "attack_type_weights": {
        "brute_force": 0.16,
        "impossible_travel": 0.14,
        "lateral_movement": 0.14,
        "device_spoofing": 0.11,
        "credential_misuse": 0.14,
        "credential_stuffing": 0.16,
        "low_and_slow_exfiltration": 0.15,
    },
    "login_fail_rate": 0.03,
    "entity_type_ratios": {"user": 0.80, "service_account": 0.10, "edge_device": 0.10},
    "num_drift_entities": 12,
    "output_path": "data/logs.csv",
    "drift_entities_path": "data/drift_entities.json",
    # canonical_field -> output column name; remap here to match a different log schema
    "columns": {
        "event_id": "event_id",
        "timestamp": "timestamp",
        "user_id": "entity_id",  # canonical key stays "user_id"; output column is now "entity_id"
        "entity_type": "entity_type",
        "event_type": "event_type",
        "src_ip": "src_ip",
        "country": "country",
        "city": "city",
        "lat": "lat",
        "lon": "lon",
        "device_id": "device_id",
        "device_fingerprint": "device_fingerprint",
        "resource": "resource",
        "auth_method": "auth_method",
        "session_duration": "session_duration",
        "command_sequence": "command_sequence",
        "success": "success",
        "label": "label",
    },
    # backward-compat: also emit the entity id under this column name (old code expects "user_id")
    "legacy_user_id_alias": "user_id",
}

CANONICAL_COLUMN_ORDER = list(DEFAULT_CONFIG["columns"].keys())

CITIES = [
    {"country": "United States", "city": "New York", "lat": 40.7128, "lon": -74.0060},
    {"country": "United States", "city": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"country": "United States", "city": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"country": "United Kingdom", "city": "London", "lat": 51.5074, "lon": -0.1278},
    {"country": "Germany", "city": "Berlin", "lat": 52.5200, "lon": 13.4050},
    {"country": "France", "city": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"country": "India", "city": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    {"country": "India", "city": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"country": "Singapore", "city": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"country": "Japan", "city": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"country": "Australia", "city": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"country": "Brazil", "city": "Sao Paulo", "lat": -23.5505, "lon": -46.6333},
    {"country": "Canada", "city": "Toronto", "lat": 43.6532, "lon": -79.3832},
    {"country": "South Africa", "city": "Johannesburg", "lat": -26.2041, "lon": 28.0473},
    {"country": "United Arab Emirates", "city": "Dubai", "lat": 25.2048, "lon": 55.2708},
    {"country": "Netherlands", "city": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
]

RESOURCES_BY_DEPT = {
    "engineering": ["eng-repo-01", "eng-repo-02", "ci-jenkins-01", "k8s-prod-01", "k8s-staging-02", "dev-vpn-gateway"],
    "finance": ["finance-erp", "finance-fileserver", "payroll-db", "tax-reporting-app"],
    "hr": ["hr-portal", "hr-fileserver", "employee-records-db"],
    "sales": ["crm-app", "sales-dashboard", "salesforce-sync"],
    "it": ["it-helpdesk", "admin-console", "backup-server-01", "domain-controller-01"],
    "legal": ["legal-vault", "contracts-db"],
    "marketing": ["marketing-cms", "analytics-dashboard"],
}
SHARED_RESOURCES = ["email-server", "vpn-gateway", "intranet-portal", "file-share-common"]
SENSITIVE_RESOURCES = [
    "payroll-db", "employee-records-db", "domain-controller-01",
    "tax-reporting-app", "legal-vault", "contracts-db", "admin-console",
]
EDGE_RESOURCES = ["iot-telemetry-gateway", "edge-config-service", "vpn-gateway"]
PRIVILEGED_RESOURCES = sorted(set(SENSITIVE_RESOURCES) | {"k8s-prod-01", "backup-server-01", "ci-jenkins-01"})
ALL_RESOURCES = sorted(
    {r for lst in RESOURCES_BY_DEPT.values() for r in lst}
    | set(SHARED_RESOURCES) | set(SENSITIVE_RESOURCES) | set(EDGE_RESOURCES)
)

OS_CHOICES = ["Windows 11", "macOS 14", "Ubuntu 22.04"]
BROWSER_CHOICES = ["Chrome", "Firefox", "Edge", "Safari"]

AUTH_METHODS = ["password", "token", "certificate", "biometric"]
AUTH_METHOD_PREFS = {
    "user": {"password": 0.70, "biometric": 0.20, "token": 0.10, "certificate": 0.00},
    "service_account": {"token": 0.60, "certificate": 0.40, "password": 0.00, "biometric": 0.00},
    "edge_device": {"certificate": 0.90, "token": 0.10, "password": 0.00, "biometric": 0.00},
}
# Plausible secondary methods per entity type -- deliberately excludes methods that type would
# never organically use (e.g. a human never authenticates via device certificate), so those
# stay genuinely novel if an attack introduces them (real signal for is_new_auth_method).
AUTH_METHOD_ALTERNATES = {
    "user": ["password", "biometric", "token"],
    "service_account": ["token", "certificate"],
    "edge_device": ["certificate", "token"],
}
COMMAND_POOL = [
    "whoami", "ls -la", "id", "uname -a", "ps aux", "netstat -an", "systemctl status",
    "sudo su", "cat /etc/passwd", "cat /etc/shadow", "chmod 777 /data", "history -c",
    "scp file remote:", "curl http://internal", "wget http://internal/payload",
    "rm -rf /tmp/cache", "tar -czf backup.tar.gz /data", "ssh admin@host",
]

AVG_EVENTS_PER_INCIDENT = {
    "brute_force": 27,
    "impossible_travel": 2,
    "lateral_movement": 20,
    "device_spoofing": 3,
    "credential_misuse": 2,
    "credential_stuffing": 69,
    "low_and_slow_exfiltration": 7,
}


def _rand_hex(rng, n):
    return "".join(rng.choices("0123456789abcdef", k=n))


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _jitter_coords(lat, lon, rng, spread=0.05):
    return lat + rng.uniform(-spread, spread), lon + rng.uniform(-spread, spread)


def _random_timestamp(start_ts, end_ts, rng):
    delta = (end_ts - start_ts).total_seconds()
    return start_ts + pd.Timedelta(seconds=rng.uniform(0, delta))


def _make_device(rng):
    os_choice = rng.choice(OS_CHOICES)
    browser = rng.choice(BROWSER_CHOICES)
    device_id = f"dev-{_rand_hex(rng, 10)}"
    fingerprint = hashlib.sha256(f"{device_id}-{os_choice}-{browser}-{rng.random()}".encode()).hexdigest()[:16]
    return {"device_id": device_id, "device_fingerprint": fingerprint, "os": os_choice, "browser": browser}


def _sample_auth_method(user, rng):
    """Usually the entity's preferred method; a small chance of a legitimate alternate."""
    if rng.random() < 0.02:
        return rng.choice(AUTH_METHOD_ALTERNATES[user["entity_type"]])
    return user["auth_method_pref"]


def _sample_session_duration(user, rng):
    # rng.lognormvariate (stdlib) is the same lognormal(mu, sigma) distribution as
    # np.random.lognormal -- using it keeps every draw on the one seeded `rng` instance
    # instead of numpy's separate global random state.
    return round(rng.lognormvariate(user["session_duration_mu"], 0.4), 1)


def _sample_command_sequence(event_type, resource, rng):
    """Only privileged sessions (host_connect to a privileged resource) carry a command trail."""
    if event_type == "host_connect" and resource in PRIVILEGED_RESOURCES:
        n = rng.randint(2, 5)
        return ";".join(rng.choices(COMMAND_POOL, k=n))
    return ""


def generate_user_profiles(config, rng):
    departments = list(RESOURCES_BY_DEPT.keys())
    entity_types = list(config["entity_type_ratios"].keys())
    type_weights = list(config["entity_type_ratios"].values())

    users = []
    for i in range(config["num_users"]):
        entity_type = rng.choices(entity_types, weights=type_weights, k=1)[0]
        home_city = rng.choice(CITIES)

        if entity_type == "user":
            id_prefix = "user"
            dept = rng.choice(departments)
            work_start = rng.randint(6, 9)
            work_end = min(23, work_start + rng.randint(7, 10))
            weekend_factor = rng.uniform(0.1, 0.35)
            num_devices = rng.choice([1, 1, 1, 2])
            dept_resources = RESOURCES_BY_DEPT[dept]
            usual_resources = sorted(set(
                rng.sample(dept_resources, k=min(len(dept_resources), rng.randint(2, 4)))
                + rng.sample(SHARED_RESOURCES, k=rng.randint(1, 2))
            ))
            daily_event_mean = rng.uniform(11, 31)
            login_ratio = rng.uniform(0.12, 0.22)
            session_duration_mu = rng.uniform(6.5, 8.0)
            regular_interval = False
        elif entity_type == "service_account":
            id_prefix = "svc"
            dept = rng.choice(["it", "engineering"])
            work_start, work_end = 0, 23
            weekend_factor = 1.0
            num_devices = 1
            dept_resources = RESOURCES_BY_DEPT[dept]
            usual_resources = sorted(set(
                rng.sample(dept_resources, k=min(len(dept_resources), rng.randint(2, 4)))
                + rng.sample(SHARED_RESOURCES, k=rng.randint(1, 2))
            ))
            daily_event_mean = rng.uniform(20, 50)
            login_ratio = rng.uniform(0.03, 0.08)
            session_duration_mu = rng.uniform(8.5, 10.5)
            regular_interval = False
        else:  # edge_device
            id_prefix = "dev"
            dept = None
            work_start, work_end = 0, 23
            weekend_factor = 1.0
            num_devices = 1
            usual_resources = sorted(set(rng.sample(EDGE_RESOURCES, k=rng.randint(1, 2))))
            daily_event_mean = rng.uniform(20, 40)
            login_ratio = rng.uniform(0.01, 0.03)
            session_duration_mu = rng.uniform(2.5, 4.0)
            regular_interval = True

        devices = [_make_device(rng) for _ in range(num_devices)]
        auth_pref_dist = AUTH_METHOD_PREFS[entity_type]
        auth_method_pref = rng.choices(list(auth_pref_dist.keys()), weights=list(auth_pref_dist.values()), k=1)[0]

        users.append({
            "user_id": f"{id_prefix}_{i + 1:04d}",
            "entity_type": entity_type,
            "department": dept,
            "home_city": home_city,
            "work_start": work_start,
            "work_end": work_end,
            "devices": devices,
            "usual_resources": usual_resources,
            "daily_event_mean": daily_event_mean,
            "weekend_factor": weekend_factor,
            "login_ratio": login_ratio,
            "home_subnet": f"10.{(i // 256) % 256}.{i % 256}",
            "auth_method_pref": auth_method_pref,
            "session_duration_mu": session_duration_mu,
            "regular_interval": regular_interval,
        })
    return users


def _sample_timestamp(day, user, rng):
    if rng.random() < 0.9:
        hour = rng.randint(user["work_start"], user["work_end"])
    else:
        hour = rng.randint(0, 23)
    return day + pd.Timedelta(hours=hour, minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))


def generate_normal_events(users, config, rng, np_rng, start_ts):
    events = []
    for user in users:
        for day_offset in range(config["num_days"]):
            day = start_ts + pd.Timedelta(days=day_offset)
            is_weekend = day.weekday() >= 5
            lam = user["daily_event_mean"] * (user["weekend_factor"] if is_weekend else 1.0)
            n_events = int(np_rng.poisson(lam)) if lam > 0 else 0
            if n_events == 0:
                continue

            if user["regular_interval"]:
                interval_minutes = 1440 / n_events
                timestamps = [
                    day + pd.Timedelta(minutes=max(0, k * interval_minutes + rng.uniform(-2, 2)))
                    for k in range(n_events)
                ]
            else:
                timestamps = [_sample_timestamp(day, user, rng) for _ in range(n_events)]

            for ts in timestamps:
                is_login = rng.random() < user["login_ratio"]
                event_type = "login" if is_login else rng.choice(["file_access", "host_connect"])
                if is_login:
                    resource = "auth-server"
                    success = rng.random() > config["login_fail_rate"]
                else:
                    resource = rng.choice(user["usual_resources"])
                    success = rng.random() > 0.01
                device = rng.choice(user["devices"])
                lat, lon = _jitter_coords(user["home_city"]["lat"], user["home_city"]["lon"], rng)
                events.append({
                    "timestamp": ts,
                    "user_id": user["user_id"],
                    "entity_type": user["entity_type"],
                    "event_type": event_type,
                    "src_ip": f"{user['home_subnet']}.{rng.randint(2, 254)}",
                    "country": user["home_city"]["country"],
                    "city": user["home_city"]["city"],
                    "lat": lat,
                    "lon": lon,
                    "device_id": device["device_id"],
                    "device_fingerprint": device["device_fingerprint"],
                    "resource": resource,
                    "auth_method": _sample_auth_method(user, rng),
                    "session_duration": _sample_session_duration(user, rng),
                    "command_sequence": _sample_command_sequence(event_type, resource, rng),
                    "success": success,
                    "label": "normal",
                })
    return events


def _num_incidents(config, attack_type):
    target_attack_events = config["total_events"] * config["attack_ratio"]
    target_events = target_attack_events * config["attack_type_weights"][attack_type]
    return max(1, round(target_events / AVG_EVENTS_PER_INCIDENT[attack_type]))


def _inject_brute_force(users, config, rng, fake, start_ts, end_ts, num_incidents):
    events = []
    for _ in range(num_incidents):
        user = rng.choice(users)
        incident_start = _random_timestamp(start_ts, end_ts, rng)
        foreign_cities = [c for c in CITIES if c["country"] != user["home_city"]["country"]]
        attacker_city = rng.choice(foreign_cities)
        attacker_ips = [fake.ipv4() for _ in range(rng.randint(1, 3))]
        fingerprint = hashlib.sha256(f"attacker-{_rand_hex(rng, 12)}".encode()).hexdigest()[:16]
        device_id = f"dev-{fingerprint[:10]}"
        n_attempts = rng.randint(15, 40)
        for i in range(n_attempts):
            ts = incident_start + pd.Timedelta(seconds=rng.randint(0, 1800))
            success = i == n_attempts - 1 and rng.random() < 0.3
            lat, lon = _jitter_coords(attacker_city["lat"], attacker_city["lon"], rng)
            events.append({
                "timestamp": ts,
                "user_id": user["user_id"],
                "entity_type": user["entity_type"],
                "event_type": "login",
                "src_ip": rng.choice(attacker_ips),
                "country": attacker_city["country"],
                "city": attacker_city["city"],
                "lat": lat,
                "lon": lon,
                "device_id": device_id,
                "device_fingerprint": fingerprint,
                "resource": "auth-server",
                "auth_method": "password",
                "session_duration": _sample_session_duration(user, rng) if success else 0.0,
                "command_sequence": "",
                "success": success,
                "label": "brute_force",
            })
    return events


def _inject_impossible_travel(users, config, rng, fake, start_ts, end_ts, num_incidents):
    events = []
    for _ in range(num_incidents):
        user = rng.choice(users)
        t1 = _random_timestamp(start_ts, end_ts - pd.Timedelta(hours=3), rng)
        home = user["home_city"]
        far_cities = [c for c in CITIES if _haversine_km(c["lat"], c["lon"], home["lat"], home["lon"]) > 3000]
        if not far_cities:
            continue
        city2 = rng.choice(far_cities)
        t2 = t1 + pd.Timedelta(minutes=rng.randint(20, 150))
        device1 = rng.choice(user["devices"])
        fingerprint2 = hashlib.sha256(f"travel-{_rand_hex(rng, 12)}".encode()).hexdigest()[:16]
        lat1, lon1 = _jitter_coords(home["lat"], home["lon"], rng)
        lat2, lon2 = _jitter_coords(city2["lat"], city2["lon"], rng)
        events.append({
            "timestamp": t1, "user_id": user["user_id"], "entity_type": user["entity_type"], "event_type": "login",
            "src_ip": f"{user['home_subnet']}.{rng.randint(2, 254)}",
            "country": home["country"], "city": home["city"], "lat": lat1, "lon": lon1,
            "device_id": device1["device_id"], "device_fingerprint": device1["device_fingerprint"],
            "resource": "auth-server", "auth_method": _sample_auth_method(user, rng),
            "session_duration": _sample_session_duration(user, rng), "command_sequence": "",
            "success": True, "label": "impossible_travel",
        })
        events.append({
            "timestamp": t2, "user_id": user["user_id"], "entity_type": user["entity_type"], "event_type": "login",
            "src_ip": fake.ipv4(),
            "country": city2["country"], "city": city2["city"], "lat": lat2, "lon": lon2,
            "device_id": f"dev-{fingerprint2[:10]}", "device_fingerprint": fingerprint2,
            "resource": "auth-server", "auth_method": _sample_auth_method(user, rng),
            "session_duration": _sample_session_duration(user, rng), "command_sequence": "",
            "success": True, "label": "impossible_travel",
        })
    return events


def _inject_lateral_movement(users, config, rng, fake, start_ts, end_ts, num_incidents):
    events = []
    for _ in range(num_incidents):
        user = rng.choice(users)
        incident_start = _random_timestamp(start_ts, end_ts, rng)
        candidates = [r for r in ALL_RESOURCES if r not in user["usual_resources"]]
        n_hosts = min(rng.randint(10, 30), len(candidates))
        hosts = rng.sample(candidates, k=n_hosts)
        device = rng.choice(user["devices"])
        for i, host in enumerate(hosts):
            ts = incident_start + pd.Timedelta(seconds=sum(rng.randint(20, 120) for _ in range(i + 1)))
            lat, lon = _jitter_coords(user["home_city"]["lat"], user["home_city"]["lon"], rng)
            events.append({
                "timestamp": ts, "user_id": user["user_id"], "entity_type": user["entity_type"],
                "event_type": "host_connect",
                "src_ip": f"{user['home_subnet']}.{rng.randint(2, 254)}",
                "country": user["home_city"]["country"], "city": user["home_city"]["city"],
                "lat": lat, "lon": lon,
                "device_id": device["device_id"], "device_fingerprint": device["device_fingerprint"],
                "resource": host, "auth_method": _sample_auth_method(user, rng),
                "session_duration": _sample_session_duration(user, rng),
                "command_sequence": _sample_command_sequence("host_connect", host, rng),
                "success": True, "label": "lateral_movement",
            })
    return events


def _inject_device_spoofing(users, config, rng, fake, start_ts, end_ts, num_incidents):
    events = []
    for _ in range(num_incidents):
        user = rng.choice(users)
        incident_start = _random_timestamp(start_ts, end_ts, rng)
        fingerprint = hashlib.sha256(f"spoof-{_rand_hex(rng, 12)}".encode()).hexdigest()[:16]
        device_id = f"dev-{fingerprint[:10]}"
        n_events = rng.randint(1, 5)
        for i in range(n_events):
            ts = incident_start + pd.Timedelta(minutes=sum(rng.randint(5, 30) for _ in range(i + 1)))
            event_type = "login" if i == 0 else rng.choice(["file_access", "host_connect"])
            resource = "auth-server" if event_type == "login" else rng.choice(user["usual_resources"] + ["vpn-gateway"])
            lat, lon = _jitter_coords(user["home_city"]["lat"], user["home_city"]["lon"], rng, spread=1.5)
            events.append({
                "timestamp": ts, "user_id": user["user_id"], "entity_type": user["entity_type"],
                "event_type": event_type,
                "src_ip": fake.ipv4(),
                "country": user["home_city"]["country"], "city": user["home_city"]["city"],
                "lat": lat, "lon": lon,
                "device_id": device_id, "device_fingerprint": fingerprint,
                "resource": resource, "auth_method": _sample_auth_method(user, rng),
                "session_duration": _sample_session_duration(user, rng),
                "command_sequence": _sample_command_sequence(event_type, resource, rng),
                "success": True, "label": "device_spoofing",
            })
    return events


def _inject_credential_misuse(users, config, rng, fake, start_ts, end_ts, num_incidents):
    events = []
    for _ in range(num_incidents):
        user = rng.choice(users)
        day = start_ts + pd.Timedelta(days=rng.randint(0, config["num_days"] - 1))
        odd_hour = rng.choice([0, 1, 2, 3, 4, 23])
        device = rng.choice(user["devices"])
        sensitive_candidates = [r for r in SENSITIVE_RESOURCES if r not in user["usual_resources"]] or SENSITIVE_RESOURCES
        for _i in range(rng.randint(1, 3)):
            ts = day + pd.Timedelta(hours=odd_hour, minutes=rng.randint(0, 59))
            resource = rng.choice(sensitive_candidates)
            lat, lon = _jitter_coords(user["home_city"]["lat"], user["home_city"]["lon"], rng)
            event_type = rng.choice(["file_access", "host_connect"])
            events.append({
                "timestamp": ts, "user_id": user["user_id"], "entity_type": user["entity_type"],
                "event_type": event_type,
                "src_ip": f"{user['home_subnet']}.{rng.randint(2, 254)}",
                "country": user["home_city"]["country"], "city": user["home_city"]["city"],
                "lat": lat, "lon": lon,
                "device_id": device["device_id"], "device_fingerprint": device["device_fingerprint"],
                "resource": resource, "auth_method": _sample_auth_method(user, rng),
                "session_duration": _sample_session_duration(user, rng),
                "command_sequence": _sample_command_sequence(event_type, resource, rng),
                "success": True, "label": "credential_misuse",
            })
    return events


def _inject_credential_stuffing(users, config, rng, fake, start_ts, end_ts, num_incidents):
    """Many target entity_ids, few shared source_ips/fingerprint, high failure rate."""
    events = []
    for _ in range(num_incidents):
        campaign_start = _random_timestamp(start_ts, end_ts, rng)
        attacker_ips = [fake.ipv4() for _ in range(rng.randint(1, 3))]
        attacker_city = rng.choice(CITIES)
        fingerprint = hashlib.sha256(f"stuffing-{_rand_hex(rng, 12)}".encode()).hexdigest()[:16]
        device_id = f"dev-{fingerprint[:10]}"
        n_targets = min(rng.randint(15, 40), len(users))
        targets = rng.sample(users, k=n_targets)
        campaign_span_minutes = rng.randint(15, 90)
        for target in targets:
            for _i in range(rng.randint(1, 4)):
                ts = campaign_start + pd.Timedelta(minutes=rng.uniform(0, campaign_span_minutes))
                success = rng.random() < 0.04
                lat, lon = _jitter_coords(attacker_city["lat"], attacker_city["lon"], rng)
                events.append({
                    "timestamp": ts, "user_id": target["user_id"], "entity_type": target["entity_type"],
                    "event_type": "login",
                    "src_ip": rng.choice(attacker_ips),
                    "country": attacker_city["country"], "city": attacker_city["city"],
                    "lat": lat, "lon": lon,
                    "device_id": device_id, "device_fingerprint": fingerprint,
                    "resource": "auth-server", "auth_method": "password",
                    "session_duration": _sample_session_duration(target, rng) if success else 0.0,
                    "command_sequence": "",
                    "success": success, "label": "credential_stuffing",
                })
    return events


def _inject_low_and_slow_exfiltration(users, config, rng, fake, start_ts, end_ts, num_incidents):
    """Sparse off-hours accesses to an expanding set of resources over 1-2 weeks, one entity."""
    events = []
    for _ in range(num_incidents):
        user = rng.choice(users)
        span_days = rng.randint(7, 14)
        latest_start = end_ts - pd.Timedelta(days=span_days + 1)
        if latest_start <= start_ts:
            continue
        campaign_start = _random_timestamp(start_ts, latest_start, rng)
        device = rng.choice(user["devices"])
        candidates = [r for r in ALL_RESOURCES if r not in user["usual_resources"]]
        n_targets = min(rng.randint(8, 20), len(candidates))
        target_resources = rng.sample(candidates, k=n_targets)
        for day_offset in range(span_days):
            if rng.random() < 0.55:
                continue
            day = campaign_start + pd.Timedelta(days=day_offset)
            odd_hour = rng.choice([0, 1, 2, 3, 4, 22, 23])
            for _i in range(rng.randint(1, 2)):
                ts = day + pd.Timedelta(hours=odd_hour, minutes=rng.randint(0, 59))
                resource = rng.choice(target_resources)
                event_type = rng.choice(["file_access", "host_connect"])
                lat, lon = _jitter_coords(user["home_city"]["lat"], user["home_city"]["lon"], rng)
                events.append({
                    "timestamp": ts, "user_id": user["user_id"], "entity_type": user["entity_type"],
                    "event_type": event_type,
                    "src_ip": f"{user['home_subnet']}.{rng.randint(2, 254)}",
                    "country": user["home_city"]["country"], "city": user["home_city"]["city"],
                    "lat": lat, "lon": lon,
                    "device_id": device["device_id"], "device_fingerprint": device["device_fingerprint"],
                    "resource": resource, "auth_method": _sample_auth_method(user, rng),
                    "session_duration": _sample_session_duration(user, rng),
                    "command_sequence": _sample_command_sequence(event_type, resource, rng),
                    "success": True, "label": "low_and_slow_exfiltration",
                })
    return events


def _inject_insider_drift(users, config, rng, start_ts, end_ts, num_entities):
    """Legitimate footprint expansion -- labeled "normal", returned separately for tracking."""
    events = []
    drift_records = []
    chosen = rng.sample(users, k=min(num_entities, len(users)))
    for user in chosen:
        span_days = rng.randint(10, 20)
        latest_start = end_ts - pd.Timedelta(days=span_days + 1)
        if latest_start <= start_ts:
            continue
        drift_start = _random_timestamp(start_ts, latest_start, rng)
        device = rng.choice(user["devices"])
        candidates = [r for r in ALL_RESOURCES if r not in user["usual_resources"] and r not in SENSITIVE_RESOURCES]
        if not candidates:
            continue
        n_new = min(rng.randint(3, 8), len(candidates))
        new_resources = rng.sample(candidates, k=n_new)
        entity_events = []
        for day_offset in range(span_days):
            if rng.random() < 0.6:
                continue
            day = drift_start + pd.Timedelta(days=day_offset)
            progress = (day_offset + 1) / span_days
            available = new_resources[: max(1, int(len(new_resources) * progress))]
            for _i in range(rng.randint(1, 3)):
                ts = _sample_timestamp(day, user, rng)
                resource = rng.choice(available)
                event_type = rng.choice(["file_access", "host_connect"])
                lat, lon = _jitter_coords(user["home_city"]["lat"], user["home_city"]["lon"], rng)
                entity_events.append({
                    "timestamp": ts, "user_id": user["user_id"], "entity_type": user["entity_type"],
                    "event_type": event_type,
                    "src_ip": f"{user['home_subnet']}.{rng.randint(2, 254)}",
                    "country": user["home_city"]["country"], "city": user["home_city"]["city"],
                    "lat": lat, "lon": lon,
                    "device_id": device["device_id"], "device_fingerprint": device["device_fingerprint"],
                    "resource": resource, "auth_method": _sample_auth_method(user, rng),
                    "session_duration": _sample_session_duration(user, rng),
                    "command_sequence": _sample_command_sequence(event_type, resource, rng),
                    "success": True, "label": "normal",
                })
        if entity_events:
            events.extend(entity_events)
            drift_records.append({
                "entity_id": user["user_id"],
                "entity_type": user["entity_type"],
                "drift_start": str(drift_start),
                "drift_end": str(drift_start + pd.Timedelta(days=span_days)),
                "resources_added": new_resources,
                "n_drift_events": len(entity_events),
            })
    return events, drift_records


ATTACK_GENERATORS = {
    "brute_force": _inject_brute_force,
    "impossible_travel": _inject_impossible_travel,
    "lateral_movement": _inject_lateral_movement,
    "device_spoofing": _inject_device_spoofing,
    "credential_misuse": _inject_credential_misuse,
    "credential_stuffing": _inject_credential_stuffing,
    "low_and_slow_exfiltration": _inject_low_and_slow_exfiltration,
}


def generate_dataset(config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}

    rng = random.Random(config["seed"])
    np_rng = np.random.default_rng(config["seed"])
    fake = Faker()
    Faker.seed(config["seed"])

    start_ts = pd.Timestamp(config["start_date"])
    end_ts = start_ts + pd.Timedelta(days=config["num_days"])

    users = generate_user_profiles(config, rng)
    events = generate_normal_events(users, config, rng, np_rng, start_ts)

    for attack_type, generator_fn in ATTACK_GENERATORS.items():
        n_incidents = _num_incidents(config, attack_type)
        events.extend(generator_fn(users, config, rng, fake, start_ts, end_ts, n_incidents))

    drift_events, drift_records = _inject_insider_drift(
        users, config, rng, start_ts, end_ts, config["num_drift_entities"]
    )
    events.extend(drift_events)

    df = pd.DataFrame(events)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "event_id", [f"EVT{i:07d}" for i in range(len(df))])

    df = df.rename(columns=config["columns"])
    ordered_cols = [config["columns"].get(c, c) for c in CANONICAL_COLUMN_ORDER]
    df = df[ordered_cols]

    alias = config.get("legacy_user_id_alias")
    if alias:
        entity_col = config["columns"]["user_id"]
        df.insert(df.columns.get_loc(entity_col) + 1, alias, df[entity_col])

    return df, drift_records


def main():
    config = DEFAULT_CONFIG
    df, drift_records = generate_dataset(config)

    out_path = Path(config["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    drift_path = Path(config["drift_entities_path"])
    drift_path.parent.mkdir(parents=True, exist_ok=True)
    with open(drift_path, "w") as f:
        json.dump(drift_records, f, indent=2)

    ts_col = config["columns"]["timestamp"]
    entity_col = config["columns"]["user_id"]
    entity_type_col = config["columns"]["entity_type"]
    label_col = config["columns"]["label"]

    print(f"Wrote {len(df)} events to {out_path}")
    print(f"Date range: {df[ts_col].min()} to {df[ts_col].max()}")
    print(f"Unique entities: {df[entity_col].nunique()}")
    print("\nEntities per type:")
    print(df.drop_duplicates(entity_col)[entity_type_col].value_counts())
    print("\nEvents per label:")
    print(df[label_col].value_counts())
    print(f"\nInsider drift entities tracked: {len(drift_records)} (saved to {drift_path})")


if __name__ == "__main__":
    main()
