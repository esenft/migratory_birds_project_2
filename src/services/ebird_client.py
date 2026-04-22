from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import requests

from src.utils.state_config import STATE_CONFIG

EBIRD_API_BASE = "https://api.ebird.org/v2/data/obs"

EBIRD_API_KEY = os.environ.get("EBIRD_API_KEY")
if not EBIRD_API_KEY:
    raise ValueError("Missing EBIRD_API_KEY in environment variables")

HEADERS = {
    "X-eBirdApiToken": EBIRD_API_KEY
}


def get_region_code(state: str) -> str:
    if state not in STATE_CONFIG:
        raise ValueError(f"Unsupported state: {state}")
    return STATE_CONFIG[state]["ebird_region"]


def fetch_recent_observations_for_region(region_code: str, back_days: int = 30) -> List[dict]:
    """
    Fetch recent observations for a region from eBird.
    """
    url = f"{EBIRD_API_BASE}/{region_code}/recent"
    params = {
        "back": back_days,
        "maxResults": 10000,
    }

    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_observation_count(value) -> int:
    """
    eBird obs count may be numeric, missing, or 'X'.
    Convert unknowns to 1 as a minimal presence signal.
    """
    if value is None:
        return 1
    if isinstance(value, str):
        value = value.strip()
        if value.upper() == "X" or value == "":
            return 1
        try:
            return int(float(value))
        except ValueError:
            return 1
    try:
        return int(value)
    except Exception:
        return 1


def to_dataframe(obs_json: List[dict]) -> pd.DataFrame:
    if not obs_json:
        return pd.DataFrame()

    df = pd.DataFrame(obs_json)

    if df.empty:
        return df

    if "obsDt" in df.columns:
        # Normalize to timezone-naive datetimes so comparisons are consistent
        dt = pd.to_datetime(df["obsDt"], errors="coerce")
        try:
            df["obsDt"] = dt.dt.tz_localize(None)
        except TypeError:
            # Already naive
            df["obsDt"] = dt

    if "howMany" in df.columns:
        df["obs_count"] = df["howMany"].apply(normalize_observation_count)
    else:
        df["obs_count"] = 1

    return df


def filter_species(df: pd.DataFrame, common_name: str) -> pd.DataFrame:
    """
    Filter to a target bird using eBird common name.
    """
    if df.empty:
        return df

    if "comName" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    return df[df["comName"] == common_name].copy()


def window_count(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> int:
    if df.empty or "obsDt" not in df.columns:
        return 0

    mask = (df["obsDt"] >= start_dt) & (df["obsDt"] < end_dt)
    return int(df.loc[mask, "obs_count"].sum())


def get_target_state_recent_count_features(state: str, common_name: str) -> Dict[str, float]:
    """
    Approximate weekly lag features for the target state:
    - lag_1_obs_count: last 7 days
    - lag_2_obs_count: 8-14 days ago
    - lag_3_obs_count: 15-21 days ago
    """
    region_code = get_region_code(state)
    obs_json = fetch_recent_observations_for_region(region_code, back_days=30)
    df = to_dataframe(obs_json)
    df = filter_species(df, common_name)

    now = datetime.utcnow()
    w1_start = now - timedelta(days=7)
    w2_start = now - timedelta(days=14)
    w3_start = now - timedelta(days=21)
    w4_start = now - timedelta(days=28)

    lag_1 = window_count(df, w1_start, now)
    lag_2 = window_count(df, w2_start, w1_start)
    lag_3 = window_count(df, w3_start, w2_start)
    lag_4 = window_count(df, w4_start, w3_start)

    rolling_2wk_mean = (lag_1 + lag_2) / 2.0
    rolling_3wk_mean = (lag_1 + lag_2 + lag_3) / 3.0
    rolling_4wk_max = max(lag_1, lag_2, lag_3, lag_4)

    return {
        "lag_1_obs_count": float(lag_1),
        "lag_2_obs_count": float(lag_2),
        "lag_3_obs_count": float(lag_3),
        "rolling_2wk_mean_obs_count": float(rolling_2wk_mean),
        "rolling_3wk_mean_obs_count": float(rolling_3wk_mean),
        "rolling_4wk_max_obs_count": float(rolling_4wk_max),
        "had_recent_activity": int((lag_1 > 0) or (lag_2 > 0)),
    }


def get_southern_corridor_signal(state: str, common_name: str) -> Dict[str, float]:
    """
    Build corridor-level context from states south of the target.
    Not part of the current trained model, but useful for UI/context
    and future model upgrades.
    """
    if state not in STATE_CONFIG:
        raise ValueError(f"Unsupported state: {state}")

    south_states = STATE_CONFIG[state]["south_states"]

    total_last_7d = 0
    total_prev_7d = 0

    now = datetime.utcnow()
    w1_start = now - timedelta(days=7)
    w2_start = now - timedelta(days=14)

    for south_state in south_states:
        region_code = get_region_code(south_state)
        obs_json = fetch_recent_observations_for_region(region_code, back_days=14)
        df = to_dataframe(obs_json)
        df = filter_species(df, common_name)

        total_last_7d += window_count(df, w1_start, now)
        total_prev_7d += window_count(df, w2_start, w1_start)

    return {
        "south_corridor_obs_last_7d": float(total_last_7d),
        "south_corridor_obs_prev_7d": float(total_prev_7d),
    }


if __name__ == "__main__":
    test_state = "Massachusetts"
    test_bird = "Ruby-throated Hummingbird"

    target_features = get_target_state_recent_count_features(test_state, test_bird)
    corridor_features = get_southern_corridor_signal(test_state, test_bird)

    print(f"Recent eBird features for {test_bird} in {test_state}:")
    for key, value in target_features.items():
        print(f"{key}: {value}")

    print("\nSouthern corridor signal:")
    for key, value in corridor_features.items():
        print(f"{key}: {value}")