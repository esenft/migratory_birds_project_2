from __future__ import annotations

import time
from typing import Dict, Tuple

import pandas as pd
import requests

from src.utils.state_config import STATE_COORDS

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
]


def get_state_coordinates(state: str) -> Tuple[float, float]:
    if state not in STATE_COORDS:
        raise ValueError(
            f"Unsupported state '{state}'. Supported states: {list(STATE_COORDS.keys())}"
        )
    return STATE_COORDS[state]


def fetch_recent_daily_weather(
    state: str,
    past_days: int = 14,
    max_retries: int = 4,
    timeout_seconds: int = 45,
) -> pd.DataFrame:
    """
    Fetch recent daily weather for a state centroid from Open-Meteo.

    Uses retries with exponential backoff because public APIs can occasionally
    timeout or rate limit.
    """
    lat, lon = get_state_coordinates(state)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "timezone": "America/New_York",
        "past_days": past_days,
        "forecast_days": 1,
    }

    backoff = 3

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Fetching weather for {state} (attempt {attempt}/{max_retries})...")

            response = requests.get(
                FORECAST_API_URL,
                params=params,
                timeout=timeout_seconds,
            )

            if response.status_code == 429:
                print(f"Rate limited by Open-Meteo for {state}. Sleeping {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue

            response.raise_for_status()
            data = response.json()

            if "daily" not in data:
                raise ValueError(f"No daily weather returned for {state}: {data}")

            daily = pd.DataFrame(data["daily"])

            if daily.empty:
                raise ValueError(f"Empty weather data returned for {state}")

            required_cols = [
                "time",
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
                "wind_speed_10m_max",
            ]
            missing = [col for col in required_cols if col not in daily.columns]
            if missing:
                raise ValueError(
                    f"Missing expected weather columns for {state}: {missing}"
                )

            daily["date"] = pd.to_datetime(daily["time"])
            daily["stateProvince"] = state

            return daily

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                print(
                    f"Weather request failed for {state}: {type(e).__name__}. "
                    f"Retrying in {backoff}s..."
                )
                time.sleep(backoff)
                backoff *= 2
            else:
                break

        except requests.exceptions.RequestException as e:
            # Includes non-timeout HTTP errors after raise_for_status
            raise RuntimeError(f"Open-Meteo request failed for {state}: {e}") from e

        except Exception as e:
            raise RuntimeError(f"Unexpected weather parsing error for {state}: {e}") from e

    raise RuntimeError(
        f"Failed to fetch weather for {state} after {max_retries} attempts"
    ) from last_error


def summarize_weather_features(daily_df: pd.DataFrame) -> Dict[str, float]:
    """
    Convert recent daily weather into the same feature names used by the model.

    - current 7-day window = most recent 7 days
    - lag_1 window = the prior 7 days
    """
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    if len(daily_df) < 14:
        raise ValueError(
            f"Need at least 14 days of weather data to build current + lagged weekly features. "
            f"Only found {len(daily_df)} rows."
        )

    current_week = daily_df.iloc[-7:]
    prior_week = daily_df.iloc[-14:-7]

    features = {
        "temp_mean_7d": float(current_week["temperature_2m_mean"].mean()),
        "temp_max_7d": float(current_week["temperature_2m_max"].max()),
        "temp_min_7d": float(current_week["temperature_2m_min"].min()),
        "precip_sum_7d": float(current_week["precipitation_sum"].sum()),
        "rain_sum_7d": float(current_week["rain_sum"].sum()),
        "snowfall_sum_7d": float(current_week["snowfall_sum"].sum()),
        "wind_max_7d": float(current_week["wind_speed_10m_max"].max()),
        "lag_1_temp_mean_7d": float(prior_week["temperature_2m_mean"].mean()),
        "lag_1_temp_max_7d": float(prior_week["temperature_2m_max"].max()),
        "lag_1_temp_min_7d": float(prior_week["temperature_2m_min"].min()),
        "lag_1_precip_sum_7d": float(prior_week["precipitation_sum"].sum()),
        "lag_1_rain_sum_7d": float(prior_week["rain_sum"].sum()),
        "lag_1_snowfall_sum_7d": float(prior_week["snowfall_sum"].sum()),
        "lag_1_wind_max_7d": float(prior_week["wind_speed_10m_max"].max()),
    }

    return features


def get_weather_features_for_state(state: str) -> Dict[str, float]:
    daily_df = fetch_recent_daily_weather(state=state, past_days=14)
    return summarize_weather_features(daily_df)


if __name__ == "__main__":
    test_state = "Massachusetts"
    weather_features = get_weather_features_for_state(test_state)

    print(f"Weather features for {test_state}:")
    for key, value in weather_features.items():
        print(f"{key}: {value}")