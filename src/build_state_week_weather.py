import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path("data/weather")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "state_week_weather.parquet"

# Approximate state centroids for the East Coast / New England corridor
STATE_COORDS = {
    "Florida": (27.6648, -81.5158),
    "Georgia": (32.1656, -82.9001),
    "South Carolina": (33.8361, -81.1637),
    "North Carolina": (35.7596, -79.0193),
    "Virginia": (37.4316, -78.6569),
    "Maryland": (39.0458, -76.6413),
    "Delaware": (38.9108, -75.5277),
    "New Jersey": (40.0583, -74.4057),
    "Pennsylvania": (41.2033, -77.1945),
    "New York": (42.9538, -75.5268),
    "Connecticut": (41.6032, -73.0877),
    "Rhode Island": (41.5801, -71.4774),
    "Massachusetts": (42.4072, -71.3824),
    "Vermont": (44.5588, -72.5778),
    "New Hampshire": (43.1939, -71.5724),
    "Maine": (45.2538, -69.4455),
}

START_DATE = "2016-03-01"
END_DATE = "2024-06-30"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
]


def fetch_state_weather(state: str, lat: float, lon: float) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "America/New_York",
        "models": "era5",
    }

    max_retries = 5
    backoff = 5  # seconds

    for attempt in range(max_retries):
        try:
            print(f"Fetching {state} (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(url, params=params, timeout=60)

            if response.status_code == 429:
                print(f"Rate limited for {state}. Sleeping {backoff} seconds before retrying...")
                time.sleep(backoff)
                backoff *= 2
                continue

            response.raise_for_status()
            data = response.json()

            if "daily" not in data:
                raise ValueError(f"No 'daily' weather data returned for {state}: {data}")

            daily = pd.DataFrame(data["daily"])

            if daily.empty:
                raise ValueError(f"Empty daily weather data returned for {state}")

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
                raise ValueError(f"Missing expected columns for {state}: {missing}")

            daily["date"] = pd.to_datetime(daily["time"])
            daily["stateProvince"] = state
            daily["year"] = daily["date"].dt.year
            daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)

            # Short pause even after success to avoid hammering the API
            time.sleep(2)

            return daily

        except Exception as e:
            print(f"Error fetching {state}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying {state} after {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise RuntimeError(f"Failed to fetch weather for {state} after {max_retries} attempts") from e


def main():
    all_daily = []

    for state, (lat, lon) in STATE_COORDS.items():
        df_state = fetch_state_weather(state, lat, lon)
        all_daily.append(df_state)

        # Extra delay between states
        time.sleep(3)

    daily_df = pd.concat(all_daily, ignore_index=True)

    # Aggregate daily weather to state-week
    weekly_df = (
        daily_df.groupby(["stateProvince", "year", "week_of_year"], as_index=False)
        .agg(
            temp_mean_7d=("temperature_2m_mean", "mean"),
            temp_max_7d=("temperature_2m_max", "max"),
            temp_min_7d=("temperature_2m_min", "min"),
            precip_sum_7d=("precipitation_sum", "sum"),
            rain_sum_7d=("rain_sum", "sum"),
            snowfall_sum_7d=("snowfall_sum", "sum"),
            wind_max_7d=("wind_speed_10m_max", "max"),
            n_days=("date", "count"),
        )
    )

    # Sort so lag features are built correctly
    weekly_df = weekly_df.sort_values(
        ["stateProvince", "year", "week_of_year"]
    ).reset_index(drop=True)

    # Create lagged weekly weather features
    weather_cols = [
        "temp_mean_7d",
        "temp_max_7d",
        "temp_min_7d",
        "precip_sum_7d",
        "rain_sum_7d",
        "snowfall_sum_7d",
        "wind_max_7d",
    ]

    for col in weather_cols:
        weekly_df[f"lag_1_{col}"] = weekly_df.groupby("stateProvince")[col].shift(1)

    weekly_df.to_parquet(OUT_PATH, index=False)

    print(f"\nWrote weather table to {OUT_PATH}")
    print(f"Rows: {len(weekly_df)}")
    print("\nSample:")
    print(weekly_df.head())


if __name__ == "__main__":
    main()