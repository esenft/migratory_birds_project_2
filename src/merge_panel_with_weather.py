import duckdb
from pathlib import Path

PANEL_PATH = "data/panel/species_state_week_panel.parquet"
WEATHER_PATH = "data/weather/state_week_weather.parquet"

OUT_DIR = Path("data/panel")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "species_state_week_panel_with_weather.parquet"

con = duckdb.connect()

query = f"""
COPY (
    SELECT
        p.*,
        w.temp_mean_7d,
        w.temp_max_7d,
        w.temp_min_7d,
        w.precip_sum_7d,
        w.rain_sum_7d,
        w.snowfall_sum_7d,
        w.wind_max_7d,
        w.lag_1_temp_mean_7d,
        w.lag_1_temp_max_7d,
        w.lag_1_temp_min_7d,
        w.lag_1_precip_sum_7d,
        w.lag_1_rain_sum_7d,
        w.lag_1_snowfall_sum_7d,
        w.lag_1_wind_max_7d
    FROM read_parquet('{PANEL_PATH}') p
    LEFT JOIN read_parquet('{WEATHER_PATH}') w
      ON p.stateProvince = w.stateProvince
     AND p.year = w.year
     AND p.week_of_year = w.week_of_year
)
TO '{OUT_PATH}'
(FORMAT parquet, OVERWRITE_OR_IGNORE);
"""

con.execute(query)

count = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{OUT_PATH}')
""").fetchone()[0]

missing_weather = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{OUT_PATH}')
    WHERE temp_mean_7d IS NULL
""").fetchone()[0]

print(f"Wrote merged panel to {OUT_PATH}")
print(f"Rows: {count}")
print(f"Rows missing weather: {missing_weather}")