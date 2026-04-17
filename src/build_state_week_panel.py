import duckdb
from pathlib import Path

PARQUET_GLOB = "data/parquet/ebird_partitioned/**/*.parquet"
OUT_DIR = Path("data/panel")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "species_state_week_panel.parquet"

# Replace with your actual 4 bird species
TARGET_SPECIES = [
    "Archilochus colubris",   # Ruby-throated Hummingbird
    "Setophaga petechia",     # Yellow Warbler
    "Icterus galbula",        # Baltimore Oriole
    "Agelaius phoeniceus",    # Red-winged Blackbird
]

# Focus corridor for migration toward New England
TARGET_STATES = [
    "Florida",
    "Georgia",
    "South Carolina",
    "North Carolina",
    "Virginia",
    "Maryland",
    "Delaware",
    "New Jersey",
    "Pennsylvania",
    "New York",
    "Connecticut",
    "Rhode Island",
    "Massachusetts",
    "Vermont",
    "New Hampshire",
    "Maine",
]

# Spring window
SPRING_MONTHS = [3, 4, 5, 6]

con = duckdb.connect()

species_sql = ", ".join(f"'{s}'" for s in TARGET_SPECIES)
states_sql = ", ".join(f"'{s}'" for s in TARGET_STATES)
months_sql = ", ".join(str(m) for m in SPRING_MONTHS)

query = f"""
COPY (
    WITH filtered AS (
        SELECT
            species,
            stateProvince,
            eventDate,
            year,
            month,
            week(eventDate) AS week_of_year
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE species IN ({species_sql})
          AND stateProvince IN ({states_sql})
          AND month IN ({months_sql})
          AND eventDate IS NOT NULL
    ),

    obs_counts AS (
        SELECT
            species,
            stateProvince,
            year,
            week_of_year,
            MIN(eventDate) AS first_event_date_in_week,
            COUNT(*) AS obs_count
        FROM filtered
        GROUP BY 1, 2, 3, 4
    ),

    all_species AS (
        SELECT DISTINCT species FROM filtered
    ),

    all_states AS (
        SELECT DISTINCT stateProvince FROM filtered
    ),

    all_years AS (
        SELECT DISTINCT year FROM filtered
    ),

    all_weeks AS (
        SELECT DISTINCT week(eventDate) AS week_of_year
        FROM filtered
    ),

    full_grid AS (
        SELECT
            s.species,
            st.stateProvince,
            y.year,
            w.week_of_year
        FROM all_species s
        CROSS JOIN all_states st
        CROSS JOIN all_years y
        CROSS JOIN all_weeks w
    ),

    panel AS (
        SELECT
            g.species,
            g.stateProvince,
            g.year,
            g.week_of_year,
            COALESCE(o.obs_count, 0) AS obs_count,
            o.first_event_date_in_week
        FROM full_grid g
        LEFT JOIN obs_counts o
          ON g.species = o.species
         AND g.stateProvince = o.stateProvince
         AND g.year = o.year
         AND g.week_of_year = o.week_of_year
    ),

    add_peak AS (
        SELECT
            *,
            MAX(obs_count) OVER (
                PARTITION BY species, stateProvince, year
            ) AS yearly_peak_count
        FROM panel
    ),

    labeled AS (
        SELECT
            *,
            CASE WHEN obs_count >= 1 THEN 1 ELSE 0 END AS present_ge_1,
            CASE WHEN obs_count >= 5 THEN 1 ELSE 0 END AS present_ge_5,
            CASE
                WHEN yearly_peak_count > 0
                 AND obs_count >= 0.10 * yearly_peak_count
                THEN 1 ELSE 0
            END AS present_rel_10pct_peak
        FROM add_peak
    )

    SELECT *
    FROM labeled
    ORDER BY species, stateProvince, year, week_of_year
)
TO '{OUT_PATH}'
(FORMAT parquet, OVERWRITE_OR_IGNORE);
"""

con.execute(query)
print(f"Wrote panel dataset to {OUT_PATH}")