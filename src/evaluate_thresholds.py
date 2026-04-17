import duckdb

PANEL_PATH = "data/panel/species_state_week_panel.parquet"

con = duckdb.connect()

print("\nOverall label balance:")
overall = con.execute(f"""
    SELECT
        SUM(present_ge_1) AS n_present_ge_1,
        AVG(present_ge_1) AS pct_present_ge_1,
        SUM(present_ge_5) AS n_present_ge_5,
        AVG(present_ge_5) AS pct_present_ge_5,
        SUM(present_rel_10pct_peak) AS n_present_rel_10pct_peak,
        AVG(present_rel_10pct_peak) AS pct_present_rel_10pct_peak,
        COUNT(*) AS total_rows
    FROM read_parquet('{PANEL_PATH}')
""").fetchdf()
print(overall)

print("\nBy species:")
by_species = con.execute(f"""
    SELECT
        species,
        COUNT(*) AS total_rows,
        AVG(present_ge_1) AS pct_present_ge_1,
        AVG(present_ge_5) AS pct_present_ge_5,
        AVG(present_rel_10pct_peak) AS pct_present_rel_10pct_peak
    FROM read_parquet('{PANEL_PATH}')
    GROUP BY species
    ORDER BY species
""").fetchdf()
print(by_species)

print("\nBy state:")
by_state = con.execute(f"""
    SELECT
        stateProvince,
        COUNT(*) AS total_rows,
        AVG(present_ge_1) AS pct_present_ge_1,
        AVG(present_ge_5) AS pct_present_ge_5,
        AVG(present_rel_10pct_peak) AS pct_present_rel_10pct_peak
    FROM read_parquet('{PANEL_PATH}')
    GROUP BY stateProvince
    ORDER BY stateProvince
""").fetchdf()
print(by_state)