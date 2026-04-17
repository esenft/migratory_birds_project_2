import duckdb

PARQUET_GLOB = "data/parquet/ebird_partitioned/**/*.parquet"

con = duckdb.connect()

print("\nDistinct species sample:")
species_df = con.execute(f"""
    SELECT species, COUNT(*) AS n
    FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
    WHERE species IS NOT NULL
    GROUP BY species
    ORDER BY n DESC
    LIMIT 50
""").fetchdf()
print(species_df)

print("\nDistinct stateProvince sample:")
states_df = con.execute(f"""
    SELECT stateProvince, COUNT(*) AS n
    FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
    WHERE stateProvince IS NOT NULL
    GROUP BY stateProvince
    ORDER BY n DESC
    LIMIT 50
""").fetchdf()
print(states_df)

print("\nCounts by month:")
months_df = con.execute(f"""
    SELECT month, COUNT(*) AS n
    FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
    GROUP BY month
    ORDER BY month
""").fetchdf()
print(months_df)