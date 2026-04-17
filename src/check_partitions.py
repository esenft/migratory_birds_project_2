import duckdb

PARQUET_GLOB = "data/parquet/ebird_partitioned/**/*.parquet"

con = duckdb.connect()

df = con.execute(f"""
    SELECT year, month, count(*) AS n_rows
    FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
    GROUP BY 1, 2
    ORDER BY 1, 2
""").fetchdf()

print(df)