import duckdb

CSV_PATH = "data/raw/ebird_data.csv"

con = duckdb.connect()

print("Sample rows:")
print(con.execute(f"""
    SELECT *
    FROM read_csv_auto('{CSV_PATH}')
    LIMIT 5
""").fetchdf())

print("\nSchema:")
print(con.execute(f"""
    DESCRIBE
    SELECT *
    FROM read_csv_auto('{CSV_PATH}')
""").fetchdf())