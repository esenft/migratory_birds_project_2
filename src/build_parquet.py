import duckdb
from pathlib import Path

CSV_PATH = "data/raw/ebird_data.csv"
OUT_DIR = "data/parquet/ebird_partitioned"

KEEP_COLUMNS = [
    "eventDate",
    "year",
    "month",
    "day",
    "countryCode",
    "stateProvince",
    "locality",
    "decimalLatitude",
    "decimalLongitude",
    "kingdom",
    "phylum",
    "class",
    '"order"',
    "family",
    "genus",
    "species",
    "taxonRank",
    "scientificName",
    "occurrenceStatus",
    "individualCount",
    "taxonKey",
    "speciesKey"
]

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

select_cols = ",\n            ".join(KEEP_COLUMNS)

query = f"""
COPY (
    SELECT
        {select_cols}
    FROM read_csv_auto('{CSV_PATH}')
    WHERE eventDate IS NOT NULL
)
TO '{OUT_DIR}'
(
    FORMAT parquet,
    PARTITION_BY (year, month),
    OVERWRITE_OR_IGNORE
);
"""

con.execute(query)
print(f"Wrote partitioned parquet to {OUT_DIR}")