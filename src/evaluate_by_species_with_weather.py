import duckdb
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

PANEL_PATH = "data/panel/species_state_week_panel_with_weather.parquet"
MODEL_PATH = "models/present_rel_10pct_peak_logreg_with_weather.joblib"

TARGET_COL = "present_rel_10pct_peak"
TEST_YEAR = 2024

con = duckdb.connect()

query = f"""
    SELECT *
    FROM read_parquet('{PANEL_PATH}')
    ORDER BY species, stateProvince, year, week_of_year
"""

df = con.execute(query).fetchdf()

if df.empty:
    raise ValueError("Weather-merged panel dataframe is empty.")

# Sort for lag feature creation
df = df.sort_values(["species", "stateProvince", "year", "week_of_year"]).reset_index(drop=True)

group_cols = ["species", "stateProvince"]

# Rebuild bird-count lag features exactly as in training
df["lag_1_obs_count"] = df.groupby(group_cols)["obs_count"].shift(1)
df["lag_2_obs_count"] = df.groupby(group_cols)["obs_count"].shift(2)
df["lag_3_obs_count"] = df.groupby(group_cols)["obs_count"].shift(3)

df["rolling_2wk_mean_obs_count"] = (
    df.groupby(group_cols)["obs_count"]
      .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
)

df["rolling_3wk_mean_obs_count"] = (
    df.groupby(group_cols)["obs_count"]
      .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
)

df["rolling_4wk_max_obs_count"] = (
    df.groupby(group_cols)["obs_count"]
      .transform(lambda s: s.shift(1).rolling(4, min_periods=1).max())
)

# Seasonal features
df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)
df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)

df["had_recent_activity"] = (
    (df["lag_1_obs_count"].fillna(0) > 0) |
    (df["lag_2_obs_count"].fillna(0) > 0)
).astype(int)

feature_cols = [
    "species",
    "stateProvince",
    "week_of_year",
    "lag_1_obs_count",
    "lag_2_obs_count",
    "lag_3_obs_count",
    "rolling_2wk_mean_obs_count",
    "rolling_3wk_mean_obs_count",
    "rolling_4wk_max_obs_count",
    "week_sin",
    "week_cos",
    "had_recent_activity",
    "temp_mean_7d",
    "temp_max_7d",
    "temp_min_7d",
    "precip_sum_7d",
    "rain_sum_7d",
    "snowfall_sum_7d",
    "wind_max_7d",
    "lag_1_temp_mean_7d",
    "lag_1_temp_max_7d",
    "lag_1_temp_min_7d",
    "lag_1_precip_sum_7d",
    "lag_1_rain_sum_7d",
    "lag_1_snowfall_sum_7d",
    "lag_1_wind_max_7d",
]

test_df = df[df["year"] == TEST_YEAR].copy()

if test_df.empty:
    raise ValueError(f"No rows found for TEST_YEAR={TEST_YEAR}")

X_test = test_df[feature_cols]
y_test = test_df[TARGET_COL].astype(int)

model = joblib.load(MODEL_PATH)

test_df["y_true"] = y_test
test_df["y_pred"] = model.predict(X_test)
test_df["y_proba"] = model.predict_proba(X_test)[:, 1]

print("=== OVERALL TEST METRICS ===")
print("Accuracy:", round(accuracy_score(test_df["y_true"], test_df["y_pred"]), 4))
print("Precision:", round(precision_score(test_df["y_true"], test_df["y_pred"]), 4))
print("Recall:", round(recall_score(test_df["y_true"], test_df["y_pred"]), 4))
print("F1:", round(f1_score(test_df["y_true"], test_df["y_pred"]), 4))
print("ROC-AUC:", round(roc_auc_score(test_df["y_true"], test_df["y_proba"]), 4))
print()

rows = []

for species, group in test_df.groupby("species"):
    y_true_sp = group["y_true"]
    y_pred_sp = group["y_pred"]
    y_proba_sp = group["y_proba"]

    try:
        auc = roc_auc_score(y_true_sp, y_proba_sp)
    except ValueError:
        auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true_sp, y_pred_sp, labels=[0, 1]).ravel()

    rows.append({
        "species": species,
        "n_rows": len(group),
        "positive_rate": y_true_sp.mean(),
        "accuracy": accuracy_score(y_true_sp, y_pred_sp),
        "precision": precision_score(y_true_sp, y_pred_sp, zero_division=0),
        "recall": recall_score(y_true_sp, y_pred_sp, zero_division=0),
        "f1": f1_score(y_true_sp, y_pred_sp, zero_division=0),
        "roc_auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })

results = pd.DataFrame(rows).sort_values("f1", ascending=False)

print("=== TEST METRICS BY SPECIES ===")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()

print("=== ERROR COUNTS BY SPECIES ===")
error_summary = (
    results[["species", "fp", "fn"]]
    .assign(total_errors=lambda d: d["fp"] + d["fn"])
    .sort_values("total_errors", ascending=False)
)
print(error_summary.to_string(index=False))