import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

PANEL_PATH = "data/panel/species_state_week_panel_with_weather.parquet"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "present_rel_10pct_peak"

TRAIN_END_YEAR = 2022
VALID_YEAR = 2023
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

df = df.sort_values(["species", "stateProvince", "year", "week_of_year"]).reset_index(drop=True)

group_cols = ["species", "stateProvince"]

# Bird-count lag features
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

train_df = df[df["year"] <= TRAIN_END_YEAR].copy()
valid_df = df[df["year"] == VALID_YEAR].copy()
test_df = df[df["year"] == TEST_YEAR].copy()

if train_df.empty or valid_df.empty or test_df.empty:
    raise ValueError(
        f"One of the splits is empty. "
        f"Train rows={len(train_df)}, Valid rows={len(valid_df)}, Test rows={len(test_df)}"
    )

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

categorical_cols = ["species", "stateProvince"]
numeric_cols = [
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

X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL].astype(int)

X_valid = valid_df[feature_cols]
y_valid = valid_df[TARGET_COL].astype(int)

X_test = test_df[feature_cols]
y_test = test_df[TARGET_COL].astype(int)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

print(f"Training rows: {len(train_df)}")
print(f"Validation rows: {len(valid_df)}")
print(f"Test rows: {len(test_df)}")
print(f"Target column: {TARGET_COL}")
print()

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)
valid_proba = model.predict_proba(X_valid)[:, 1]

print("=== VALIDATION ===")
print("Accuracy:", accuracy_score(y_valid, valid_pred))
print("ROC-AUC:", roc_auc_score(y_valid, valid_proba))
print("Confusion Matrix:")
print(confusion_matrix(y_valid, valid_pred))
print("\nClassification Report:")
print(classification_report(y_valid, valid_pred, digits=4))
print()

test_pred = model.predict(X_test)
test_proba = model.predict_proba(X_test)[:, 1]

print("=== TEST ===")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("ROC-AUC:", roc_auc_score(y_test, test_proba))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nClassification Report:")
print(classification_report(y_test, test_pred, digits=4))

model_path = MODEL_DIR / f"{TARGET_COL}_logreg_with_weather.joblib"
joblib.dump(model, model_path)

print()
print(f"Saved model to {model_path}")