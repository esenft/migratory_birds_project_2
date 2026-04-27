from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from src.services.weather_client import get_weather_features_for_state
from src.services.ebird_client import (
    get_target_state_recent_count_features,
    get_southern_corridor_signal,
)
from src.services.species_map import COMMON_TO_SCIENTIFIC

MODEL_PATH = Path("models/present_rel_10pct_peak_logreg_with_weather.joblib")

def interpret_probability(prob: float) -> str:
    if prob < 0.30:
        return "Low likelihood"
    elif prob < 0.60:
        return "Emerging signal"
    elif prob < 0.80:
        return "Likely present"
    return "Strong presence signal"

def estimate_arrival_window(
    common_name: str,
    state: str,
    probability_present: float,
    corridor_last_7d: float,
    local_last_7d: float,
    week_of_year: int,
) -> str:
    if probability_present >= 0.5:
        return "Likely present now"

    if probability_present >= 0.4:
        return "Likely within 1–2 weeks"

    if probability_present >= 0.3:
        if corridor_last_7d > 0 or local_last_7d > 0:
            return "Possible arrival within 2–3 weeks"
        return "No strong arrival signal yet"

    if corridor_last_7d > 0:
        return "Possible arrival within 3–4 weeks"

    return "No strong arrival signal yet"

NORTHERN_STATES = {"Maine", "New Hampshire", "Vermont"}

def apply_migration_sanity_check(
    state: str,
    common_name: str,
    probability_present: float,
    local_last_7d: float,
    corridor_last_7d: float,
) -> dict:
    warning = None
    adjusted_label = None

    # For far-northern states, avoid overconfident "present" calls
    # based only on weak local signal.
    if state in NORTHERN_STATES and probability_present >= 0.8:
        if local_last_7d < 5:
            warning = (
                "Caution: this is a northern-state prediction with limited recent local activity. "
                "This may indicate early reports rather than broad statewide presence."
            )
            adjusted_label = "Emerging signal"

    return {
        "migration_warning": warning,
        "adjusted_interpretation": adjusted_label,
    }

def build_model_features(state: str, common_name: str) -> Dict[str, Any]:
    """
    Build a single feature dictionary matching the trained model schema.
    """
    if common_name not in COMMON_TO_SCIENTIFIC:
        raise ValueError(
            f"Unsupported bird '{common_name}'. Supported birds: {list(COMMON_TO_SCIENTIFIC.keys())}"
        )

    scientific_name = COMMON_TO_SCIENTIFIC[common_name]

    weather_features = get_weather_features_for_state(state)
    ebird_features = get_target_state_recent_count_features(state, common_name)
    corridor_features = get_southern_corridor_signal(state, common_name)

    now = datetime.now()
    week_of_year = int(now.isocalendar().week)

    # Match the same feature names the model was trained on
    model_features = {
        "species": scientific_name,
        "stateProvince": state,
        "week_of_year": week_of_year,
        **ebird_features,
        "week_sin": float(__import__("math").sin(2 * __import__("math").pi * week_of_year / 52.0)),
        "week_cos": float(__import__("math").cos(2 * __import__("math").pi * week_of_year / 52.0)),
        **weather_features,
    }

    return {
        "model_features": model_features,
        "corridor_features": corridor_features,
        "scientific_name": scientific_name,
    }


def predict_presence(state: str, common_name: str) -> Dict[str, Any]:
    """
    Run real-time prediction using the trained weather-augmented model.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Make sure the weather model has been trained and saved."
        )

    model = joblib.load(MODEL_PATH)

    built = build_model_features(state, common_name)
    model_features = built["model_features"]

    X = pd.DataFrame([model_features])

    probability_present = float(model.predict_proba(X)[0][1])
    prediction = int(model.predict(X)[0])
    label = "Present" if prediction == 1 else "Not Present"
    interpretation = interpret_probability(probability_present)

    sanity_check = apply_migration_sanity_check(
        state=state,
        common_name=common_name,
        probability_present=probability_present,
        local_last_7d=model_features["lag_1_obs_count"],
        corridor_last_7d=built["corridor_features"]["south_corridor_obs_last_7d"],
    )

    arrival_estimate = estimate_arrival_window(
    common_name=common_name,
    state=state,
    probability_present=probability_present,
    corridor_last_7d=built["corridor_features"]["south_corridor_obs_last_7d"],
    local_last_7d=model_features["lag_1_obs_count"],
    week_of_year=model_features["week_of_year"],
)

    return {
    "bird_common_name": common_name,
    "bird_scientific_name": built["scientific_name"],
    "state": state,
    "prediction": prediction,
    "prediction_label": label,
    "probability_present": probability_present,
    "interpretation": interpretation,  
    "arrival_estimate": arrival_estimate,  
    "model_features": model_features,
    "corridor_features": built["corridor_features"],
    "migration_warning": sanity_check["migration_warning"],
    "adjusted_interpretation": sanity_check["adjusted_interpretation"],
}


if __name__ == "__main__":
    test_state = "Massachusetts"
    test_bird = "Ruby-throated Hummingbird"

    result = predict_presence(test_state, test_bird)

    print("=== REAL-TIME PREDICTION ===")
    print(f"Bird: {result['bird_common_name']} ({result['bird_scientific_name']})")
    print(f"State: {result['state']}")
    print(f"Prediction: {result['prediction_label']}")
    print(f"Probability Present: {result['probability_present']:.4f}")

    print("\n=== MODEL FEATURES ===")
    for key, value in result["model_features"].items():
        print(f"{key}: {value}")

    print("\n=== SOUTHERN CORRIDOR SIGNAL ===")
    for key, value in result["corridor_features"].items():
        print(f"{key}: {value}")

    print(f"Interpretation: {result['interpretation']}")

    print(f"Arrival estimate: {result['arrival_estimate']}")