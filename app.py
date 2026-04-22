import streamlit as st

from src.predict_realtime import predict_presence
from src.services.species_map import COMMON_TO_SCIENTIFIC

NORTHEAST_STATES = [
    "Massachusetts",
    "New Hampshire",
    "Maine",
    "Vermont",
    "Rhode Island",
    "Connecticut",
    "New York",
]

BIRD_OPTIONS = list(COMMON_TO_SCIENTIFIC.keys())

st.set_page_config(
    page_title="Spring Migration Predictor",
    page_icon="🐦",
    layout="centered",
)

st.title("🐦 Spring Bird Migration Predictor")
st.write(
    "Select a Northeast state and a bird species to estimate real-time spring presence "
    "using recent eBird activity and weather."
)

state = st.selectbox("Select a state", NORTHEAST_STATES)
bird = st.selectbox("Select a bird", BIRD_OPTIONS)

if st.button("Generate Prediction"):
    with st.spinner("Fetching live weather and recent sightings..."):
        try:
            result = predict_presence(state, bird)

            prob = result["probability_present"]
            label = result["prediction_label"]
            interpretation = result["interpretation"]

            st.subheader("Prediction")
            st.metric("Presence Probability", f"{prob:.1%}")
            st.write(f"**Classification:** {label}")
            st.write(f"**Interpretation:** {interpretation}")

            st.subheader("Bird")
            st.write(f"**Common name:** {result['bird_common_name']}")
            st.write(f"**Scientific name:** {result['bird_scientific_name']}")
            st.write(f"**State:** {result['state']}")

            st.subheader("Recent eBird Signal")
            st.write(
                f"- Last 7 days in target state: **{result['model_features']['lag_1_obs_count']}**"
            )
            st.write(
                f"- Previous 7 days in target state: **{result['model_features']['lag_2_obs_count']}**"
            )
            st.write(
                f"- Southern corridor last 7 days: **{result['corridor_features']['south_corridor_obs_last_7d']}**"
            )
            st.write(
                f"- Southern corridor previous 7 days: **{result['corridor_features']['south_corridor_obs_prev_7d']}**"
            )

            st.subheader("Recent Weather")
            st.write(
                f"- Mean temperature (7d): **{result['model_features']['temp_mean_7d']:.1f}°C**"
            )
            st.write(
                f"- Max temperature (7d): **{result['model_features']['temp_max_7d']:.1f}°C**"
            )
            st.write(
                f"- Min temperature (7d): **{result['model_features']['temp_min_7d']:.1f}°C**"
            )
            st.write(
                f"- Precipitation sum (7d): **{result['model_features']['precip_sum_7d']:.1f} mm**"
            )
            st.write(
                f"- Max wind speed (7d): **{result['model_features']['wind_max_7d']:.1f} km/h**"
            )

            with st.expander("Model Feature Details"):
                st.json(result["model_features"])

        except Exception as e:
            st.error(f"Prediction failed: {e}")