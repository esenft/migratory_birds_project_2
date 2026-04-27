import streamlit as st
import time

from src.predict_realtime import predict_presence
from src.services.species_map import COMMON_TO_SCIENTIFIC

def c_to_f(c):
    return (c * 9/5) + 32

def mm_to_in(mm):
    return mm / 25.4

def kmh_to_mph(kmh):
    return kmh * 0.621371

NORTHEAST_STATES = [
    "Connecticut",
    "Maine",
    "Massachusetts",
    "New Hampshire",
    "New York",
    "Rhode Island",
    "Vermont",
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

    st.subheader("Agent Workflow")

    # Weather Agent
    weather_status = st.empty()
    weather_status.info("🌦️ Weather Agent: Fetching live weather data...")
    
    # eBird Agent
    ebird_status = st.empty()
    corridor_status = st.empty()
    
    # Decision Agent
    decision_status = st.empty()

    try:
        # Step 1: Weather
        weather_status.info("🌦️ Weather Agent: Fetching live weather data...")
        time.sleep(2.5)

        # Step 2: eBird
        ebird_status.info("🐦 eBird Agent: Fetching recent bird sightings...")
        time.sleep(2.5)

        # Step 3: Migration Corridor
        corridor_status.info("🧭 Migration Corridor Agent: Analyzing migration front signals...")
        time.sleep(2.5)

        # Step 4: Decision
        decision_status.info("🤖 Decision Agent: Running machine learning model...")
        time.sleep(2.5)

        # Run prediction
        result = predict_presence(state, bird)
        time.sleep(2.5)

        # Update statuses to success
        weather_status.success("🌦️ Weather Agent: Completed")
        ebird_status.success("🐦 eBird Agent: Completed")
        corridor_status.success("🧭 Migration Corridor Agent: Completed")
        decision_status.success("🤖 Decision Agent: Completed")

        # --- Existing output below ---
        prob = result["probability_present"]
        label = result["prediction_label"]
        interpretation = result["interpretation"]

        st.subheader("Prediction")
        st.metric("Presence Probability", f"{prob:.1%}")
        st.write(f"**Classification:** {label}")
        final_interpretation = result.get("adjusted_interpretation") or interpretation
        st.write(f"**Interpretation:** {final_interpretation}")

        if result.get("migration_warning"):
            st.warning(result["migration_warning"])

        if label == "Not Present":
            st.write(f"**Estimated broader arrival window:** {result['arrival_estimate']}")
        else:
            st.success("Likely present now based on current model signals.")

        with st.expander("Explain prediction"):
            st.subheader("Recent eBird Activity Signal")

            st.caption(
                "eBird activity signal based on recent reports. "
                "Values reflect reported individual counts and reporting intensity, "
                "not exact population size or number of sightings."
            )

            local_recent = result["model_features"]["lag_1_obs_count"]
            local_previous = result["model_features"]["lag_2_obs_count"]
            corridor_recent = result["corridor_features"]["south_corridor_obs_last_7d"]
            corridor_previous = result["corridor_features"]["south_corridor_obs_prev_7d"]

            st.write(f"- Target state activity, last 7 days: **{local_recent:,.0f}**")
            st.write(f"- Target state activity, previous 7 days: **{local_previous:,.0f}**")
            st.write(f"- Southern corridor activity, last 7 days: **{corridor_recent:,.0f}**")
            st.write(f"- Southern corridor activity, previous 7 days: **{corridor_previous:,.0f}**")

            st.subheader("Recent Weather")

            temp_mean_f = c_to_f(result["model_features"]["temp_mean_7d"])
            temp_max_f = c_to_f(result["model_features"]["temp_max_7d"])
            temp_min_f = c_to_f(result["model_features"]["temp_min_7d"])
            precip_in = mm_to_in(result["model_features"]["precip_sum_7d"])
            wind_mph = kmh_to_mph(result["model_features"]["wind_max_7d"])

            st.write(f"- Mean temperature, last 7 days: **{temp_mean_f:.1f}°F**")
            st.write(f"- Max temperature, last 7 days: **{temp_max_f:.1f}°F**")
            st.write(f"- Min temperature, last 7 days: **{temp_min_f:.1f}°F**")
            st.write(f"- Precipitation, last 7 days: **{precip_in:.2f} inches**")
            st.write(f"- Max wind speed, last 7 days: **{wind_mph:.1f} mph**")

            st.subheader("Model Details")
            st.write(f"- Bird used by model: **{result['bird_scientific_name']}**")
            st.write(f"- State: **{result['state']}**")
            st.write(f"- Week of year: **{result['model_features']['week_of_year']}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        try:
            result = predict_presence(state, bird)

            prob = result["probability_present"]
            label = result["prediction_label"]
            interpretation = result["interpretation"]

            st.subheader("Prediction")
            st.metric("Raw Model Presence Probability", f"{prob:.1%}")
            st.write(f"**Classification:** {label}")
            st.write(f"**Interpretation:** {interpretation}")

            if prob < 0.8:
                st.write(f"**Estimated arrival:** {result['arrival_estimate']}")
            else:
              st.success("Likely already present in this state")

            st.subheader("Bird")
            st.write(f"**Common name:** {result['bird_common_name']}")
            st.write(f"**Scientific name:** {result['bird_scientific_name']}")
            st.write(f"**State:** {result['state']}")

            st.subheader("Recent eBird Signal")
            st.info(
                "These values reflect recent eBird reporting activity and individual counts, "
                "and should be interpreted as relative activity rather than exact population size."
            )

            local_recent = result["model_features"]["lag_1_obs_count"]
            local_previous = result["model_features"]["lag_2_obs_count"]
            corridor_recent = result["corridor_features"]["south_corridor_obs_last_7d"]
            corridor_previous = result["corridor_features"]["south_corridor_obs_prev_7d"]

            st.write(f"- Target state activity (last 7 days): **{local_recent:,.0f}**")
            st.write(f"- Target state activity (previous 7 days): **{local_previous:,.0f}**")
            st.write(f"- Southern corridor activity (last 7 days): **{corridor_recent:,.0f}**")
            st.write(f"- Southern corridor activity (previous 7 days): **{corridor_previous:,.0f}**")

            st.subheader("Recent Weather")

            temp_mean_f = c_to_f(result["model_features"]["temp_mean_7d"])
            temp_max_f = c_to_f(result["model_features"]["temp_max_7d"])
            temp_min_f = c_to_f(result["model_features"]["temp_min_7d"])
            precip_in = mm_to_in(result["model_features"]["precip_sum_7d"])
            wind_mph = kmh_to_mph(result["model_features"]["wind_max_7d"])

            st.write(f"- Mean temperature (7d): **{temp_mean_f:.1f}°F**")
            st.write(f"- Max temperature (7d): **{temp_max_f:.1f}°F**")
            st.write(f"- Min temperature (7d): **{temp_min_f:.1f}°F**")

            st.write(f"- Precipitation (7d): **{precip_in:.2f} inches**")
            st.write(f"- Max wind speed (7d): **{wind_mph:.1f} mph**")
            
            with st.expander("Model Feature Details"):
                st.json(result["model_features"])

        except Exception as e:
            st.error(f"Prediction failed: {e}")