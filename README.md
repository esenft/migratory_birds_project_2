# Spring Bird Migration Predictor

## Overview

This project is a real-time bird migration prediction system.

It uses:
- A trained machine learning model
- Recent bird observation data from the eBird API
- Live weather data from the Open-Meteo API

The goal is to predict whether a selected bird species is currently present in a given state, and if not, estimate when it is likely to arrive.

The system is designed as an agent-based workflow:
- Weather Agent (fetches weather data)
- eBird Agent (fetches recent sightings)
- Migration Corridor Agent (tracks movement northward)
- Decision Agent (runs the machine learning model)

The app provides:
- Presence probability
- Classification (Present / Not Present)
- Estimated arrival window
- Explanation of the prediction

Project structure: 
app.py                 → Streamlit user interface  
src/                   → Core logic and scripts  
src/services/          → API clients (eBird, weather)  
data/                  → Local datasets (not included in repo)  
models/                → Trained model file  

Required dependencies:
duckdb  
pandas  
pyarrow  
scikit-learn  
joblib  
requests  
streamlit  
python-dotenv (optional but safe to include)

Requirements:
- Python 3.10 or higher (recommended: 3.12)

## API Keys
The project requires API keys from eBird, which can be generated from: https://ebird.org/api/keygen

## Environment Variables

You must set:

EBIRD_API_KEY

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-project-folder>
pip install duckdb pandas pyarrow scikit-learn joblib requests streamlit python-dotenv numpy

```markdown
If the app does not load, ensure port 8501 is open and accessible.
