STATE_CONFIG = {
    "Massachusetts": {
        "lat": 42.4072,
        "lon": -71.3824,
        "ebird_region": "US-MA",
        "south_states": ["Connecticut", "New York", "New Jersey", "Pennsylvania", "Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "New Hampshire": {
        "lat": 43.1939,
        "lon": -71.5724,
        "ebird_region": "US-NH",
        "south_states": ["Massachusetts", "Connecticut", "New York", "New Jersey", "Pennsylvania", "Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "Maine": {
        "lat": 45.2538,
        "lon": -69.4455,
        "ebird_region": "US-ME",
        "south_states": ["New Hampshire", "Massachusetts", "Connecticut", "New York", "New Jersey", "Pennsylvania", "Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "Vermont": {
        "lat": 44.5588,
        "lon": -72.5778,
        "ebird_region": "US-VT",
        "south_states": ["Massachusetts", "Connecticut", "New York", "New Jersey", "Pennsylvania", "Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "Rhode Island": {
        "lat": 41.5801,
        "lon": -71.4774,
        "ebird_region": "US-RI",
        "south_states": ["Connecticut", "New York", "New Jersey", "Pennsylvania", "Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "Connecticut": {
        "lat": 41.6032,
        "lon": -73.0877,
        "ebird_region": "US-CT",
        "south_states": ["New York", "New Jersey", "Pennsylvania", "Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "New York": {
        "lat": 42.9538,
        "lon": -75.5268,
        "ebird_region": "US-NY",
        "south_states": ["New Jersey", "Pennsylvania", "Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "Pennsylvania": {
        "lat": 41.2033,
        "lon": -77.1945,
        "ebird_region": "US-PA",
        "south_states": ["Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "New Jersey": {
        "lat": 40.0583,
        "lon": -74.4057,
        "ebird_region": "US-NJ",
        "south_states": ["Maryland", "Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "Maryland": {
        "lat": 39.0458,
        "lon": -76.6413,
        "ebird_region": "US-MD",
        "south_states": ["Virginia", "North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "Virginia": {
        "lat": 37.4316,
        "lon": -78.6569,
        "ebird_region": "US-VA",
        "south_states": ["North Carolina", "South Carolina", "Georgia", "Florida"],
    },
    "North Carolina": {
        "lat": 35.7596,
        "lon": -79.0193,
        "ebird_region": "US-NC",
        "south_states": ["South Carolina", "Georgia", "Florida"],
    },
    "South Carolina": {
        "lat": 33.8361,
        "lon": -81.1637,
        "ebird_region": "US-SC",
        "south_states": ["Georgia", "Florida"],
    },
    "Georgia": {
        "lat": 32.1656,
        "lon": -82.9001,
        "ebird_region": "US-GA",
        "south_states": ["Florida"],
    },
    "Florida": {
        "lat": 27.6648,
        "lon": -81.5158,
        "ebird_region": "US-FL",
        "south_states": [],
    },
}

STATE_COORDS = {state: (cfg["lat"], cfg["lon"]) for state, cfg in STATE_CONFIG.items()}