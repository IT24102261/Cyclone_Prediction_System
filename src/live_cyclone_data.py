"""
LIVE CYCLONE DATA SYSTEM
------------------------
Downloads active cyclone observations from NOAA IBTrACS,
filters North Indian Ocean cyclones, and prepares features
for the existing LSTM prediction model.

Data source: NOAA NCEI IBTrACS v04r01
"""

import requests
import pandas as pd
import numpy as np
from io import StringIO


# ============================================================
# CONFIGURATION
# ============================================================

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ACTIVE.list.v04r01.csv"
)

BASIN = "NI"

MIN_OBSERVATIONS = 5

EXPECTED_INTERVAL_HOURS = 3

FEATURE_COLUMNS = [
    "LAT",
    "LON",
    "MOVEMENT_SPEED",
    "DIR_SIN",
    "DIR_COS"
]


# ============================================================
# 1. DOWNLOAD ACTIVE CYCLONE DATA
# ============================================================

def download_active_cyclones():

    print("\nConnecting to NOAA IBTrACS...")

    try:

        headers = {
            "User-Agent": "CyclonePredictionResearchProject/1.0"
        }

        response = requests.get(
            IBTRACS_URL,
            headers=headers,
            timeout=60
        )

        response.raise_for_status()

        print("NOAA data downloaded successfully.")

        df = pd.read_csv(
            StringIO(response.text),
            skiprows=[1],
            low_memory=False
        )

        print(f"Downloaded observations: {len(df)}")

        return df

    except requests.exceptions.RequestException as error:

        print("\nERROR: Unable to download NOAA data.")
        print(error)

        return None

    except Exception as error:

        print("\nERROR: Unable to read NOAA CSV data.")
        print(error)

        return None


# ============================================================
# 2. CLEAN CYCLONE DATA
# ============================================================

def clean_cyclone_data(df):

    if df is None or df.empty:
        return pd.DataFrame()

    required_columns = [
        "SID",
        "SEASON",
        "NAME",
        "ISO_TIME",
        "LAT",
        "LON",
        "BASIN"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    # Convert time to UTC
    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"],
        errors="coerce",
        utc=True
    )

    # Convert coordinates to numeric values
    df["LAT"] = pd.to_numeric(
        df["LAT"],
        errors="coerce"
    )

    df["LON"] = pd.to_numeric(
        df["LON"],
        errors="coerce"
    )

    # Remove invalid observations
    df = df.dropna(
        subset=[
            "SID",
            "ISO_TIME",
            "LAT",
            "LON"
        ]
    )

    # Remove invalid coordinates
    df = df[
        df["LAT"].between(-90, 90)
        &
        df["LON"].between(-180, 360)
    ]

    # Remove duplicate observations
    df = df.drop_duplicates(
        subset=["SID", "ISO_TIME"]
    )

    # Sort cyclone observations
    df = df.sort_values(
        ["SID", "ISO_TIME"]
    ).reset_index(drop=True)

    return df


# ============================================================
# 3. FILTER NORTH INDIAN OCEAN CYCLONES
# ============================================================

def get_indian_ocean_cyclones(df):

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    df["BASIN"] = (
        df["BASIN"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    indian_ocean_df = df[
        df["BASIN"] == BASIN
    ].copy()

    indian_ocean_df = indian_ocean_df.sort_values(
        ["SID", "ISO_TIME"]
    ).reset_index(drop=True)

    return indian_ocean_df


# ============================================================
# 4. HAVERSINE DISTANCE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):

    earth_radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return earth_radius * c


# ============================================================
# 5. CALCULATE MOVEMENT DIRECTION
# ============================================================

def calculate_bearing(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    delta_lon = lon2 - lon1

    x = (
        np.sin(delta_lon)
        * np.cos(lat2)
    )

    y = (
        np.cos(lat1) * np.sin(lat2)
        -
        np.sin(lat1)
        * np.cos(lat2)
        * np.cos(delta_lon)
    )

    bearing = np.degrees(
        np.arctan2(x, y)
    )

    return (bearing + 360) % 360


# ============================================================
# 6. PREPARE CYCLONE FEATURES
# ============================================================

def prepare_cyclone_features(df):

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    df = df.sort_values(
        ["SID", "ISO_TIME"]
    ).reset_index(drop=True)

    df["PREV_LAT"] = (
        df.groupby("SID")["LAT"].shift(1)
    )

    df["PREV_LON"] = (
        df.groupby("SID")["LON"].shift(1)
    )

    df["PREV_TIME"] = (
        df.groupby("SID")["ISO_TIME"].shift(1)
    )

    df["TIME_DIFF"] = (
        df["ISO_TIME"] - df["PREV_TIME"]
    ).dt.total_seconds() / 3600

    df["DISTANCE_KM"] = haversine(
        df["PREV_LAT"],
        df["PREV_LON"],
        df["LAT"],
        df["LON"]
    )

    # Movement speed in km/h
    df["MOVEMENT_SPEED"] = (
        df["DISTANCE_KM"] / df["TIME_DIFF"]
    )

    df["MOVEMENT_DIRECTION"] = calculate_bearing(
        df["PREV_LAT"],
        df["PREV_LON"],
        df["LAT"],
        df["LON"]
    )

    # Direction features used by the trained LSTM
    direction_radians = np.radians(
        df["MOVEMENT_DIRECTION"]
    )

    df["DIR_SIN"] = np.sin(
        direction_radians
    )

    df["DIR_COS"] = np.cos(
        direction_radians
    )

    # Remove invalid movement calculations
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# 7. IDENTIFY CYCLONES SUITABLE FOR LSTM PREDICTION
# ============================================================

def get_available_cyclones(df):

    available = []

    if df.empty:
        return pd.DataFrame()

    for sid, group in df.groupby("SID"):

        group = group.sort_values(
            "ISO_TIME"
        ).reset_index(drop=True)

        # Require at least five observations
        if len(group) < MIN_OBSERVATIONS:
            continue

        # Select latest five observations
        recent = group.tail(
            MIN_OBSERVATIONS
        ).copy()

        # Check that the five observations are consecutive
        time_intervals = (
            recent["ISO_TIME"]
            .diff()
            .dt.total_seconds()
            .div(3600)
            .dropna()
        )

        valid_intervals = time_intervals.between(
            2.5,
            3.5
        )

        if not valid_intervals.all():
            continue

        # Ensure required model features are available
        if recent[FEATURE_COLUMNS].isna().any().any():
            continue

        if not np.isfinite(
            recent[FEATURE_COLUMNS].to_numpy(dtype=float)
        ).all():
            continue

        latest = recent.iloc[-1]

        available.append({
            "SID": sid,
            "NAME": latest["NAME"],
            "SEASON": latest["SEASON"],
            "LAST_OBSERVATION": latest["ISO_TIME"],
            "LAT": latest["LAT"],
            "LON": latest["LON"],
            "MOVEMENT_SPEED": latest["MOVEMENT_SPEED"],
            "MOVEMENT_DIRECTION": latest["MOVEMENT_DIRECTION"],
            "OBSERVATIONS": len(group)
        })

    return pd.DataFrame(available)


# ============================================================
# 8. MAIN LIVE DATA FUNCTION
# ============================================================

def get_live_cyclone_data():

    raw_data = download_active_cyclones()

    if raw_data is None:
        return pd.DataFrame(), pd.DataFrame()

    cleaned_data = clean_cyclone_data(
        raw_data
    )

    indian_ocean_data = get_indian_ocean_cyclones(
        cleaned_data
    )

    featured_data = prepare_cyclone_features(
        indian_ocean_data
    )

    available_cyclones = get_available_cyclones(
        featured_data
    )

    return featured_data, available_cyclones


# ============================================================
# 9. TEST THE LIVE DATA SYSTEM
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("LIVE CYCLONE DATA SYSTEM")
    print("======================================")

    featured_data, available_cyclones = (
        get_live_cyclone_data()
    )

    if featured_data.empty:

        print("\nNo North Indian Ocean observations found.")

    else:

        print("\n======================================")
        print("DATA DOWNLOAD SUMMARY")
        print("======================================")

        print(
            "Total observations:",
            len(featured_data)
        )

        print(
            "Unique cyclones:",
            featured_data["SID"].nunique()
        )

        print(
            "Available for LSTM prediction:",
            len(available_cyclones)
        )

        print("\n======================================")
        print("AVAILABLE CYCLONES")
        print("======================================")

        if available_cyclones.empty:

            print(
                "No cyclones currently have enough "
                "consecutive observations for prediction."
            )

        else:

            print(
                available_cyclones[
                    [
                        "SID",
                        "NAME",
                        "LAST_OBSERVATION",
                        "LAT",
                        "LON",
                        "MOVEMENT_SPEED",
                        "MOVEMENT_DIRECTION"
                    ]
                ].to_string(index=False)
            )

        # Save downloaded data
        featured_data.to_csv(
            "dataset/processed/live_cyclone_features.csv",
            index=False
        )

        available_cyclones.to_csv(
            "dataset/processed/available_live_cyclones.csv",
            index=False
        )

        print("\nLive cyclone data saved successfully.")

    print("\n======================================")
    print("LIVE DATA PROCESSING COMPLETED")
    print("======================================")