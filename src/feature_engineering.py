import pandas as pd
import numpy as np
import os

# Project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Input and output paths
input_file = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "cleaned_cyclones.csv"
)

output_file = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "cyclone_features.csv"
)

print("Loading cleaned cyclone dataset...")

df = pd.read_csv(input_file)

# Convert date
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

# Sort cyclone observations
df = df.sort_values(["SID", "ISO_TIME"]).copy()

# Calculate previous positions
df["PREV_LAT"] = df.groupby("SID")["LAT"].shift(1)
df["PREV_LON"] = df.groupby("SID")["LON"].shift(1)

# Calculate latitude and longitude changes
df["LAT_CHANGE"] = df["LAT"] - df["PREV_LAT"]
df["LON_CHANGE"] = df["LON"] - df["PREV_LON"]

# Calculate time difference in hours
df["TIME_DIFF"] = (
    df.groupby("SID")["ISO_TIME"]
    .diff()
    .dt.total_seconds() / 3600
)

# Haversine formula to calculate distance in kilometres
def haversine(lat1, lon1, lat2, lon2):

    earth_radius = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return earth_radius * c


# Calculate distance travelled
df["DISTANCE_KM"] = haversine(
    df["PREV_LAT"],
    df["PREV_LON"],
    df["LAT"],
    df["LON"]
)

# Calculate movement speed in km/h
df["MOVEMENT_SPEED"] = (
    df["DISTANCE_KM"] / df["TIME_DIFF"]
)

# Calculate movement direction in degrees
def calculate_direction(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    direction = np.degrees(np.arctan2(x, y))

    return (direction + 360) % 360


df["MOVEMENT_DIRECTION"] = calculate_direction(
    df["PREV_LAT"],
    df["PREV_LON"],
    df["LAT"],
    df["LON"]
)

# Remove invalid movement calculations
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Save feature dataset
df.to_csv(output_file, index=False)

print("\nFeature engineering completed!")

print("Total observations:", len(df))

print("\nNew features:")
print([
    "LAT_CHANGE",
    "LON_CHANGE",
    "TIME_DIFF",
    "DISTANCE_KM",
    "MOVEMENT_SPEED",
    "MOVEMENT_DIRECTION"
])

print("\nSample results:")
print(
    df[[
        "SID",
        "LAT",
        "LON",
        "DISTANCE_KM",
        "MOVEMENT_SPEED",
        "MOVEMENT_DIRECTION"
    ]].head(10)
)

print("\nSaved to:", output_file)