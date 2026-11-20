import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path(
    "dataset/processed/cira_historical_track.csv"
)

OUTPUT_FILE = Path(
    "dataset/processed/cira_3hour_lstm_input.csv"
)


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

    return 2 * earth_radius * np.arcsin(
        np.sqrt(a)
    )


def calculate_bearing(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    delta_lon = lon2 - lon1

    x = np.sin(delta_lon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        -
        np.sin(lat1) * np.cos(lat2)
        * np.cos(delta_lon)
    )

    bearing = np.degrees(
        np.arctan2(x, y)
    )

    return (bearing + 360) % 360


def prepare_data():

    print("Loading CIRA historical observations...")

    df = pd.read_csv(INPUT_FILE)

    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"],
        utc=True
    )

    df = df.sort_values("ISO_TIME")

    df = df.drop_duplicates(
        subset=["ISO_TIME"]
    )

    df = df.set_index("ISO_TIME")

    # Interpolate latitude and longitude at 3-hour intervals.
    # Do not extrapolate outside the observed time range.
    three_hour = df[
        ["LAT", "LON", "INTENSITY"]
    ].resample("3h").asfreq()

    actual_times = set(df.index)

    three_hour["IS_SYNTHETIC"] = [
        timestamp not in actual_times
        for timestamp in three_hour.index
    ]

    three_hour[
        ["LAT", "LON", "INTENSITY"]
    ] = three_hour[
        ["LAT", "LON", "INTENSITY"]
    ].interpolate(
        method="time",
        limit_area="inside"
    )

    three_hour = three_hour.dropna(
        subset=["LAT", "LON"]
    )

    # Calculate movement features
    three_hour["PREV_LAT"] = three_hour["LAT"].shift(1)
    three_hour["PREV_LON"] = three_hour["LON"].shift(1)

    three_hour["DISTANCE_KM"] = haversine(
        three_hour["PREV_LAT"],
        three_hour["PREV_LON"],
        three_hour["LAT"],
        three_hour["LON"]
    )

    three_hour["TIME_DIFF"] = 3.0

    three_hour["MOVEMENT_SPEED"] = (
        three_hour["DISTANCE_KM"] / 3.0
    )

    three_hour["MOVEMENT_DIRECTION"] = calculate_bearing(
        three_hour["PREV_LAT"],
        three_hour["PREV_LON"],
        three_hour["LAT"],
        three_hour["LON"]
    )

    direction_radians = np.radians(
        three_hour["MOVEMENT_DIRECTION"]
    )

    three_hour["DIR_SIN"] = np.sin(
        direction_radians
    )

    three_hour["DIR_COS"] = np.cos(
        direction_radians
    )

    three_hour = three_hour.replace(
        [np.inf, -np.inf],
        np.nan
    )

    three_hour = three_hour.dropna(
        subset=[
            "MOVEMENT_SPEED",
            "DIR_SIN",
            "DIR_COS"
        ]
    )

    three_hour = three_hour.reset_index()

    three_hour.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n3-hour dataset created successfully!")

    print("Total observations:", len(three_hour))

    print(
        "Actual observations:",
        (~three_hour["IS_SYNTHETIC"]).sum()
    )

    print(
        "Synthetic observations:",
        three_hour["IS_SYNTHETIC"].sum()
    )

    print("\nLatest five observations:")

    print(
        three_hour.tail(5)[
            [
                "ISO_TIME",
                "LAT",
                "LON",
                "MOVEMENT_SPEED",
                "DIR_SIN",
                "DIR_COS",
                "IS_SYNTHETIC"
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":

    prepare_data()