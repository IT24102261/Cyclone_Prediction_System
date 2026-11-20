import pandas as pd
import numpy as np

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = Path("dataset/processed")

OUTPUT_DIR = DATA_DIR / "active_lstm_inputs"

SUMMARY_FILE = DATA_DIR / "active_cyclone_summary.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# HAVERSINE DISTANCE
# ==========================================

def haversine(lat1, lon1, lat2, lon2):

    radius = 6371.0

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

    return 2 * radius * np.arcsin(
        np.sqrt(a)
    )


# ==========================================
# BEARING
# ==========================================

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


# ==========================================
# PREPARE ONE CYCLONE
# ==========================================

def prepare_cyclone(file_path, sid):

    print("\n" + "=" * 55)
    print("Preparing cyclone:", sid)
    print("=" * 55)

    df = pd.read_csv(file_path)

    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"],
        errors="coerce",
        utc=True
    )

    for column in ["LAT", "LON", "INTENSITY"]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=["ISO_TIME", "LAT", "LON"]
    )

    df = df.sort_values(
        "ISO_TIME"
    ).drop_duplicates(
        subset=["ISO_TIME"]
    )

    original_count = len(df)

    if original_count < 5:

        print("Not enough historical observations.")

        return None

    # Create a three-hour timeline
    df = df.set_index("ISO_TIME")

    actual_times = set(df.index)

    three_hour = df[
        ["LAT", "LON", "INTENSITY"]
    ].resample("3h").asfreq()

    # Mark interpolated positions
    three_hour["IS_SYNTHETIC"] = [
        timestamp not in actual_times
        for timestamp in three_hour.index
    ]

    # Interpolate only between actual observations
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

    # Movement calculations
    three_hour["PREV_LAT"] = (
        three_hour["LAT"].shift(1)
    )

    three_hour["PREV_LON"] = (
        three_hour["LON"].shift(1)
    )

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

    radians = np.radians(
        three_hour["MOVEMENT_DIRECTION"]
    )

    three_hour["DIR_SIN"] = np.sin(radians)

    three_hour["DIR_COS"] = np.cos(radians)

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

    # Save cyclone-specific input
    output_file = (
        OUTPUT_DIR / f"{sid}_3hour_input.csv"
    )

    three_hour.to_csv(
        output_file,
        index=False
    )

    # Check latest five observations
    latest_five = three_hour.tail(5)

    intervals = (
        latest_five["ISO_TIME"]
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    eligible = (
        len(latest_five) == 5
        and intervals.eq(10800).all()
    )

    synthetic_count = int(
        latest_five["IS_SYNTHETIC"].sum()
    )

    print("Original observations:", original_count)

    print("Prepared observations:", len(three_hour))

    print(
        "Total synthetic observations:",
        int(three_hour["IS_SYNTHETIC"].sum())
    )

    print(
        "Synthetic observations in latest five:",
        synthetic_count
    )

    print("LSTM eligible:", eligible)

    print("Saved:", output_file)

    return {

        "SID": sid,

        "ORIGINAL_OBSERVATIONS": original_count,

        "PREPARED_OBSERVATIONS": len(three_hour),

        "SYNTHETIC_OBSERVATIONS": int(
            three_hour["IS_SYNTHETIC"].sum()
        ),

        "LATEST_FIVE_SYNTHETIC": synthetic_count,

        "LSTM_ELIGIBLE": eligible,

        "LATEST_OBSERVATION": three_hour["ISO_TIME"].max()

    }


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 55)
    print("MULTI-CYCLONE DATA PREPARATION")
    print("=" * 55)

    if not SUMMARY_FILE.exists():

        print("Active cyclone summary not found.")

        return

    active_cyclones = pd.read_csv(
        SUMMARY_FILE
    )

    results = []

    for _, cyclone in active_cyclones.iterrows():

        sid = str(cyclone["SID"])

        track_file = DATA_DIR / f"{sid}_track.csv"

        if not track_file.exists():

            print(
                f"\nTrack file not found for {sid}"
            )

            continue

        try:

            result = prepare_cyclone(
                track_file,
                sid
            )

            if result is not None:

                results.append(result)

        except Exception as error:

            print(
                f"\nError processing {sid}: {error}"
            )

    if results:

        summary = pd.DataFrame(results)

        summary.to_csv(
            OUTPUT_DIR / "preparation_summary.csv",
            index=False
        )

        print("\n" + "=" * 55)
        print("PREPARATION SUMMARY")
        print("=" * 55)

        print(
            summary.to_string(index=False)
        )

    print("\nMULTI-CYCLONE PREPARATION COMPLETED")


if __name__ == "__main__":

    main()