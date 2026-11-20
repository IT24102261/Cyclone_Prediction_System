import sys
import numpy as np
import pandas as pd

from pathlib import Path

from prediction_engine import load_prediction_model


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "active_lstm_inputs"
)

OUTPUT_DIR = BASE_DIR / "results"

FEATURE_COLUMNS = [
    "LAT",
    "LON",
    "MOVEMENT_SPEED",
    "DIR_SIN",
    "DIR_COS"
]

SEQUENCE_LENGTH = 5
FORECAST_STEPS = 8
INTERVAL_HOURS = 3


# =====================================================
# DISTANCE CALCULATION
# =====================================================

def haversine(lat1, lon1, lat2, lon2):

    radius = 6371.0

    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )

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


# =====================================================
# BEARING CALCULATION
# =====================================================

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


# =====================================================
# PREDICT SELECTED CYCLONE
# =====================================================

def predict_active_cyclone(sid):

    sid = str(sid).strip()

    print("\n" + "=" * 55)
    print("ACTIVE CYCLONE LSTM PREDICTION")
    print("=" * 55)

    print("Selected Storm ID:", sid)

    # -------------------------------------------------
    # LOAD INPUT
    # -------------------------------------------------

    input_file = INPUT_DIR / f"{sid}_3hour_input.csv"

    if not input_file.exists():

        raise FileNotFoundError(
            f"No prepared input found for {sid}"
        )

    df = pd.read_csv(input_file)

    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"],
        errors="coerce",
        utc=True
    )

    df = df.sort_values(
        "ISO_TIME"
    ).reset_index(drop=True)

    if len(df) < SEQUENCE_LENGTH:

        raise ValueError(
            "Not enough observations for prediction."
        )

    # Latest five observations
    recent = df.tail(
        SEQUENCE_LENGTH
    ).copy()

    # Validate time intervals
    intervals = (
        recent["ISO_TIME"]
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    if not intervals.eq(10800).all():

        raise ValueError(
            "Latest five observations are not consecutive "
            "three-hour intervals."
        )

    if recent[FEATURE_COLUMNS].isna().any().any():

        raise ValueError(
            "Missing values found in LSTM input."
        )

    synthetic_count = int(
        recent["IS_SYNTHETIC"].sum()
    )

    print("\nInput observations:", len(recent))

    print(
        "Synthetic observations:",
        synthetic_count
    )

    print(
        "Latest observation:",
        recent["ISO_TIME"].iloc[-1]
    )

    # -------------------------------------------------
    # LOAD MODEL AND SCALER
    # -------------------------------------------------

    print("\nLoading existing LSTM model...")

    model, scaler = load_prediction_model()

    print("Model loaded successfully.")

    # -------------------------------------------------
    # PREPARE MODEL INPUT
    # -------------------------------------------------

    input_features = recent[
        FEATURE_COLUMNS
    ].copy()

    scaled_features = scaler.transform(
        input_features
    )

    sequence = scaled_features.reshape(
        1,
        SEQUENCE_LENGTH,
        len(FEATURE_COLUMNS)
    )

    current_lat = float(
        recent["LAT"].iloc[-1]
    )

    current_lon = float(
        recent["LON"].iloc[-1]
    )

    current_time = recent["ISO_TIME"].iloc[-1]

    current_speed = float(
        recent["MOVEMENT_SPEED"].iloc[-1]
    )

    current_sin = float(
        recent["DIR_SIN"].iloc[-1]
    )

    current_cos = float(
        recent["DIR_COS"].iloc[-1]
    )

    predictions = []

    # -------------------------------------------------
    # RECURSIVE FORECAST
    # -------------------------------------------------

    print("\nGenerating 24-hour forecast...")

    for step in range(1, FORECAST_STEPS + 1):

        prediction = model.predict(
            sequence,
            verbose=0
        )

        delta_lat = float(
            prediction[0][0]
        )

        delta_lon = float(
            prediction[0][1]
        )

        previous_lat = current_lat
        previous_lon = current_lon

        current_lat += delta_lat

        current_lon += delta_lon

        # Normalize longitude
        current_lon = (
            (current_lon + 180) % 360
        ) - 180

        distance = haversine(
            previous_lat,
            previous_lon,
            current_lat,
            current_lon
        )

        current_speed = (
            distance / INTERVAL_HOURS
        )

        direction = calculate_bearing(
            previous_lat,
            previous_lon,
            current_lat,
            current_lon
        )

        direction_radians = np.radians(
            direction
        )

        current_sin = np.sin(
            direction_radians
        )

        current_cos = np.cos(
            direction_radians
        )

        current_time += pd.Timedelta(
            hours=INTERVAL_HOURS
        )

        predictions.append({

            "SID": sid,

            "STEP": step,

            "HOURS_AHEAD": (
                step * INTERVAL_HOURS
            ),

            "ISO_TIME": current_time,

            "LAT": current_lat,

            "LON": current_lon,

            "MOVEMENT_SPEED": current_speed,

            "MOVEMENT_DIRECTION": direction

        })

        # Prepare next recursive input
        next_features = pd.DataFrame(
            [[
                current_lat,
                current_lon,
                current_speed,
                current_sin,
                current_cos
            ]],
            columns=FEATURE_COLUMNS
        )

        scaled_next = scaler.transform(
            next_features
        )

        sequence = np.concatenate(
            [
                sequence[:, 1:, :],
                scaled_next.reshape(
                    1,
                    1,
                    len(FEATURE_COLUMNS)
                )
            ],
            axis=1
        )

    # -------------------------------------------------
    # SAVE PREDICTION
    # -------------------------------------------------

    prediction_df = pd.DataFrame(
        predictions
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR / f"{sid}_lstm_prediction.csv"
    )

    prediction_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 55)
    print("PREDICTION RESULTS")
    print("=" * 55)

    print(
        prediction_df.to_string(index=False)
    )

    print("\nSaved:", output_file)

    print("\nPREDICTION COMPLETED")

    return prediction_df


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            "py src/predict_active_cyclone.py <STORM_ID>"
        )

        sys.exit(1)

    selected_sid = sys.argv[1]

    try:

        predict_active_cyclone(
            selected_sid
        )

    except Exception as error:

        print("\nPREDICTION ERROR:", error)

        sys.exit(1)