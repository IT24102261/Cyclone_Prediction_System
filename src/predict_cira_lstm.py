import numpy as np
import pandas as pd
from pathlib import Path

from prediction_engine import load_prediction_model


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = Path(
    "dataset/processed/cira_3hour_lstm_input.csv"
)

OUTPUT_FILE = Path(
    "results/cira_lstm_prediction.csv"
)

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


# ==========================================
# DISTANCE CALCULATION
# ==========================================

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

    return radius * 2 * np.arcsin(np.sqrt(a))


# ==========================================
# BEARING CALCULATION
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
# LOAD DATA AND MODEL
# ==========================================

def main():

    print("\n======================================")
    print("CIRA + LSTM CYCLONE PREDICTION")
    print("======================================")

    # Load prepared CIRA observations
    df = pd.read_csv(INPUT_FILE)

    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"],
        utc=True
    )

    df = df.sort_values(
        "ISO_TIME"
    ).reset_index(drop=True)

    # Check required observations
    if len(df) < SEQUENCE_LENGTH:

        raise ValueError(
            "Not enough observations for LSTM prediction."
        )

    # Latest five observations
    recent = df.tail(
        SEQUENCE_LENGTH
    ).copy()

    if recent[FEATURE_COLUMNS].isna().any().any():

        raise ValueError(
            "Missing values in LSTM input features."
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

    # Load existing trained model and scaler
    print("\nLoading existing LSTM model...")

    model, scaler = load_prediction_model()

    print("Model loaded successfully.")

    # Prepare model input
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

    # Latest position and timestamp
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

    # ======================================
    # RECURSIVE 24-HOUR PREDICTION
    # ======================================

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

        # Calculate predicted movement
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
                    1, 1, len(FEATURE_COLUMNS)
                )
            ],
            axis=1
        )

    # ======================================
    # SAVE RESULTS
    # ======================================

    prediction_df = pd.DataFrame(
        predictions
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n======================================")
    print("24-HOUR PREDICTION RESULTS")
    print("======================================")

    print(
        prediction_df.to_string(index=False)
    )

    print("\nPrediction saved to:")
    print(OUTPUT_FILE)

    print("\n======================================")
    print("PREDICTION COMPLETED")
    print("======================================")


if __name__ == "__main__":

    main()