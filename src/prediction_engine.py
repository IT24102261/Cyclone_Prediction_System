import os
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "cyclone_features.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "cyclone_lstm.keras"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "models",
    "feature_scaler.pkl"
)


# ==========================================
# MODEL CONFIGURATION
# ==========================================

SEQUENCE_LENGTH = 5
INTERVAL_HOURS = 3
FORECAST_HOURS = 24
FORECAST_STEPS = 8

FEATURE_COLUMNS = [
    "LAT",
    "LON",
    "MOVEMENT_SPEED",
    "DIR_SIN",
    "DIR_COS"
]


# ==========================================
# LOAD DATA
# ==========================================

def load_cyclone_data():

    df = pd.read_csv(
        DATA_FILE,
        dtype={"SID": str}
    )

    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"],
        errors="coerce"
    )

    numeric_columns = [
        "LAT",
        "LON",
        "MOVEMENT_SPEED",
        "MOVEMENT_DIRECTION",
        "TIME_DIFF"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    required_columns = [
        "SID",
        "ISO_TIME",
        "LAT",
        "LON",
        "MOVEMENT_SPEED",
        "MOVEMENT_DIRECTION",
        "TIME_DIFF"
    ]

    df = df.dropna(
        subset=required_columns
    )

    df = df[
        df["LAT"].between(-90, 90)
        & df["LON"].between(-180, 180)
    ].copy()

    df = df[
        df["TIME_DIFF"].between(2.5, 3.5)
    ].copy()

    df = df[
        df["MOVEMENT_SPEED"].between(0, 100)
    ].copy()

    df = df.drop_duplicates(
        subset=["SID", "ISO_TIME"]
    )

    direction_rad = np.radians(
        df["MOVEMENT_DIRECTION"]
    )

    df["DIR_SIN"] = np.sin(direction_rad)
    df["DIR_COS"] = np.cos(direction_rad)

    df = df.sort_values(
        ["SID", "ISO_TIME"]
    ).reset_index(drop=True)

    return df


# ==========================================
# LOAD MODEL AND SCALER
# ==========================================

def load_prediction_model():

    model = load_model(MODEL_FILE)

    scaler = joblib.load(SCALER_FILE)

    return model, scaler


# ==========================================
# HAVERSINE DISTANCE
# ==========================================

def haversine(lat1, lon1, lat2, lon2):

    radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1

    dlon = (
        (lon2 - lon1 + np.pi) % (2 * np.pi)
    ) - np.pi

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2 * radius *
        np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


# ==========================================
# CALCULATE BEARING
# ==========================================

def calculate_bearing(
    lat1, lon1, lat2, lon2
):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlon = np.radians(
        (lon2 - lon1 + 180) % 360 - 180
    )

    y = np.sin(dlon) * np.cos(lat2)

    x = (
        np.cos(lat1) * np.sin(lat2)
        -
        np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    bearing = np.degrees(
        np.arctan2(y, x)
    )

    return (bearing + 360) % 360


# ==========================================
# CHECK ELIGIBLE CYCLONES
# ==========================================

def get_eligible_cyclones(df):

    eligible = []

    for sid, cyclone in df.groupby("SID"):

        cyclone = cyclone.sort_values(
            "ISO_TIME"
        ).reset_index(drop=True)

        if len(cyclone) < SEQUENCE_LENGTH:
            continue

        last_five = cyclone.tail(
            SEQUENCE_LENGTH
        )

        intervals = (
            last_five["ISO_TIME"]
            .diff()
            .dropna()
            .dt.total_seconds() / 3600
        )

        if len(intervals) != 4:
            continue

        if not intervals.between(2.5, 3.5).all():
            continue

        latest = cyclone.iloc[-1]

        eligible.append({

            "SID": str(sid),

            "NAME": str(
                latest.get("NAME", "UNNAMED")
            ),

            "SEASON": int(latest["SEASON"]),

            "OBSERVATIONS": len(cyclone),

            "LAST_OBSERVATION": latest["ISO_TIME"]

        })

    return pd.DataFrame(eligible)


# ==========================================
# PREDICT SELECTED CYCLONE
# ==========================================

def predict_cyclone(
    sid,
    df,
    model,
    scaler
):

    cyclone = df[
        df["SID"].astype(str) == str(sid)
    ].copy()

    cyclone = cyclone.sort_values(
        "ISO_TIME"
    ).reset_index(drop=True)

    if len(cyclone) < SEQUENCE_LENGTH:
        raise ValueError(
            "Selected cyclone has fewer than five valid observations."
        )

    # Use five consecutive observations

    historical_input = cyclone.tail(
        SEQUENCE_LENGTH
    ).copy()

    intervals = (
        historical_input["ISO_TIME"]
        .diff()
        .dropna()
        .dt.total_seconds() / 3600
    )

    if not intervals.between(2.5, 3.5).all():
        raise ValueError(
            "Selected cyclone does not have five consecutive 3-hour observations."
        )

    # Keep the full valid historical track for visualization

    historical = cyclone[
        ["ISO_TIME", "LAT", "LON"]
    ].copy()

    historical["TYPE"] = "Historical"

    # Prepare model input

    initial_features = historical_input[
        FEATURE_COLUMNS
    ].copy()

    current_sequence = scaler.transform(
        initial_features
    ).astype(np.float32)

    current_lat = float(
        historical_input["LAT"].iloc[-1]
    )

    current_lon = float(
        historical_input["LON"].iloc[-1]
    )

    current_time = historical_input[
        "ISO_TIME"
    ].iloc[-1]

    # Generate predictions

    predicted_path = []

    for step in range(1, FORECAST_STEPS + 1):

        prediction = model.predict(
            current_sequence.reshape(
                1,
                SEQUENCE_LENGTH,
                len(FEATURE_COLUMNS)
            ),
            verbose=0
        )[0]

        delta_lat = float(prediction[0])
        delta_lon = float(prediction[1])

        next_lat = current_lat + delta_lat

        next_lon = (
            (current_lon + delta_lon + 180) % 360
        ) - 180

        if not -90 <= next_lat <= 90:
            raise ValueError(
                "Predicted latitude is outside valid range."
            )

        current_time += pd.Timedelta(
            hours=INTERVAL_HOURS
        )

        distance = haversine(
            current_lat,
            current_lon,
            next_lat,
            next_lon
        )

        next_speed = distance / INTERVAL_HOURS

        next_direction = calculate_bearing(
            current_lat,
            current_lon,
            next_lat,
            next_lon
        )

        direction_rad = np.radians(
            next_direction
        )

        next_features = pd.DataFrame(
            [[
                next_lat,
                next_lon,
                next_speed,
                np.sin(direction_rad),
                np.cos(direction_rad)
            ]],
            columns=FEATURE_COLUMNS
        )

        normalized_features = scaler.transform(
            next_features
        )

        current_sequence = np.vstack([
            current_sequence[1:],
            normalized_features
        ]).astype(np.float32)

        predicted_path.append({

            "STEP": step,

            "FORECAST_HOURS": (
                step * INTERVAL_HOURS
            ),

            "ISO_TIME": current_time,

            "LAT": next_lat,

            "LON": next_lon,

            "MOVEMENT_SPEED": next_speed,

            "MOVEMENT_DIRECTION": next_direction

        })

        current_lat = next_lat
        current_lon = next_lon

    predicted = pd.DataFrame(
        predicted_path
    )

    cyclone_name = str(
        cyclone["NAME"].iloc[-1]
    )

    season = int(
        cyclone["SEASON"].iloc[-1]
    )

    metadata = {
        "SID": str(sid),
        "NAME": cyclone_name,
        "SEASON": season,
        "LAST_OBSERVATION": historical_input[
            "ISO_TIME"
        ].iloc[-1]
    }

    return historical, predicted, metadata