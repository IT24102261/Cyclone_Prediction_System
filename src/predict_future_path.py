
import numpy as np
import pandas as pd
import os
import joblib

from tensorflow.keras.models import load_model

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR, "dataset", "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR, "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)

DATA_FILE = os.path.join(
    DATA_DIR, "cyclone_features.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR, "cyclone_lstm.keras"
)

SCALER_FILE = os.path.join(
    MODEL_DIR, "feature_scaler.pkl"
)

# ==========================================
# 2. CONFIGURATION
# ==========================================

SEQUENCE_LENGTH = 5

INTERVAL_HOURS = 3

FORECAST_HOURS = 24

FORECAST_STEPS = (
    FORECAST_HOURS // INTERVAL_HOURS
)

FEATURE_COLUMNS = [
    "LAT",
    "LON",
    "MOVEMENT_SPEED",
    "DIR_SIN",
    "DIR_COS"
]

# ==========================================
# 3. CHECK REQUIRED FILES
# ==========================================

print("Checking project files...")

required_files = [
    DATA_FILE,
    MODEL_FILE,
    SCALER_FILE
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

print("All required files found!")

# ==========================================
# 4. LOAD DATASET
# ==========================================

print("\nLoading cyclone dataset...")

df = pd.read_csv(
    DATA_FILE,
    dtype={"SID": str}
)

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

print("Dataset shape:", df.shape)

print("Available columns:")
print(df.columns.tolist())

# ==========================================
# 5. LOAD MODEL AND SCALER
# ==========================================

print("\nLoading trained LSTM model...")

model = load_model(
    MODEL_FILE
)

scaler = joblib.load(
    SCALER_FILE
)

print("Model loaded successfully!")

# ==========================================
# 6. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "SID",
    "ISO_TIME",
    "LAT",
    "LON",
    "MOVEMENT_SPEED",
    "MOVEMENT_DIRECTION",
    "TIME_DIFF"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# ==========================================
# 7. CLEAN DATA
# ==========================================

print("\nCleaning cyclone observations...")

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

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

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

# ==========================================
# 8. CREATE DIRECTION FEATURES
# ==========================================

direction_rad = np.radians(
    df["MOVEMENT_DIRECTION"]
)

df["DIR_SIN"] = np.sin(
    direction_rad
)

df["DIR_COS"] = np.cos(
    direction_rad
)

df = df.sort_values(
    ["SID", "ISO_TIME"]
).reset_index(drop=True)

print("Valid observations:", len(df))

print(
    "Unique cyclones:",
    df["SID"].nunique()
)

# ==========================================
# 9. SELECT LATEST ELIGIBLE CYCLONE
# ==========================================

print("\nSelecting latest cyclone...")

selected_cyclone = None

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

    if not intervals.between(
        2.5, 3.5
    ).all():

        continue

    if selected_cyclone is None:

        selected_cyclone = (
            sid,
            cyclone
        )

    elif (
        cyclone["ISO_TIME"].iloc[-1]
        >
        selected_cyclone[1]["ISO_TIME"].iloc[-1]
    ):

        selected_cyclone = (
            sid,
            cyclone
        )

if selected_cyclone is None:

    raise ValueError(
        "No cyclone with five consecutive observations found."
    )

sid, cyclone = selected_cyclone

historical = cyclone.tail(
    SEQUENCE_LENGTH
).copy()

print("\nSelected cyclone:", sid)

if "NAME" in historical.columns:

    print(
        "Cyclone name:",
        historical["NAME"].iloc[-1]
    )

print(
    "Last observation:",
    historical["ISO_TIME"].iloc[-1]
)

# ==========================================
# 10. PREPARE INITIAL INPUT
# ==========================================

print("\nPreparing LSTM input...")

initial_features = historical[
    FEATURE_COLUMNS
].copy()

current_sequence = scaler.transform(
    initial_features
).astype(np.float32)

current_lat = float(
    historical["LAT"].iloc[-1]
)

current_lon = float(
    historical["LON"].iloc[-1]
)

current_time = historical[
    "ISO_TIME"
].iloc[-1]

# ==========================================
# 11. GEOGRAPHICAL FUNCTIONS
# ==========================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1

    dlon = (
        (lon2 - lon1 + np.pi)
        % (2 * np.pi)
    ) - np.pi

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2 * R *
        np.arcsin(
            np.sqrt(
                np.clip(a, 0, 1)
            )
        )
    )


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
        np.sin(lat1) * np.cos(lat2)
        * np.cos(dlon)
    )

    bearing = np.degrees(
        np.arctan2(y, x)
    )

    return (bearing + 360) % 360


# ==========================================
# 12. GENERATE FUTURE PREDICTIONS
# ==========================================

print("\nGenerating future cyclone trajectory...")

predicted_path = []

for step in range(
    1,
    FORECAST_STEPS + 1
):

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

    # Calculate next position

    next_lat = current_lat + delta_lat

    next_lon = current_lon + delta_lon

    next_lon = (
        (next_lon + 180) % 360
    ) - 180

    if not -90 <= next_lat <= 90:

        raise ValueError(
            "Predicted latitude is outside valid range."
        )

    # Update timestamp

    current_time += pd.Timedelta(
        hours=INTERVAL_HOURS
    )

    # Calculate movement speed

    distance = haversine(
        current_lat,
        current_lon,
        next_lat,
        next_lon
    )

    next_speed = (
        distance / INTERVAL_HOURS
    )

    # Calculate direction

    next_direction = calculate_bearing(
        current_lat,
        current_lon,
        next_lat,
        next_lon
    )

    direction_rad = np.radians(
        next_direction
    )

    # Prepare next input

    next_features = np.array([[
        next_lat,
        next_lon,
        next_speed,
        np.sin(direction_rad),
        np.cos(direction_rad)
    ]])

    normalized_features = scaler.transform(
        next_features
    )

    # Update sequence

    current_sequence = np.vstack([
        current_sequence[1:],
        normalized_features
    ]).astype(np.float32)

    # Save prediction

    predicted_path.append({

        "STEP": step,

        "FORECAST_HOURS":
            step * INTERVAL_HOURS,

        "ISO_TIME": current_time,

        "LAT": next_lat,

        "LON": next_lon,

        "MOVEMENT_SPEED": next_speed,

        "MOVEMENT_DIRECTION": next_direction

    })

    current_lat = next_lat

    current_lon = next_lon

    print(
        f"{step * INTERVAL_HOURS} hours: "
        f"Latitude={next_lat:.3f}, "
        f"Longitude={next_lon:.3f}, "
        f"Speed={next_speed:.2f} km/h"
    )

# ==========================================
# 13. SAVE PREDICTED PATH
# ==========================================

print("\nSaving prediction results...")

prediction_df = pd.DataFrame(
    predicted_path
)

prediction_file = os.path.join(
    RESULT_DIR,
    "predicted_cyclone_path.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

# ==========================================
# 14. SAVE HISTORICAL PATH
# ==========================================

historical_path = historical[
    ["ISO_TIME", "LAT", "LON"]
].copy()

historical_path["TYPE"] = "Historical"

historical_file = os.path.join(
    RESULT_DIR,
    "historical_cyclone_path.csv"
)

historical_path.to_csv(
    historical_file,
    index=False
)

# ==========================================
# 15. DISPLAY RESULTS
# ==========================================

print("\n====================================")
print("CYCLONE FORECAST COMPLETED")
print("====================================")

print("Cyclone ID:", sid)

print(
    "Forecast duration:",
    FORECAST_HOURS,
    "hours"
)

print(
    "Prediction steps:",
    FORECAST_STEPS
)

print("\nPredicted cyclone trajectory:")

print(
    prediction_df[
        [
            "FORECAST_HOURS",
            "LAT",
            "LON",
            "MOVEMENT_SPEED"
        ]
    ].to_string(index=False)
)

print("\nFiles saved successfully!")

print("Predicted path:", prediction_file)

print("Historical path:", historical_file)

print("\nCyclone trajectory generation completed!")
