import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "cyclone_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading cyclone dataset...")

df = pd.read_csv(INPUT_FILE)

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"],
    errors="coerce"
)

# ==========================================
# 3. CONVERT NUMERICAL COLUMNS
# ==========================================

numeric_columns = [
    "LAT",
    "LON",
    "TIME_DIFF",
    "MOVEMENT_SPEED",
    "MOVEMENT_DIRECTION"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# ==========================================
# 4. CLEAN DATA
# ==========================================

df = df.dropna(
    subset=[
        "SID",
        "ISO_TIME",
        "LAT",
        "LON",
        "TIME_DIFF",
        "MOVEMENT_SPEED",
        "MOVEMENT_DIRECTION"
    ]
)

# Remove duplicate observations
df = df.drop_duplicates(
    subset=["SID", "ISO_TIME"]
)

# Validate coordinates
df = df[
    df["LAT"].between(-90, 90) &
    df["LON"].between(-180, 180)
].copy()

# ==========================================
# 5. FILTER THREE-HOUR OBSERVATIONS
# ==========================================

print("\nFiltering three-hour observations...")

df = df[
    df["TIME_DIFF"].between(2.5, 3.5)
].copy()

# Remove unrealistic movement speeds
df = df[
    df["MOVEMENT_SPEED"].between(0, 100)
].copy()

df = df.sort_values(
    ["SID", "ISO_TIME"]
).reset_index(drop=True)

print("Valid observations:", len(df))
print("Unique cyclones:", df["SID"].nunique())

# ==========================================
# 6. PRESERVE ACTUAL COORDINATES
# ==========================================

# IMPORTANT:
# Save actual coordinates BEFORE normalization.

df["ACTUAL_LAT"] = df["LAT"].copy()

df["ACTUAL_LON"] = df["LON"].copy()

# ==========================================
# 7. CREATE DIRECTION FEATURES
# ==========================================

direction_rad = np.radians(
    df["MOVEMENT_DIRECTION"]
)

df["DIR_SIN"] = np.sin(direction_rad)

df["DIR_COS"] = np.cos(direction_rad)

# ==========================================
# 8. SELECT INPUT FEATURES
# ==========================================

feature_columns = [
    "LAT",
    "LON",
    "MOVEMENT_SPEED",
    "DIR_SIN",
    "DIR_COS"
]

# ==========================================
# 9. SPLIT BY CYCLONE ID
# ==========================================

print("\nSplitting cyclone datasets...")

cyclone_ids = df["SID"].unique()

if len(cyclone_ids) < 2:
    raise ValueError(
        "Not enough cyclones for train/test splitting."
    )

train_ids, test_ids = train_test_split(
    cyclone_ids,
    test_size=0.20,
    random_state=42
)

train_df = df[
    df["SID"].isin(train_ids)
].copy()

test_df = df[
    df["SID"].isin(test_ids)
].copy()

print("Training cyclones:", len(train_ids))
print("Testing cyclones:", len(test_ids))

# ==========================================
# 10. NORMALIZE INPUT FEATURES ONLY
# ==========================================

print("\nNormalizing input features...")

scaler = MinMaxScaler()

train_df.loc[:, feature_columns] = (
    scaler.fit_transform(
        train_df[feature_columns]
    )
)

test_df.loc[:, feature_columns] = (
    scaler.transform(
        test_df[feature_columns]
    )
)

# Actual coordinates remain unchanged.

# ==========================================
# 11. CREATE DISPLACEMENT SEQUENCES
# ==========================================

SEQUENCE_LENGTH = 5


def create_sequences(data):

    X = []
    y = []

    for sid, cyclone in data.groupby("SID"):

        cyclone = cyclone.sort_values(
            "ISO_TIME"
        ).reset_index(drop=True)

        # Normalized input features
        input_values = cyclone[
            feature_columns
        ].values

        # Actual geographical coordinates
        actual_coordinates = cyclone[
            ["ACTUAL_LAT", "ACTUAL_LON"]
        ].values

        timestamps = cyclone[
            "ISO_TIME"
        ].values

        for i in range(
            len(cyclone) - SEQUENCE_LENGTH
        ):

            # Check consecutive observations
            time_differences = (
                np.diff(
                    timestamps[
                        i:i + SEQUENCE_LENGTH + 1
                    ]
                ) / np.timedelta64(1, "h")
            )

            # Require approximately 3-hour intervals
            if not np.all(
                (time_differences >= 2.5) &
                (time_differences <= 3.5)
            ):
                continue

            # Five historical observations
            sequence = input_values[
                i:i + SEQUENCE_LENGTH
            ]

            # Last observed actual position
            current_position = actual_coordinates[
                i + SEQUENCE_LENGTH - 1
            ]

            # Actual future position
            future_position = actual_coordinates[
                i + SEQUENCE_LENGTH
            ]

            # Calculate geographical displacement
            delta_lat = (
                future_position[0] -
                current_position[0]
            )

            # Correct longitude wrapping
            delta_lon = (
                future_position[1] -
                current_position[1] + 180
            ) % 360 - 180

            X.append(sequence)

            y.append([
                delta_lat,
                delta_lon
            ])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


# ==========================================
# 12. GENERATE TRAINING SEQUENCES
# ==========================================

print("\nCreating training sequences...")

X_train, y_train = create_sequences(
    train_df
)

print("Creating testing sequences...")

X_test, y_test = create_sequences(
    test_df
)

# ==========================================
# 13. VALIDATE DATASETS
# ==========================================

if len(X_train) == 0:

    raise ValueError(
        "No training sequences generated."
    )

if len(X_test) == 0:

    raise ValueError(
        "No testing sequences generated."
    )

if not np.isfinite(X_train).all():
    raise ValueError("Invalid values in X_train.")

if not np.isfinite(y_train).all():
    raise ValueError("Invalid values in y_train.")

# ==========================================
# 14. SAVE DATASETS
# ==========================================

print("\nSaving datasets...")

np.save(
    os.path.join(OUTPUT_DIR, "X_train.npy"),
    X_train
)

np.save(
    os.path.join(OUTPUT_DIR, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(OUTPUT_DIR, "X_test.npy"),
    X_test
)

np.save(
    os.path.join(OUTPUT_DIR, "y_test.npy"),
    y_test
)

# Save fitted scaler
joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "feature_scaler.pkl"
    )
)

# ==========================================
# 15. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("DISPLACEMENT PREPARATION COMPLETED")
print("==========================================")

print("\nTraining cyclones:", len(train_ids))
print("Testing cyclones:", len(test_ids))

print("\nTraining input:", X_train.shape)
print("Training output:", y_train.shape)

print("\nTesting input:", X_test.shape)
print("Testing output:", y_test.shape)

print("\nInput features:")
print(feature_columns)

print("\nOutput features:")
print([
    "Latitude displacement (degrees)",
    "Longitude displacement (degrees)"
])

print("\nExample displacement:")
print(y_train[0])

print("\nDisplacement statistics:")

print(
    "Mean latitude displacement:",
    np.mean(y_train[:, 0])
)

print(
    "Mean longitude displacement:",
    np.mean(y_train[:, 1])
)

print("\nDatasets saved successfully!")

print("Scaler saved successfully!")