
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR, "dataset", "processed"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# =====================================
# 2. LOAD DATA
# =====================================

print("Loading test datasets...")

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "feature_scaler.pkl")
)

model = load_model(
    os.path.join(MODEL_DIR, "cyclone_lstm.keras")
)

# =====================================
# 3. GENERATE LSTM PREDICTIONS
# =====================================

print("\nGenerating displacement predictions...")

predicted_displacement = model.predict(
    X_test,
    verbose=1
)

# =====================================
# 4. RECOVER ACTUAL INPUT COORDINATES
# =====================================

original_features = scaler.inverse_transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)

# Last two observed cyclone positions

previous_positions = original_features[:, -2, :2]

current_positions = original_features[:, -1, :2]

# =====================================
# 5. RECONSTRUCT LSTM PREDICTED POSITION
# =====================================

lstm_predictions = np.zeros_like(
    predicted_displacement
)

lstm_predictions[:, 0] = (
    current_positions[:, 0]
    + predicted_displacement[:, 0]
)

lstm_predictions[:, 1] = (
    current_positions[:, 1]
    + predicted_displacement[:, 1]
)

# Normalize longitude to [-180, 180)
lstm_predictions[:, 1] = (
    (lstm_predictions[:, 1] + 180) % 360
) - 180

# =====================================
# 6. STATIONARY BASELINE
# =====================================

stationary_predictions = current_positions.copy()

# =====================================
# 7. CONSTANT VELOCITY BASELINE
# =====================================

constant_velocity_predictions = np.zeros_like(
    current_positions
)

constant_velocity_predictions[:, 0] = (
    current_positions[:, 0]
    + current_positions[:, 0]
    - previous_positions[:, 0]
)

longitude_change = (
    current_positions[:, 1]
    - previous_positions[:, 1]
)

longitude_change = (
    (longitude_change + 180) % 360
) - 180

constant_velocity_predictions[:, 1] = (
    current_positions[:, 1] + longitude_change
)

constant_velocity_predictions[:, 1] = (
    (constant_velocity_predictions[:, 1] + 180) % 360
) - 180

# =====================================
# 8. HAVERSINE DISTANCE
# =====================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

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
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )

# =====================================
# 9. CALCULATE ERRORS
# =====================================

actual_lat = y_test[:, 0]
actual_lon = y_test[:, 1]

# Actual future position = current position + actual displacement

actual_future_lat = (
    current_positions[:, 0] + actual_lat
)

actual_future_lon = (
    current_positions[:, 1] + actual_lon
)

actual_future_lon = (
    (actual_future_lon + 180) % 360
) - 180

# LSTM errors

lstm_errors = haversine(
    actual_future_lat,
    actual_future_lon,
    lstm_predictions[:, 0],
    lstm_predictions[:, 1]
)

# Stationary errors

stationary_errors = haversine(
    actual_future_lat,
    actual_future_lon,
    stationary_predictions[:, 0],
    stationary_predictions[:, 1]
)

# Constant velocity errors

velocity_errors = haversine(
    actual_future_lat,
    actual_future_lon,
    constant_velocity_predictions[:, 0],
    constant_velocity_predictions[:, 1]
)

# =====================================
# 10. DISPLAY RESULTS
# =====================================

models = {
    "Displacement LSTM": lstm_errors,
    "Stationary": stationary_errors,
    "Constant Velocity": velocity_errors
}

print("\n====================================")
print("CYCLONE MODEL COMPARISON")
print("====================================")

for name, errors in models.items():

    print(f"\n{name}")

    print(
        "Mean error:",
        round(np.mean(errors), 3),
        "km"
    )

    print(
        "Median error:",
        round(np.median(errors), 3),
        "km"
    )

    print(
        "90th percentile:",
        round(np.percentile(errors, 90), 3),
        "km"
    )

# =====================================
# 11. CALCULATE IMPROVEMENT
# =====================================

baseline_mean = np.mean(velocity_errors)

lstm_mean = np.mean(lstm_errors)

improvement = (
    (baseline_mean - lstm_mean)
    / baseline_mean
) * 100

print("\n====================================")
print("LSTM VS CONSTANT VELOCITY")
print("====================================")

print(
    "LSTM improvement:",
    round(improvement, 2),
    "%"
)

# =====================================
# 12. VISUALIZE RESULTS
# =====================================

plt.figure(figsize=(11, 6))

plt.boxplot(
    [
        lstm_errors,
        stationary_errors,
        velocity_errors
    ],
    tick_labels=[
        "Displacement LSTM",
        "Stationary",
        "Constant Velocity"
    ]
)

plt.ylabel("Prediction Error (km)")

plt.title(
    "Displacement LSTM vs Baseline Models"
)

plt.grid(axis="y")

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "displacement_model_comparison.png"
    )
)

plt.show()

print("\nEvaluation completed successfully!")
