import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Project paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR, "dataset", "processed"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading training datasets...")

# Load datasets
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print("Training input:", X_train.shape)
print("Training output:", y_train.shape)

# Build LSTM model
model = Sequential([
    Input(shape=(5, 5)),

    LSTM(64, return_sequences=True),

    Dropout(0.2),

    LSTM(32),

    Dense(32, activation="relu"),

    Dense(2)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print("\nLSTM Model Architecture:")
model.summary()

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# Save best model
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_cyclone_model.keras"),
    monitor="val_loss",
    save_best_only=True
)

# Train model
print("\nStarting LSTM training...")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.15,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate model
print("\nEvaluating model...")

test_loss, test_mae = model.evaluate(
    X_test,
    y_test
)

print("\nTraining completed!")

print("Test Loss:", test_loss)
print("Test MAE:", test_mae)

# Save final model
model.save(
    os.path.join(MODEL_DIR, "cyclone_lstm.keras")
)

# Plot training loss
plt.figure(figsize=(10, 6))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title("Cyclone LSTM Training Performance")

plt.legend()
plt.grid()

plt.savefig(
    os.path.join(RESULT_DIR, "training_loss.png")
)

plt.show()

print("\nModel saved successfully!")