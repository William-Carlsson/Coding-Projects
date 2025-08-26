import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import json

# === Load JSON data ===
# Assume your JSON file contains keys "X" and "y" where:
# X: List of sessions, each session is a list of 67 packets, each with 12 features
# y: List of labels (0 or 1)
with open("../update_detection_data.json", "r") as f:
    data = json.load(f)

X = np.array(data["X"])  # shape (num_samples, 67, 12)
y = np.array(data["y"])  # shape (num_samples,)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Define GRU model ===
model = Sequential([
    Masking(mask_value=0.0, input_shape=(67, 12)),  # Mask zeros if any padding exists
    GRU(64, return_sequences=False),  # 64 GRU units
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification output
])

# === Compile ===
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

# === Train ===
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    verbose=2
)

# === Save model and test data ===
model.save("gru_update_detection_model.keras")
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
