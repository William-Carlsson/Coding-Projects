import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# === Load data from JSON ===
with open("../update_detection_data.json", "r") as f:
    data = json.load(f)

X = np.array(data['X'])  # shape (num_samples, 67, 12)
y = np.array(data['y'])  # shape (num_samples,)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Build RNN model ===
model = Sequential([
    Masking(mask_value=0., input_shape=(67, 12)),
    SimpleRNN(64, return_sequences=True),
    Dropout(0.3),
    SimpleRNN(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === Train ===
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

# === Save model ===
model.save("rnn_update_detection_model.keras")

# === Save test data for separate testing script ===
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
