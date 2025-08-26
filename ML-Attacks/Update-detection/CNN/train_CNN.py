import numpy as np
import tensorflow as tf
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split



# Load data
with open("../update_detection_data.json", "r") as f:
    data = json.load(f)

X = np.array(data["X"])
y = np.array(data["y"]).astype(np.float32)
y = np.clip(y, 0, 1)  # ensure binary labels (0 or 1)

# Normalize features using StandardScaler
from sklearn.preprocessing import StandardScaler

n_samples, seq_len, n_features = X.shape
X_flat = X.reshape(-1, n_features)  # Shape: (n_samples * seq_len, n_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)  # Scale all features

X = X_scaled.reshape(n_samples, seq_len, n_features)  # Back to original shape


input_shape = X.shape[1:]  # (timesteps, features)

# Train/test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42)

# CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.summary()

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# Save model and data
model.save("cnn_ota_model.keras")

np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
