import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# === Load Data ===
df = pd.read_csv("../filtered_features.csv")
X = df.drop(columns=["label"])
y = df["label"]

# === Encode Labels ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_categorical = to_categorical(y_encoded)

# === Normalize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# === Save Preprocessors ===
joblib.dump(scaler, "ffnn_scaler.pkl")
joblib.dump(encoder, "ffnn_label_encoder.pkl")
pd.DataFrame(X_test).to_csv("FFNN_X_test.csv", index=False)
pd.DataFrame(np.argmax(y_test, axis=1)).to_csv("FFNN_y_test.csv", index=False)

# === Build Model ===
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# === Train Model ===
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# === Save Model ===
model.save("ffnn_model.h5")
