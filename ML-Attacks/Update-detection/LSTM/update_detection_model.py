import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle

with open("update_detection_data.json", "r") as f:
    data = json.load(f)

X = data["X"]
y = data["y"]

X = np.array(X)  
y = np.array(y)  

input_shape = X.shape[1:]  

model=Sequential([
    Masking(mask_value=0., input_shape=input_shape),
    LSTM(units=128, return_sequences=True,),
    LSTM(units=64, return_sequences=True),
    LSTM(units=32, return_sequences=False),
    Dropout(0.01),
    Dense(units=1, activation='sigmoid') 
])
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()

# Train/test split
# 80 %, 10 % val, 10 % test
X_temp, X_test, y_temp, y_test = train_test_split(X,y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)


model.save("lstm_ota_model.keras")

np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val) 
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)



