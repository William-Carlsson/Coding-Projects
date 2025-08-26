import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import joblib
import os
import json
from evaluation_criteria import evaluate_classification_with_plots


# Load data
df = pd.read_csv('filtered_features.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for CNN: [samples, timesteps, features]
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Define CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train_cnn, y_train_cat, epochs=30, batch_size=64, validation_split=0.1, verbose=1)

# Predict
y_pred_prob = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_prob, axis=1)

# Evaluate
results = evaluate_classification_with_plots(y_test, y_pred, y_pred_prob, name="CNN")

# Save model and label encoder
model_dir = 'saved_cnn_model'
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, 'cnn_model.keras'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))

# Save metrics to a text file
metrics_path = os.path.join(r"C:\Users\Abbe\Desktop\Master-thesis\scripts\Update-version-AutoGluon\figs", 'CNN_evaluation_metrics.txt')
# Save metrics to a text file
with open(metrics_path, 'w') as f:
    for key, value in results.items():
        if key == "Per-Class Report":
            f.write("\nPer-Class Metrics:\n")
            for cls, metrics in value.items():
                if isinstance(metrics, dict):
                    f.write(f"{str(cls):15s} " + " ".join([f"{m}:{metrics[m]:.2f}" for m in ['precision', 'recall', 'f1-score']]) + "\n")
        elif isinstance(value, (float, int)):
            f.write(f"{key}: {value:.4f}\n")
        elif isinstance(value, np.ndarray) or isinstance(value, list):
            f.write(f"{key} (per class):\n")
            for i, v in enumerate(value):
                # Handle nested arrays like confusion matrix rows
                if isinstance(v, (np.ndarray, list)):
                    v_str = ", ".join(f"{x:.4f}" for x in v)
                    f.write(f"  Class {i}: [{v_str}]\n")
                else:
                    f.write(f"  Class {i}: {v:.4f}\n")
        else:
            f.write(f"{key}: {str(value)}\n")  # fallback for anything else


print(f"Evaluation metrics saved to '{metrics_path}'")


# Save the test data (scaled features and true labels) to CSV
test_data_df = pd.DataFrame(X_test_scaled, columns=df.drop('label', axis=1).columns)
test_data_df['label'] = y_test.values

# Construct the file path with model name
test_data_path = os.path.join(r"C:\Users\Abbe\Desktop\Master-thesis\scripts\Update-version-AutoGluon\figs", 'CNN_test_data.csv')
test_data_df.to_csv(test_data_path, index=False)

print(f"Test data saved to '{test_data_path}'")
