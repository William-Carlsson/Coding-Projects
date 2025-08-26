from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from evaluation_criteria import *

import pandas as pd
import joblib
import os



# Load the CSV file
df = pd.read_csv('filtered_features.csv')  # Replace with your actual file path

# Separate features and label
X = df.drop('label', axis=1)  # Change 'label' if your column name is different
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(solver='lbfgs', max_iter=5000, verbose=1)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

results = evaluate_classification_with_plots(y_test, y_pred, y_proba, name="Logistic Regression")

# Save metrics to a text file
metrics_path = os.path.join(r"C:\Users\Abbe\Desktop\Master-thesis\scripts\Update-version-AutoGluon\figs", 'logistic_evaluation_metrics.txt')
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




# Create a directory to save the model if it doesn't exist
model_dir = 'saved_model'
os.makedirs(model_dir, exist_ok=True)

# Save the trained model and the scaler
joblib.dump(model, os.path.join(model_dir, 'logistic_model.joblib'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))

print(f"Model and scaler saved in '{model_dir}' folder.")

# Save the test data (scaled features and true labels) to CSV
test_data_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_data_df['label'] = y_test.values

# Construct the file path with model name
test_data_path = os.path.join(r"C:\Users\Abbe\Desktop\Master-thesis\scripts\Update-version-AutoGluon\figs", 'logistic_test_data.csv')
test_data_df.to_csv(test_data_path, index=False)

print(f"Test data saved to '{test_data_path}'")