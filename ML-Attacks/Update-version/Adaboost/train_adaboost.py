# train_adaboost.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load data
df = pd.read_csv("../filtered_features.csv")

# Split into features (X) and target (y)
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base estimator
base_estimator = DecisionTreeClassifier()

# AdaBoost model
ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, learning_rate=0.01, random_state=42)

# Train
ada.fit(X_train, y_train)

# Save model
joblib.dump(ada, "adaboost_model.pkl")
X_test.to_csv("Adaboost_X_test.csv", index=False)
y_test.to_csv("Adaboost_y_test.csv", index=False)

print("AdaBoost model trained and saved successfully.")