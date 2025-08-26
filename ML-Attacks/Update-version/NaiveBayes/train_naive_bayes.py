import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# === Load Data ===
df = pd.read_csv("../filtered_features.csv")
X = df.drop("label", axis=1)
y = df["label"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
nb = GaussianNB()
nb.fit(X_train, y_train)

# === Save Model and Test Data ===
joblib.dump(nb, "naive_bayes_model.pkl")
X_test.to_csv("NaiveBayes_X_test.csv", index=False)
y_test.to_csv("NaiveBayes_y_test.csv", index=False)

print("Naive Bayes model trained and saved successfully.")