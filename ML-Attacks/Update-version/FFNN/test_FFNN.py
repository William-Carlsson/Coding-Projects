import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# === Load Test Data ===
X_test = pd.read_csv("FFNN_X_test.csv")
y_test = pd.read_csv("FFNN_y_test.csv").squeeze()
class_labels = sorted(y_test.unique())

# === Encode Labels ===
encoder = LabelEncoder()
y_test_encoded = encoder.fit_transform(y_test)
y_test_cat = to_categorical(y_test_encoded)

# === Scale Features ===
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# === Load Trained FFNN Model ===
model = load_model("ffnn_model.h5")

# === Predict ===
start_time = time.time()
y_pred_proba = model.predict(X_test_scaled)
detection_time = time.time() - start_time
y_pred_encoded = np.argmax(y_pred_proba, axis=1)
y_pred = encoder.inverse_transform(y_pred_encoded)

# === Standard Metrics ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
error_rate = 1 - accuracy

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

with np.errstate(divide='ignore', invalid='ignore'):
    fpr_macro = np.mean(np.nan_to_num(FP / (FP + TN)))
    fnr_macro = np.mean(np.nan_to_num(FN / (FN + TP)))
    fdr_macro = np.mean(np.nan_to_num(FP / (TP + FP)))
    for_macro = np.mean(np.nan_to_num(FN / (FN + TN)))

# === Print Evaluation Metrics ===
print("FFNN Multi-Class Evaluation Metrics:\n")
print(f"Accuracy:               {accuracy:.4f}")
print(f"Precision (Macro):      {precision:.4f}")
print(f"Recall (Macro):         {recall:.4f}")
print(f"F1 Score (Macro):       {f1:.4f}")
print(f"Error Rate:             {error_rate:.4f}")
print(f"False Positive Rate:    {fpr_macro:.4f}")
print(f"False Negative Rate:    {fnr_macro:.4f}")
print(f"False Discovery Rate:   {fdr_macro:.4f}")
print(f"False Omission Rate:    {for_macro:.4f}")

# === ROC AUC (Macro) ===
y_test_bin = label_binarize(y_test, classes=class_labels)
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average="macro", multi_class="ovr")

# === ROC Curve with AUC in Legend ===
fpr = dict()
tpr = dict()
roc_auc_class = dict()
plt.figure(figsize=(8, 6))
for i in range(len(class_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc_class[i] = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
    plt.plot(fpr[i], tpr[i], label=f"Class {class_labels[i]} (AUC = {roc_auc_class[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: FFNN")
plt.legend(loc="lower right", fontsize="small")
plt.tight_layout()
plt.savefig("ffnn_roc_multiclass_selected.png")
plt.show()

# === Confusion Matrix Plot ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: FFNN")
plt.tight_layout()
plt.savefig("ffnn_multiclass_confusion_matrix.png")
plt.show()

# === Save Evaluation Metrics to TXT ===
with open("ffnn_evaluation_metrics.txt", "w") as f:
    f.write("FFNN Multi-Class Evaluation Metrics:\n")
    f.write(f"Accuracy:               {accuracy:.4f}\n")
    f.write(f"Precision (Macro):      {precision:.4f}\n")
    f.write(f"Recall (Macro):         {recall:.4f}\n")
    f.write(f"F1 Score (Macro):       {f1:.4f}\n")
    f.write(f"Error Rate:             {error_rate:.4f}\n")
    f.write(f"False Positive Rate:    {fpr_macro:.4f}\n")
    f.write(f"False Negative Rate:    {fnr_macro:.4f}\n")
    f.write(f"False Discovery Rate:   {fdr_macro:.4f}\n")
    f.write(f"False Omission Rate:    {for_macro:.4f}\n")
