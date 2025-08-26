import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# === Load Model and Test Data ===
model = load_model("gru_update_detection_model.keras")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# === Predict ===
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# === Core Metrics ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
error_rate = (fp + fn) / (tp + tn + fp + fn)

# === Extended Metrics ===
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
for_ = fn / (fn + tn) if (fn + tn) > 0 else 0

# === Print Metrics ===
print("\nGRU Update Detection - Binary Classification Metrics:")
print(f"Accuracy:               {accuracy:.4f}")
print(f"Precision:              {precision:.4f}")
print(f"Recall (Sensitivity):   {recall:.4f}")
print(f"F1 Score:               {f1:.4f}")
print(f"Error Rate:             {error_rate:.4f}")
print(f"False Positive Rate:    {fpr:.4f}")
print(f"False Negative Rate:    {fnr:.4f}")
print(f"False Discovery Rate:   {fdr:.4f}")
print(f"False Omission Rate:    {for_:.4f}")

# === Confusion Matrix Plot ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: GRU")
plt.tight_layout()
plt.savefig("gru_confusion_matrix.png")
plt.show()

# === ROC Curve ===
fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr_curve, tpr_curve, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guessing")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: GRU")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("gru_roc_curve.png")
plt.show()

# === Save Evaluation to TXT ===
with open("gru_evaluation_metrics.txt", "w") as f:
    f.write("GRU Update Detection - Binary Classification Metrics:\n")
    f.write(f"Accuracy:               {accuracy:.4f}\n")
    f.write(f"Precision:              {precision:.4f}\n")
    f.write(f"Recall (Sensitivity):   {recall:.4f}\n")
    f.write(f"F1 Score:               {f1:.4f}\n")
    f.write(f"Error Rate:             {error_rate:.4f}\n")
    f.write(f"False Positive Rate:    {fpr:.4f}\n")
    f.write(f"False Negative Rate:    {fnr:.4f}\n")
    f.write(f"False Discovery Rate:   {fdr:.4f}\n")
    f.write(f"False Omission Rate:    {for_:.4f}\n")
    f.write(f"ROC AUC Score:          {roc_auc:.4f}\n")
