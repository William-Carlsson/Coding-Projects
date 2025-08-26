import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# === Load Test Data ===
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# === Load Trained RNN Model ===
model = load_model("rnn_update_detection_model.keras")

# === Predict ===
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

# === Metrics ===
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
error_rate = 1 - accuracy

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
for_ = fn / (fn + tn) if (fn + tn) > 0 else 0

# === Print Metrics ===
print("\nRNN Update Detection Evaluation Metrics:")
print(f"Accuracy:               {accuracy:.4f}")
print(f"Precision:              {precision:.4f}")
print(f"Recall (Sensitivity):   {recall:.4f}")
print(f"F1 Score:               {f1:.4f}")
print(f"Error Rate:             {error_rate:.4f}")
print(f"False Positive Rate:    {fpr:.4f}")
print(f"False Negative Rate:    {fnr:.4f}")
print(f"False Discovery Rate:   {fdr:.4f}")
print(f"False Omission Rate:    {for_:.4f}")

# === Confusion Matrix Heatmap ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: RNN")
plt.tight_layout()
plt.savefig("rnn_confusion_matrix.png")
plt.show()

# === ROC Curve and AUC ===
fpr_roc, tpr_roc, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr_roc, tpr_roc, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guessing")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: RNN")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("rnn_roc_curve.png")
plt.show()

# === Save Metrics to TXT ===
with open("rnn_evaluation_metrics.txt", "w") as f:
    f.write("RNN Update Detection Evaluation Metrics:\n")
    f.write(f"Accuracy:               {accuracy:.4f}\n")
    f.write(f"Precision:              {precision:.4f}\n")
    f.write(f"Recall (Sensitivity):   {recall:.4f}\n")
    f.write(f"F1 Score:               {f1:.4f}\n")
    f.write(f"Error Rate:             {error_rate:.4f}\n")
    f.write(f"False Positive Rate:    {fpr:.4f}\n")
    f.write(f"False Negative Rate:    {fnr:.4f}\n")
    f.write(f"False Discovery Rate:   {fdr:.4f}\n")
    f.write(f"False Omission Rate:    {for_:.4f}\n")
