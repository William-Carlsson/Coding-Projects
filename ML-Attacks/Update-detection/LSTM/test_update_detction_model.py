import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and model
X_test = np.load("X_test.npy")
print("X_test size:", X_test.shape)
y_test = np.load("y_test.npy")
model = load_model("lstm_ota_model.keras")

# Predict probabilities and binary labels
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# Compute confusion matrix and metrics
conf_mat = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_mat.ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
error_rate = 1 - accuracy
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
for_ = fn / (fn + tn) if (fn + tn) > 0 else 0

# Print evaluation metrics
print("\nTest Evaluation Metrics:")
print(f"Accuracy:               {accuracy:.4f}")
print(f"Precision:              {precision:.4f}")
print(f"Recall (Sensitivity):   {recall:.4f}")
print(f"F1 Score:               {f1:.4f}")
print(f"Error Rate:             {error_rate:.4f}")
print(f"False Positive Rate:    {fpr:.4f}")
print(f"False Negative Rate:    {fnr:.4f}")
print(f"False Discovery Rate:   {fdr:.4f}")
print(f"False Omission Rate:    {for_:.4f}")

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["True 0", "True 1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: LSTM")
plt.tight_layout()
plt.savefig("confusion_matrix_update_detection_lstm.png")
plt.show()

# ROC curve and AUC
fpr_roc, tpr_roc, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr_roc, tpr_roc)

plt.figure(figsize=(8, 6))
plt.plot(fpr_roc, tpr_roc, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guessing")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: LSTM")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_update_detection_lstm.png")
plt.show()


# === Save Evaluation to TXT ===
with open("lstm_evaluation_metrics.txt", "w") as f:
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
