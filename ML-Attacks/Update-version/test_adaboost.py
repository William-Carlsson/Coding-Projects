import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

# === Load Test Data ===
X_test = pd.read_csv("Adaboost_Data/Adaboost_X_test.csv")
y_test = pd.read_csv("Adaboost_Data/Adaboost_y_test.csv").squeeze()
print(y_test.value_counts())

# === Load Trained AdaBoost Model ===
ada = joblib.load("Adaboost_Data/adaboost_model.pkl")

# === Class Labels ===
class_labels = sorted(y_test.unique())
print(f"Class Labels: {class_labels}")

# === Predict ===
y_pred = ada.predict(X_test)
y_pred_proba = ada.predict_proba(X_test)

# === Basic Metrics ===
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

precision_weighted = precision_score(y_test, y_pred, average="weighted")
recall_weighted = recall_score(y_test, y_pred, average="weighted")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("\nAdaBoost Multi-Class Evaluation Metrics:")
print(f"Accuracy:               {accuracy:.4f}")
print(f"Precision (Macro):      {precision_macro:.4f}")
print(f"Recall (Macro):         {recall_macro:.4f}")
print(f"F1 Score (Macro):       {f1_macro:.4f}")
print(f"Precision (Weighted):   {precision_weighted:.4f}")
print(f"Recall (Weighted):      {recall_weighted:.4f}")
print(f"F1 Score (Weighted):    {f1_weighted:.4f}")

# === Confusion Matrix ===
print("\nConfusion Matrix:")
print(np.unique(y_pred))
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("AdaBoost Multi-Class Confusion Matrix")
plt.tight_layout()
plt.savefig("Adaboost_Data/adaboost_multiclass_confusion_matrix.png")
plt.show()

# === ROC AUC (Macro) ===
y_test_bin = label_binarize(y_test, classes=class_labels)
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average="macro", multi_class="ovr")
print(f"\nMacro-Averaged ROC AUC Score: {roc_auc:.4f}")

# === Plot ROC Curves ===
fpr = dict()
tpr = dict()
for i in range(len(class_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])

plt.figure()
for i in range(len(class_labels)):
    plt.plot(fpr[i], tpr[i], label=f"Class {class_labels[i]}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AdaBoost ROC Curves")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig("Adaboost_Data/adaboost_roc_multiclass_selected.png")
plt.show()
