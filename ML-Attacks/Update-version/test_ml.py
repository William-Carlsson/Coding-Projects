import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from autogluon.tabular import TabularPredictor

# Load data
test_data = pd.read_csv("test.csv")
y_true = test_data["label"]
class_labels = sorted(y_true.unique())

# Load predictor
predictor = TabularPredictor.load("AutogluonModels/ag-20250612_145536")

# Predict class labels and probabilities
y_pred = predictor.predict(test_data)
y_pred_proba = predictor.predict_proba(test_data)

# Basic Metrics
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average="macro")
recall_macro = recall_score(y_true, y_pred, average="macro")
f1_macro = f1_score(y_true, y_pred, average="macro")

precision_weighted = precision_score(y_true, y_pred, average="weighted")
recall_weighted = recall_score(y_true, y_pred, average="weighted")
f1_weighted = f1_score(y_true, y_pred, average="weighted")

# Print metrics
print("\nAutoGluon Multi-Class Evaluation Metrics:")
print(f"Accuracy:               {accuracy:.4f}")
print(f"Precision (Macro):      {precision_macro:.4f}")
print(f"Recall (Macro):         {recall_macro:.4f}")
print(f"F1 Score (Macro):       {f1_macro:.4f}")
print(f"Precision (Weighted):   {precision_weighted:.4f}")
print(f"Recall (Weighted):      {recall_weighted:.4f}")
print(f"F1 Score (Weighted):    {f1_weighted:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred, labels=class_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(xticks_rotation=45)
plt.title("AutoGluon Multi-Class Confusion Matrix")
plt.tight_layout()
plt.savefig("autogluon_multiclass_confusion_matrix.png")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# ROC Curve (Multi-Class AUC Macro)
y_true_bin = label_binarize(y_true, classes=class_labels)
y_pred_proba_array = y_pred_proba[class_labels].to_numpy()
roc_auc = roc_auc_score(y_true_bin, y_pred_proba_array, average="macro", multi_class="ovr")

print(f"\nMacro-Averaged ROC AUC Score: {roc_auc:.4f}")

# Optional: One-vs-rest ROC curves
fpr = dict()
tpr = dict()
for i, class_id in enumerate(class_labels):
    fpr[class_id], tpr[class_id], _ = roc_curve(y_true_bin[:, i], y_pred_proba_array[:, i])

# Plot ROC curves for a few selected classes
plt.figure()
for class_id in class_labels[:5]:  # Limit to first 5 to avoid clutter
    plt.plot(fpr[class_id], tpr[class_id], label=f"Class {class_id}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Selected Classes)")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig("autogluon_roc_multiclass_selected.png")
plt.show()
