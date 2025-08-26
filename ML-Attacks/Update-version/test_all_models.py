import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import os

# === Load predictor and test data ===
predictor = TabularPredictor.load("AutogluonModels/ag-20250625_154040")
test_data = pd.read_csv("test.csv")
label_column = predictor.label
y_true = test_data[label_column]
class_labels = sorted(y_true.unique())
classes = np.array(class_labels)

# === Create output directory ===
os.makedirs("model_evaluation_outputs", exist_ok=True)

# === Loop over all models ===
all_models = predictor.model_names()
results = []

for model_name in all_models:
    print(f"\n Evaluating Model: {model_name}")
    try:
        y_pred = predictor.predict(test_data, model=model_name)
        y_pred_proba = predictor.predict_proba(test_data, model=model_name)

        # Basic Metrics
        accuracy = accuracy_score(y_true, y_pred)
        error_rate = 1 - accuracy
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix: {model_name}")
        plt.tight_layout()
        plt.savefig(f"model_evaluation_outputs/conf_matrix_heatmap_{model_name}.png")
        plt.close()

        # Advanced Metrics: FPR, FNR, FDR, FOR
        fpr_list, fnr_list, fdr_list, for_list = [], [], [], []
        for i, class_label in enumerate(classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)

            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
            FDR = FP / (TP + FP) if (TP + FP) > 0 else 0
            FOR = FN / (FN + TN) if (FN + TN) > 0 else 0

            fpr_list.append(FPR)
            fnr_list.append(FNR)
            fdr_list.append(FDR)
            for_list.append(FOR)

        fpr_macro = np.mean(fpr_list)
        fnr_macro = np.mean(fnr_list)
        fdr_macro = np.mean(fdr_list)
        for_macro = np.mean(for_list)

        # ROC AUC (macro)
        y_true_bin = label_binarize(y_true, classes=class_labels)
        y_pred_proba_array = y_pred_proba[class_labels].to_numpy()
        roc_auc_macro = roc_auc_score(y_true_bin, y_pred_proba_array, average="macro", multi_class="ovr")

        # ROC Curve
        fpr, tpr, auc_scores = {}, {}, {}
        plt.figure(figsize=(8, 6))
        for i, class_id in enumerate(class_labels):
            fpr[class_id], tpr[class_id], _ = roc_curve(y_true_bin[:, i], y_pred_proba_array[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_pred_proba_array[:, i])
            auc_scores[class_id] = auc
            plt.plot(fpr[class_id], tpr[class_id], label=f"Class {class_id} (AUC = {auc:.2f})")

        plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve : {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"model_evaluation_outputs/roc_curve_{model_name}.png")
        plt.close()

        # Store all metrics
        results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision_macro, 4),
        "Recall": round(recall_macro, 4),
        "F1 Score": round(f1_macro, 4),
        "Error Rate": round(error_rate, 4),
        "False Positive Rate": round(fpr_macro, 4),
        "False Negative Rate": round(fnr_macro, 4),
        "False Discovery Rate": round(fdr_macro, 4),
        "False Omission Rate": round(for_macro, 4)
        })

    except Exception as e:
        print(f"Error evaluating model '{model_name}': {e}")

# === Save All Metrics to CSV ===
df_results = pd.DataFrame(results)
df_results.to_csv("model_evaluation_outputs/all_model_metrics.csv", index=False)
print("\n All model evaluations complete. Results saved in 'model_evaluation_outputs/' folder.")
