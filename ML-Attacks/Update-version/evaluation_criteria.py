import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize

def compute_additional_metrics(cm):
    """Compute FPR, FNR, FDR, and FOR for each class."""
    metrics = {}
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Avoid divide-by-zero
    FPR = FP / (FP + TN + 1e-10)
    FNR = FN / (FN + TP + 1e-10)
    FDR = FP / (FP + TP + 1e-10)
    FOR = FN / (FN + TN + 1e-10)

    metrics['FPR'] = FPR
    metrics['FNR'] = FNR
    metrics['FDR'] = FDR
    metrics['FOR'] = FOR

    return metrics


def compute_multiclass_roc_auc(y_true, y_proba, labels):
    """Compute macro-averaged ROC AUC, ignoring missing classes in y_true."""
    try:
        y_true_bin = label_binarize(y_true, classes=labels)
        aucs = []
        for i, cls in enumerate(labels):
            y_true_cls = y_true_bin[:, i]
            y_score_cls = y_proba[:, i]
            # Skip if only one class present
            if len(np.unique(y_true_cls)) < 2:
                continue
            auc = roc_auc_score(y_true_cls, y_score_cls)
            aucs.append(auc)
        return np.mean(aucs) if aucs else float('nan')
    except Exception as e:
        print(f"Error computing ROC AUC: {e}")
        return float('nan')


def plot_confusion_matrix(cm, class_names, name, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(name + " Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curves(y_true, y_proba, class_names, name, save_path=None):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(8, 6))

    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(name + " ROC Curves")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def evaluate_classification_with_plots(y_true, y_pred, y_proba=None, class_names=None, fig_dir="figs", name=None):
    os.makedirs(fig_dir, exist_ok=True)
    start_time = time.time()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = sorted(list(set(y_true)))

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    error_rate = 1 - accuracy
    detection_time = time.time() - start_time
    avg_mistake_rate = error_rate
    query_error_prob = np.mean(y_true != y_pred)

    # Per-class report
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # ROC Curves & AUC
    auc_macro = None
    if y_proba is not None and len(class_names) > 1:
        try:
            auc_macro = compute_multiclass_roc_auc(y_true, y_proba, labels=np.unique(np.concatenate([y_true, y_pred])))
            plot_roc_curves(y_true, y_proba, class_names, name, save_path=os.path.join(fig_dir, name + "_roc_curves.png"))
        except Exception as e:
            print(f"Warning: Couldn't compute ROC AUC. {e}")

    # Save confusion matrix
    plot_confusion_matrix(cm, class_names, name, save_path=os.path.join(fig_dir, name + "_confusion_matrix.png"))

    # Compute additional metrics
    additional_metrics = compute_additional_metrics(cm)

    return {
        "Accuracy": accuracy,
        "Precision (macro)": precision_macro,
        "Recall (macro)": recall_macro,
        "F1 Score (macro)": f1_macro,
        "Error Rate": error_rate,
        "Detection Time (s)": detection_time,
        "Average Mistake Rate": avg_mistake_rate,
        "Query Error Probability": query_error_prob,
        "ROC AUC (macro)": auc_macro,
        "Confusion Matrix": cm,
        "Per-Class Report": class_report,
        "FPR (per class)": additional_metrics['FPR'],
        "FNR (per class)": additional_metrics['FNR'],
        "FDR (per class)": additional_metrics['FDR'],
        "FOR (per class)": additional_metrics['FOR'],
        "FPR (macro)": np.mean(additional_metrics['FPR']),
        "FNR (macro)": np.mean(additional_metrics['FNR']),
        "FDR (macro)": np.mean(additional_metrics['FDR']),
        "FOR (macro)": np.mean(additional_metrics['FOR']),
    }