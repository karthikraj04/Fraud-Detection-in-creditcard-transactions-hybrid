import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

def build_hybrid_score(xgb_prob, iso_score, w1=0.7, w2=0.3):
    """
    Combine XGBoost probability and Isolation Forest anomaly score
    to create a final hybrid fraud score.
    """
    # Normalize anomaly score (0-1)
    iso_norm = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min())

    # Weighted combination
    hybrid = (w1 * xgb_prob) + (w2 * iso_norm)

    return hybrid


def evaluate_hybrid(hybrid_scores, y_test, threshold=0.5):
    """
    Evaluate hybrid scores by converting to binary predictions.
    """

    y_pred = (hybrid_scores >= threshold).astype(int)

    print("\n===== HYBRID MODEL RESULTS =====\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # AUROC
    roc_auc = roc_auc_score(y_test, hybrid_scores)
    print("ROC-AUC:", roc_auc)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, hybrid_scores)
    pr_auc = auc(recall, precision)
    print("PR-AUC:", pr_auc)

    return roc_auc, pr_auc
