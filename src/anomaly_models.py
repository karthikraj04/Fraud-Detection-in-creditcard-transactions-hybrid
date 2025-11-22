from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import numpy as np


def train_isolation_forest(X_train, y_train, X_test, y_test):
    """
    Train Isolation Forest on NORMAL samples only.
    Evaluate on full test set.
    """

    # 1. Train ONLY on class 0 (normal transactions)
 
    X_train_normal = X_train[y_train == 0]

    model = IsolationForest(
        n_estimators=300,
        contamination='auto',   
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_normal)

    # 2. Predict anomaly scores
    test_pred_raw = model.predict(X_test)

    # Convert raw predictions to fraud labels
    y_pred = np.where(test_pred_raw == -1, 1, 0)

    # 3. Compute anomaly score 
    anomaly_score = -model.decision_function(X_test)  

    # 4. Evaluation
    print("\n===== ISOLATION FOREST RESULTS =====\n")

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, anomaly_score)
    print("ROC-AUC:", roc_auc)

    # PR-AUC 
    precision, recall, _ = precision_recall_curve(y_test, anomaly_score)
    pr_auc = auc(recall, precision)
    print("PR-AUC:", pr_auc)

    return model, anomaly_score, roc_auc, pr_auc
