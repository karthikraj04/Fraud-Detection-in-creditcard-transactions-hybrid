from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from src.imbalance import get_class_weights


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression using class weights on fraud data."""

    # Compute class weights using the function you added
    class_weights = get_class_weights(y_train)

    # Build logistic regression model
    model = LogisticRegression(
        class_weight=class_weights,
        max_iter=1000,
        solver='liblinear'
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    print("\n===== LOGISTIC REGRESSION RESULTS =====\n")

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", roc_auc)

    # PR-AUC Score (VERY important for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print("PR-AUC:", pr_auc)

    return model, roc_auc, pr_auc



def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost using class weights for imbalanced fraud data."""

    class_weights = get_class_weights(y_train)

    scale_pos = class_weights[1] / class_weights[0] 
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,   
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n===== XGBOOST RESULTS =====\n")

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", roc_auc)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print("PR-AUC:", pr_auc)

    return model, roc_auc, pr_auc
