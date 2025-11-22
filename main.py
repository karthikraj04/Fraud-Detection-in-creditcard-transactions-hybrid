from src.preprocess import load_data, preprocess_data, get_train_test
from src.supervised_models import train_logistic_regression, train_xgboost
from src.anomaly_models import train_isolation_forest
from src.hybrid_model import build_hybrid_score, evaluate_hybrid


def main():
    print("Loading dataset...")
    df = load_data("data/creditcard.csv")

    print("Preprocessing data...")
    df = preprocess_data(df)

    print("Splitting into train/test...")
    X_train, X_test, y_train, y_test = get_train_test(df)

    # 1. Train XGBoost (supervised model)
    print("\nTraining XGBoost...")
    xgb_model, xgb_rocauc, xgb_prauc = train_xgboost(X_train, y_train, X_test, y_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    # 2. Train Isolation Forest (unsupervised anomaly detection)
    print("\nTraining Isolation Forest...")
    iso_model, iso_scores, iso_rocauc, iso_prauc = train_isolation_forest(
        X_train, y_train, X_test, y_test
    )

    # 3. Hybrid Score
    print("\nBuilding Hybrid Fraud Score...")
    hybrid_scores = build_hybrid_score(xgb_prob, iso_scores, w1=0.7, w2=0.3)

    # 4. Evaluate Hybrid Model
    evaluate_hybrid(hybrid_scores, y_test, threshold=0.5)


if __name__ == "__main__":
    main()
