import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load the dataset from CSV."""
    df = pd.read_csv(path)
    return df


def scale_amount_time(df):
    """Scale ONLY Amount and Time using StandardScaler."""
    df = df.copy()

    scaler = StandardScaler()
    df[['Amount_scaled', 'Time_scaled']] = scaler.fit_transform(df[['Amount', 'Time']])

    # Drop original unscaled versions
    df = df.drop(columns=['Amount', 'Time'])

    return df


def preprocess_data(df):
    """Preprocess the dataset before model training."""

    # 1. Drop duplicates 
    df = df.drop_duplicates()

    # 2. Scale Amount and Time
    df = scale_amount_time(df)

    return df


def get_train_test(df):
    """Split features (X) and target (y), then train-test split."""

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Train-test split (keep stratify because dataset is imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
