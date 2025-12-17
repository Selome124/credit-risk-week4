import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "FraudResult"   # change ONLY if your target column has a different name
ID_COLUMNS = ["TransactionId"]  # drop ID-like columns


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features(df: pd.DataFrame):
    # Drop ID columns if they exist
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # Separate target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Keep only numeric columns (VERY IMPORTANT)
    X = X.select_dtypes(include=["int64", "float64"])

    return X, y


def split_data(X, y, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )
