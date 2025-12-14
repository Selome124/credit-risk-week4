import pandas as pd
import numpy as np


def build_features(df):
    """
    Task 3 & Task 4:
    - Separates target variable (is_high_risk) if present
    - Returns model-ready feature matrix X (DataFrame) and target vector y (array)
    """

    df = df.copy()

    # --------------------
    # TARGET EXTRACTION (Task 4)
    # --------------------
    if "is_high_risk" in df.columns:
        y = df["is_high_risk"].values
        df = df.drop(columns=["is_high_risk"])
    else:
        y = None

    # --------------------
    # FEATURE MATRIX (Task 3)
    # --------------------
    # Return X as DataFrame with column names (important for sklearn transformers)
    X = df

    return X, y


def apply_woe(X, y):
    """
    Placeholder for WoE transformation.
    Can be extended later using xverse or woe libraries.
    """
    return X


def main():
    """
    Test run for feature engineering pipeline
    """
    df = pd.read_csv("data/raw/data_with_target.csv")

    X_transformed, y = build_features(df)

    print("Feature matrix shape:", X_transformed.shape)

    if y is not None:
        print("Target vector shape:", y.shape)
    else:
        print("No target vector found")

    print("\nFirst 5 rows of feature matrix:")
    print(X_transformed.head())

    if y is not None:
        X_woe = apply_woe(X_transformed, y)
        print("\nWoE-transformed feature matrix shape:", X_woe.shape)
        print(X_woe.head())


if __name__ == "__main__":
    main()
