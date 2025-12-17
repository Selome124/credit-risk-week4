import pandas as pd
import numpy as np

from src.data_processing import build_features


def test_build_features_returns_X_and_y():
    df = pd.DataFrame({
        "amount": [100, 200, 300],
        "frequency": [1, 2, 3],
        "FraudResult": [0, 1, 0]  # Add this target column here
    })

    X, y = build_features(df)
    # your existing assertions here


def test_target_column_removed_from_features():
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "FraudResult": [0, 1, 0]  # Add this target column here
    })

    X, y = build_features(df)
    # your existing assertions here
