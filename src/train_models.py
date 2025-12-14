import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_processing import build_features


def evaluate_model(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }


def main():
    # -------------------------
    # Load and prepare data
    # -------------------------
    df = pd.read_csv("data/raw/data_with_target.csv")

    # Get feature matrix X and target vector y
    X, y = build_features(df)

    if X is None or y is None:
        raise ValueError(
            "build_features() must return both X and y. "
            "Check that is_high_risk is created correctly."
        )

    # IMPORTANT: You need the column names to do column-based preprocessing
    # Assuming build_features drops 'is_high_risk' but returns X as NumPy array,
    # we'll use the dataframe columns except target for ColumnTransformer
    feature_names = df.drop(columns=["is_high_risk"]).columns.tolist()

    # Identify categorical and numerical columns based on original dataframe
    categorical_cols = [col for col in feature_names if df[col].dtype == "object"]
    numerical_cols = [col for col in feature_names if col not in categorical_cols]

    # Split data before fitting any transformers
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # -------------------------
    # Define models
    # -------------------------
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params": {
                "model__C": [0.01, 0.1, 1, 10]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None]
            }
        }
    }

    mlflow.set_experiment("credit-risk-models")

    for model_name, cfg in models.items():
        with mlflow.start_run(run_name=model_name):

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", cfg["model"])
            ])

            grid = GridSearchCV(
                pipeline,
                cfg["params"],
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                error_score='raise'  # Helps debug if fit fails
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_prob)

            # -------------------------
            # MLflow logging
            # -------------------------
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, "model")

            print(f"\n{model_name} results:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
