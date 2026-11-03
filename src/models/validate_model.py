
from pathlib import Path

import mlflow
import pandas as pd

from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

# MLflow connection
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Registered model version
MODEL_URI = "models:/NIDS-RandomForest/1"


def main():

    print("Loading registered Random Forest model...")

    # Load registered model from MLflow
    model = mlflow.sklearn.load_model(MODEL_URI)

    print("Model loaded successfully!")

    # Load test data
    X_test = load_npz(DATA_DIR / "X_test.npz")

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    )["label"]

    print("\nTesting data:", X_test.shape)

    # Generate predictions
    predictions = model.predict(X_test)

    # Evaluate predictions
    print("\nREGISTERED MODEL VALIDATION")
    print("=" * 40)

    print(
        "Accuracy:",
        accuracy_score(y_test, predictions)
    )

    print(
        "Precision:",
        precision_score(y_test, predictions)
    )

    print(
        "Recall:",
        recall_score(y_test, predictions)
    )

    print(
        "F1-score:",
        f1_score(y_test, predictions)
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Basic prediction checks
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})

    print("\nPASS: Prediction count is correct.")
    print("PASS: Predictions contain valid labels.")
    print("\nREGISTERED MODEL VALIDATION COMPLETED!")


if __name__ == "__main__":
    main()
