import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from src.monitoring.lineage import (
    get_git_commit,
    get_dvc_data_versions,)

from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. MLFLOW CONFIGURATION
# --------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("NIDS-UNSW-NB15")


# --------------------------------------------------
# 3. LOAD PROCESSED DATA
# --------------------------------------------------

def load_processed_data():

    X_train = load_npz(DATA_DIR / "X_train.npz")
    X_test = load_npz(DATA_DIR / "X_test.npz")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["label"]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["label"]

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 4. TRAIN, EVALUATE AND TRACK A MODEL
# --------------------------------------------------


def train_and_log_model(
    model,
    model_name,
    parameters,
    X_train,
    X_test,
    y_train,
    y_test,
):

    with mlflow.start_run(run_name=model_name) as run:

        # 1. Capture code and dataset versions
        git_commit = get_git_commit()
        dataset_versions = get_dvc_data_versions()

        # 2. Record lineage information in MLflow
        mlflow.set_tag("git_commit", git_commit)

        for dataset, version in dataset_versions.items():
            mlflow.set_tag(
                f"data_version_{dataset}",
                version,
            )

        print(f"\nTraining {model_name}...")

        # 3. Record model parameters
        mlflow.log_params(parameters)

        # 4. Train the model
        model.fit(X_train, y_train)

        # 5. Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # 6. Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        # 7. Record metrics in MLflow
        mlflow.log_metrics(metrics)

        # 8. Create classification report and confusion matrix
        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
        )

        # 9. Record evaluation results as artifacts
        mlflow.log_dict(
            report,
            "classification_report.json",
        )

        mlflow.log_dict(
            {"confusion_matrix": matrix.tolist()},
            "confusion_matrix.json",
        )

        # 10. Record additional information
        mlflow.set_tag("dataset", "UNSW-NB15")
        mlflow.set_tag("task", "Binary Network Intrusion Detection")
        mlflow.set_tag("model_type", model_name)

        # 11. Log the trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # 12. Save a local copy
        model_filename = (
            model_name.lower().replace(" ", "_") + ".joblib"
        )

        joblib.dump(
            model,
            MODEL_DIR / model_filename,
        )

        # 13. Display results
        print(f"\n{model_name} RESULTS")
        print("-" * 45)

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print("\nConfusion Matrix:")
        print(matrix)

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        print("\nMLflow Run ID:", run.info.run_id)
        print("Saved local model:", model_filename)

    return metrics

# --------------------------------------------------
# 5. MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    print("Loading processed UNSW-NB15 data...")

    X_train, X_test, y_train, y_test = load_processed_data()

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    print("\nMLflow tracking URI:", mlflow.get_tracking_uri())
    print("MLflow experiment:", "NIDS-UNSW-NB15")

    # Model 1: Logistic Regression
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    logistic_parameters = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    }

    logistic_metrics = train_and_log_model(
        logistic_model,
        "Logistic Regression",
        logistic_parameters,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Model 2: Random Forest
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    random_forest_parameters = {
        "n_estimators": 100,
        "max_depth": 20,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    random_forest_metrics = train_and_log_model(
        random_forest_model,
        "Random Forest",
        random_forest_parameters,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nMODEL COMPARISON")
    print("=" * 45)

    print("\nLogistic Regression:")
    for name, value in logistic_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nRandom Forest:")
    for name, value in random_forest_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nSUCCESS: Both experiments logged to MLflow!")


if __name__ == "__main__":
    main()
