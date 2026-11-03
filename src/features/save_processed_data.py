
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, issparse, save_npz

from preprocess import (
    load_data,
    prepare_features,
    identify_feature_types,
)

from build_pipeline import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # 1. Load the original UNSW-NB15 datasets
    train_df, test_df = load_data()

    # 2. Separate features and target labels
    X_train, X_test, y_train, y_test = prepare_features(
        train_df, test_df
    )

    # 3. Identify numerical and categorical features
    numerical_features, categorical_features = identify_feature_types(
        X_train
    )

    # 4. Build the preprocessing pipeline
    preprocessor = build_preprocessor(
        numerical_features, categorical_features
    )

    # 5. Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # 6. Transform test data without fitting again
    X_test_processed = preprocessor.transform(X_test)

    # 7. Convert results into CSR sparse matrices
    X_train_sparse = (
        X_train_processed.tocsr()
        if issparse(X_train_processed)
        else csr_matrix(X_train_processed)
    )

    X_test_sparse = (
        X_test_processed.tocsr()
        if issparse(X_test_processed)
        else csr_matrix(X_test_processed)
    )

    # 8. Save processed features
    save_npz(
        PROCESSED_DIR / "X_train.npz",
        X_train_sparse
    )

    save_npz(
        PROCESSED_DIR / "X_test.npz",
        X_test_sparse
    )

    # 9. Save target labels
    pd.Series(y_train, name="label").to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False
    )

    pd.Series(y_test, name="label").to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False
    )

    # 10. Save the fitted preprocessing pipeline
    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.joblib"
    )

    # 11. Verify saved files
    X_train_check = __import__(
        "scipy.sparse", fromlist=["load_npz"]
    ).load_npz(PROCESSED_DIR / "X_train.npz")

    X_test_check = __import__(
        "scipy.sparse", fromlist=["load_npz"]
    ).load_npz(PROCESSED_DIR / "X_test.npz")

    print("\nPROCESSED DATA SAVING RESULTS")
    print("--------------------------------")
    print("Training features:", X_train_check.shape)
    print("Testing features:", X_test_check.shape)
    print("Training labels:", len(y_train))
    print("Testing labels:", len(y_test))
    print("Training format:", type(X_train_check).__name__)
    print("Testing format:", type(X_test_check).__name__)

    print("\nSaved files:")
    print("X_train.npz")
    print("X_test.npz")
    print("y_train.csv")
    print("y_test.csv")
    print("preprocessor.joblib")

    print("\nSUCCESS: Processed data saved and verified!")


if __name__ == "__main__":
    main()
