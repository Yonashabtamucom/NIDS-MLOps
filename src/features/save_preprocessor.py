
from pathlib import Path
import joblib

from preprocess import (
    load_data,
    prepare_features,
    identify_feature_types
)

from build_pipeline import build_preprocessor


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model output directory
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # Step 1: Load datasets
    train_df, test_df = load_data()

    # Step 2: Prepare features and targets
    X_train, X_test, y_train, y_test = prepare_features(
        train_df, test_df
    )

    # Step 3: Identify feature types
    numerical_features, categorical_features = (
        identify_feature_types(X_train)
    )

    # Step 4: Build preprocessing pipeline
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    # Step 5: Fit using training data only
    X_train_processed = preprocessor.fit_transform(X_train)

    # Step 6: Transform testing data
    X_test_processed = preprocessor.transform(X_test)

    # Step 7: Save fitted preprocessor
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"

    joblib.dump(preprocessor, preprocessor_path)

    # Step 8: Verify saved pipeline
    loaded_preprocessor = joblib.load(preprocessor_path)

    test_verification = loaded_preprocessor.transform(X_test)

    # Display results
    print("\nPREPROCESSING PIPELINE RESULTS")

    print("Training input:", X_train.shape)
    print("Testing input:", X_test.shape)

    print("Processed training:", X_train_processed.shape)
    print("Processed testing:", X_test_processed.shape)

    print("Training labels:", y_train.shape)
    print("Testing labels:", y_test.shape)

    print("\nSaved preprocessor:", preprocessor_path)

    print(
        "Verification:",
        test_verification.shape == X_test_processed.shape
    )

    print("\nPreprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
