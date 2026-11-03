
from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset paths
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "unsw_nb15"

TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"


def load_data():
    """Load the original training and testing datasets."""

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def prepare_features(train_df, test_df):
    """Separate input features from binary target labels."""

    columns_to_remove = ["id", "attack_cat"]
    target_column = "label"

    X_train = train_df.drop(
        columns=columns_to_remove + [target_column]
    )
    y_train = train_df[target_column]

    X_test = test_df.drop(
        columns=columns_to_remove + [target_column]
    )
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test


def identify_feature_types(X_train):
    """Identify numerical and categorical columns."""

    categorical_features = X_train.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    numerical_features = X_train.select_dtypes(
        exclude=["object", "str"]
    ).columns.tolist()

    return numerical_features, categorical_features


def main():

    # Load datasets
    train_df, test_df = load_data()

    # Prepare features and targets
    X_train, X_test, y_train, y_test = prepare_features(
        train_df, test_df
    )

    # Identify feature types
    numerical_features, categorical_features = (
        identify_feature_types(X_train)
    )

    # Display results
    print("DATA PREPARATION RESULTS")

    print("\nTraining features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("\nNumerical features:", len(numerical_features))
    print("Categorical features:", len(categorical_features))

    print("\nCategorical columns:")
    print(categorical_features)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())


if __name__ == "__main__":
    main()
