#automatically checks whether the dataset meets our requirements before the ML pipeline uses it.





from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "unsw_nb15"

TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"


def validate_dataset():

    print("=" * 60)
    print("UNSW-NB15 DATA VALIDATION")
    print("=" * 60)

    # 1. Check files
    for path in [TRAIN_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    # 2. Load datasets
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # 3. Check shapes
    print("\n1. DATASET SHAPES")
    print("Training:", train.shape)
    print("Testing:", test.shape)

    # 4. Check columns
    print("\n2. COLUMN CONSISTENCY")

    if list(train.columns) != list(test.columns):
        raise ValueError("Training and testing columns do not match.")

    print("PASS: Columns match.")

    # 5. Check missing values
    print("\n3. MISSING VALUES")

    if train.isnull().sum().sum() > 0:
        raise ValueError("Training data contains missing values.")

    if test.isnull().sum().sum() > 0:
        raise ValueError("Testing data contains missing values.")

    print("PASS: No missing values.")

    # 6. Check duplicates
    print("\n4. DUPLICATES")

    print("Training:", train.duplicated().sum())
    print("Testing:", test.duplicated().sum())

    # 7. Check target labels
    print("\n5. TARGET LABELS")

    expected_labels = {0, 1}

    if set(train["label"].unique()) != expected_labels:
        raise ValueError("Unexpected training labels.")

    if set(test["label"].unique()) != expected_labels:
        raise ValueError("Unexpected testing labels.")

    print("PASS: Both datasets contain labels 0 and 1.")

    # 8. Final result
    print("\n" + "=" * 60)
    print("DATA VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        validate_dataset()
    except Exception as error:
        print(f"\nDATA VALIDATION FAILED: {error}")
        sys.exit(1)
