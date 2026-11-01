
from pathlib import Path
import pandas as pd

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "unsw_nb15"

TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"
FEATURES_PATH = DATA_DIR / "UNSW-NB15_features.csv"


def inspect_dataset():

    print("=" * 60)
    print("UNSW-NB15 DATASET INSPECTION")
    print("=" * 60)

    # Check files
    for file_path in [TRAIN_PATH, TEST_PATH, FEATURES_PATH]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    # Load datasets
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    features = pd.read_csv(FEATURES_PATH,encoding="cp1252")

    # Clean column-name whitespace
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    # 1. Dataset sizes
    print("\n1. DATASET SHAPES")
    print("Training:", train.shape)
    print("Testing:", test.shape)
    print("Feature descriptions:", features.shape)

    # 2. Training columns
    print("\n2. TRAINING COLUMNS")
    for column in train.columns:
        print("-", column)

    # 3. Target distribution
    print("\n3. TRAINING LABEL DISTRIBUTION")
    print(train["label"].value_counts(dropna=False))

    # 4. Attack categories
    if "attack_cat" in train.columns:
        print("\n4. ATTACK CATEGORIES")
        print(train["attack_cat"].value_counts(dropna=False))

    # 5. Missing values
    print("\n5. MISSING VALUES")

    print("\nTraining missing values:")
    print(train.isnull().sum().to_string())

    print("\nTesting missing values:")
    print(test.isnull().sum().to_string())

    print("\n6. LABEL DISTRIBUTION IN TEST DATA")
    print(test["label"].value_counts().sort_index())

    print("\n7. TRAINING AND TESTING COLUMN CHECK")

    print("Same columns:", list(train.columns) == list(test.columns))

    print("Training columns:", len(train.columns))
    print("Testing columns:", len(test.columns))

    # 6. Duplicate records
    print("\n6. DUPLICATE RECORDS")
    print("Training duplicates:", train.duplicated().sum())
    print("Testing duplicates:", test.duplicated().sum())

    # 7. Data types
    print("\n7. DATA TYPES")
    print(train.dtypes)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
