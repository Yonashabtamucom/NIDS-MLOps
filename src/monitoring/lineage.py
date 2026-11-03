from pathlib import Path
import subprocess
import yaml


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_git_commit():
    """Get the current Git commit ID."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def get_dvc_data_versions():
    """Read dataset version hashes from DVC metadata files."""

    dvc_files = [
        PROJECT_ROOT / "data/raw/unsw_nb15/UNSW-NB15_features.csv.dvc",
        PROJECT_ROOT / "data/raw/unsw_nb15/UNSW_NB15_testing-set.csv.dvc",
        PROJECT_ROOT / "data/raw/unsw_nb15/UNSW_NB15_training-set.csv.dvc",
    ]

    versions = {}

    for dvc_file in dvc_files:

        with open(dvc_file, "r", encoding="utf-8") as file:
            metadata = yaml.safe_load(file)

        dataset_name = dvc_file.stem.replace(".csv", "")

        versions[dataset_name] = metadata["outs"][0]["md5"]

    return versions


if __name__ == "__main__":

    print("GIT COMMIT:")
    print(get_git_commit())

    print("\nDVC DATASET VERSIONS:")

    for dataset, version in get_dvc_data_versions().items():
        print(f"{dataset}: {version}")