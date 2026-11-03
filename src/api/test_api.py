
from pathlib import Path
import json
import urllib.request
import urllib.error

import pandas as pd


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "unsw_nb15"
    / "UNSW_NB15_training-set.csv"
)

API_URL = "http://127.0.0.1:8000/predict"


def main():

    print("Loading a real UNSW-NB15 network record...")

    # Read one original record
    df = pd.read_csv(DATA_PATH)
    df = df[df["label"] == 1].head(1)

    # Keep the original 42 input features
    actual_label = int(df.iloc[0]["label"])

    features_df = df.drop(
        columns=["id", "attack_cat", "label"]
    )

    features = json.loads(
        features_df.to_json(orient="records")
    )[0]

    # Create API request
    payload = json.dumps(
        {"features": features}
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print("Sending network traffic to NIDS API...")

    try:
        with urllib.request.urlopen(request) as response:

            result = json.loads(response.read().decode())

            print("\nAPI RESPONSE")
            print("=" * 40)

            print(json.dumps(result, indent=4))

            print("\nACTUAL DATASET LABEL")
            print(
                "Actual:",
                "Attack" if actual_label == 1 else "Normal"
            )

            print("\nAPI TEST COMPLETED SUCCESSFULLY!")

    except urllib.error.HTTPError as error:
        print("HTTP Error:", error.code)
        print(error.read().decode())

    except urllib.error.URLError as error:
        print("Connection Error:", error.reason)


if __name__ == "__main__":
    main()
