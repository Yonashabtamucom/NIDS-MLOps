
from pathlib import Path

import numpy as np
import pandas as pd

# Make the experiment reproducible
np.random.seed(42)

# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset output location
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "network_traffic.csv"


def generate_dataset():
    print("Generating synthetic network traffic...")

    # Create 4,000 normal traffic records
    normal = pd.DataFrame({
        "duration": np.random.exponential(2, 4000),
        "protocol": np.random.choice(
            ["tcp", "udp", "icmp"], 4000, p=[0.7, 0.25, 0.05]
        ),
        "src_bytes": np.random.randint(100, 5000, 4000),
        "dst_bytes": np.random.randint(100, 8000, 4000),
        "count": np.random.randint(1, 20, 4000),
        "failed_logins": np.random.choice([0, 1], 4000, p=[0.98, 0.02]),
        "same_srv_rate": np.random.uniform(0.6, 1.0, 4000),
        "label": 0
    })

    # Create 1,000 simulated attack traffic records
    attack = pd.DataFrame({
        "duration": np.random.exponential(1, 1000),
        "protocol": np.random.choice(
            ["tcp", "udp", "icmp"], 1000, p=[0.5, 0.3, 0.2]
        ),
        "src_bytes": np.random.randint(0, 1000, 1000),
        "dst_bytes": np.random.randint(0, 2000, 1000),
        "count": np.random.randint(20, 100, 1000),
        "failed_logins": np.random.choice([0, 1, 2, 3, 4], 1000),
        "same_srv_rate": np.random.uniform(0.0, 0.5, 1000),
        "label": 1
    })

    # Combine normal and attack traffic
    dataset = pd.concat([normal, attack], ignore_index=True)

    # Shuffle the records
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create the output directory if necessary
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    dataset.to_csv(OUTPUT_PATH, index=False)

    print("\nDataset generated successfully!")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Total records: {len(dataset)}")
    print("\nClass distribution:")
    print(dataset["label"].value_counts().sort_index())
    print("\nFirst five records:")
    print(dataset.head())


if __name__ == "__main__":
    generate_dataset()
