import csv
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "reports" / "monitoring"
LOG_FILE = LOG_DIR / "predictions.csv"


def log_prediction(
    prediction,
    attack_probability,
    model_name,
    model_version
):
    print(f"DEBUG: Logger received model version = {model_version}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "prediction",
                "classification",
                "attack_probability",
                "model_name",
                "model_version"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            prediction,
            "Attack" if prediction == 1 else "Normal",
            attack_probability,
            model_name,
            model_version
        ])

    print("Prediction recorded successfully.")