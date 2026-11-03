from src.monitoring.logger import log_prediction
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

MODEL_NAME = "NIDS-RandomForest"
MODEL_ALIAS = "champion"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# ==========================================
# 2. LOAD REGISTERED MODEL AND PREPROCESSOR
# ==========================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Connect to MLflow Model Registry
client = mlflow.MlflowClient()

print("Loading registered NIDS model...")

# Find the version currently assigned to champion
registered_version = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS,
)

MODEL_VERSION = registered_version.version

# Load the model using its alias
model = mlflow.sklearn.load_model(MODEL_URI)

print(f"Loaded {MODEL_NAME}, version {MODEL_VERSION}")

preprocessor = joblib.load(PREPROCESSOR_PATH)

# Original feature names and their expected order
FEATURE_NAMES = list(preprocessor.feature_names_in_)

print("Model and preprocessor loaded successfully!")


# ==========================================
# 3. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="NIDS - Network Intrusion Detection API",
    description=(
        "An ML-powered API for classifying network "
        "traffic as normal or malicious."
    ),
    version="1.0.0",
)


# ==========================================
# 4. DEFINE INPUT DATA STRUCTURE
# ==========================================

class PredictionRequest(BaseModel):

    features: dict[str, Any] = Field(
        ...,
        description="The original 42 UNSW-NB15 network features."
    )


# ==========================================
# 5. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "NIDS API is running!",
        "model": "NIDS-RandomForest",
        "version": str(MODEL_VERSION),
        "status": "ready",
    }


# ==========================================
# 6. MODEL INFORMATION ENDPOINT
# ==========================================

@app.get("/model-info")
def model_info():

    return {
        "model_name": "NIDS-RandomForest",
        "model_version": str(MODEL_VERSION),
        "input_features": len(FEATURE_NAMES),
        "prediction_classes": {
            "0": "Normal",
            "1": "Attack",
        },
    }


# ==========================================
# 7. NETWORK INTRUSION PREDICTION
# ==========================================

@app.post("/predict")
def predict(request: PredictionRequest):

    received_features = request.features

    # Check for missing features
    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in received_features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required features",
                "missing_features": missing_features,
            },
        )

    # Check for unexpected features
    extra_features = [
        feature
        for feature in received_features
        if feature not in FEATURE_NAMES
    ]

    if extra_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unexpected features",
                "extra_features": extra_features,
            },
        )

    try:

        # Preserve the original training feature order
        input_df = pd.DataFrame(
            [received_features],
            columns=FEATURE_NAMES,
        )

        # Apply the saved preprocessing pipeline
        processed_data = preprocessor.transform(input_df)

        # Generate prediction
        prediction = int(model.predict(processed_data)[0])

        # Calculate attack probability
        probabilities = model.predict_proba(processed_data)[0]

        attack_index = list(model.classes_).index(1)

        attack_probability = float(
            probabilities[attack_index]
        )

                # Record the prediction for monitoring
        log_prediction(
            prediction=prediction,
            attack_probability=round(attack_probability, 4),
            model_name="NIDS-RandomForest",
            model_version=str(MODEL_VERSION),
        )
        # Return the prediction to the API user
        return {
            "prediction": prediction,
            "classification": (
                "Attack" if prediction == 1 else "Normal"
            ),
            "attack_probability": round(
                attack_probability, 4
            ),
            "model": "NIDS-RandomForest",
            "model_version": str(MODEL_VERSION),        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(error)}",
        )
