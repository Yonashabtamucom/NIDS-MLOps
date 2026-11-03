
import mlflow
from mlflow import MlflowClient


# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Model registration details
model_name = "NIDS-RandomForest"

# Latest successful Random Forest run
run_id = "7a39ddbfee1e4fa198855541be19efc4"
model_uri = f"runs:/{run_id}/model"

# Register the model
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
)

print("\nMODEL REGISTRATION SUCCESSFUL")
print("--------------------------------")
print("Model name:", registered_model.name)
print("Model version:", registered_model.version)
print("Run ID:", run_id)
print("Model URI:", model_uri)
