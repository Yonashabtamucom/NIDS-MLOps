from src.monitoring.logger import log_prediction

# Test 1: Normal traffic
log_prediction(
    prediction=0,
    attack_probability=0.10,
    model_name="NIDS-RandomForest",
    model_version="1"
)

# Test 2: Attack traffic
log_prediction(
    prediction=1,
    attack_probability=0.9954,
    model_name="NIDS-RandomForest",
    model_version="1"
)

print("Monitoring logger test completed!")