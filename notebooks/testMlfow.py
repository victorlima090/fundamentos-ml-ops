import mlflow

# Print connection information
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active Experiment: {mlflow.get_experiment_by_name('wine-quality-experiments')}")
import mlflow
with mlflow.start_run():
    mlflow.log_metric("test_metric", 1.23)