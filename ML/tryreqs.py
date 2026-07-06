import mlflow
import tempfile
import os
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

path=os.getenv("ARTIFACT_PATH")
tracking=os.getenv("MLFLOW_TRACKING_URI")

# Set your PostgreSQL tracking URI
mlflow.set_tracking_uri(tracking)
mlflow.create_experiment(
    "try_path_experiment",
    artifact_location= path  # Windows
)

# Set the experiment (make sure it exists with the right artifact location)
mlflow.set_experiment("try_path_experiment")

with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")
    
    # Log some parameters and metrics
    mlflow.log_param("test_param", 42)
    mlflow.log_metric("test_metric", 0.95)
    
    # Create a dummy file to log as artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test artifact")
        temp_file_path = f.name
    
    # Log the artifact
    mlflow.log_artifact(temp_file_path, artifact_path="test_folder")
    
    # Also log a model (optional)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)
    mlflow.sklearn.log_model(model, "test_model")
    
    # Clean up temp file
    os.unlink(temp_file_path)
    
    print(f"✅ Artifacts should be in: {run.info.artifact_uri}")