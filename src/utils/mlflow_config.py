# src/utils/mlflow_config.py
import mlflow
import os
from pathlib import Path
import tempfile

def setup_mlflow():
    """Setup MLflow configuration"""
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent.absolute()
    
    # Create mlruns directory in project root
    mlruns_dir = project_root / 'mlruns'
    mlruns_dir.mkdir(exist_ok=True)
    
    # Create artifacts directory
    artifacts_dir = project_root / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    
    # Use SQLite database in project root
    db_path = project_root / 'mlflow.db'
    tracking_uri = f'sqlite:///{db_path}'
    
    # Format the artifact location for Windows
    artifact_location = str(artifacts_dir).replace('\\', '/')
    if not artifact_location.startswith('/'):
        artifact_location = '/' + artifact_location
    artifact_location = f'file:{artifact_location}'
    
    # Set MLflow configuration
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    os.environ['MLFLOW_ARTIFACT_ROOT'] = artifact_location
    
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment
    experiment_name = "anomaly_detection"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Format the experiment artifact location
            exp_artifact_location = str(artifacts_dir / experiment_name).replace('\\', '/')
            if not exp_artifact_location.startswith('/'):
                exp_artifact_location = '/' + exp_artifact_location
            exp_artifact_location = f'file:{exp_artifact_location}'
            
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=exp_artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
            
        # Create experiment-specific artifact directory
        (artifacts_dir / experiment_name).mkdir(exist_ok=True)
        
        return experiment_id
        
    except Exception as e:
        print(f"Warning: MLflow initialization failed: {e}")
        return None