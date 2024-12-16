# src/models/model_evaluation.py
import mlflow
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow.tensorflow
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model, preprocessor):
        self.model = model  # Can be None initially
        self.preprocessor = preprocessor
        try:
            from ..utils.mlflow_config import setup_mlflow
            self.experiment_id = setup_mlflow()
            self.mlflow_available = self.experiment_id is not None
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self.mlflow_available = False
        
    def evaluate(self, df):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not initialized")
            
        _, sequences = self.preprocessor.preprocess_data(df)
        anomalies, scores = self.model.detect_anomalies(sequences)
        
        metrics = {
            'mean_anomaly_score': float(np.mean(scores)),
            'std_anomaly_score': float(np.std(scores)),
            'anomaly_ratio': float(np.mean(anomalies))
        }
        
        return metrics
        
    def log_metrics(self, metrics, model_params=None):
        """Log metrics to MLflow"""
        if not self.mlflow_available:
            logger.warning("MLflow is not available. Metrics will not be logged.")
            return
            
        if self.model is None:
            logger.warning("Model not initialized. Metrics will be logged without model artifact.")
            
        try:
            with mlflow.start_run(experiment_id=self.experiment_id):
                if model_params:
                    mlflow.log_params(model_params)
                mlflow.log_metrics(metrics)
                
                # Log model if available
                if self.model is not None:
                    # Get model signature
                    signature = self.model.get_model_signature()
                    
                    # Create example input
                    example_input = np.random.rand(1, self.model.sequence_length, 1)
                    
                    # Log model with signature
                    mlflow.tensorflow.log_model(
                        self.model.model,
                        "model",
                        registered_model_name="anomaly_detector",
                        signature=signature,
                        input_example=example_input
                    )
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")