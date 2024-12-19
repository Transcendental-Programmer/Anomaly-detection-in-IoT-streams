import json
import logging
from kafka import KafkaConsumer
from ..utils.config import Config
from ..models.anomaly_detector import AnomalyDetector
import mlflow
import numpy as np

logger = logging.getLogger(__name__)

class IoTDataConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            Config.KAFKA_TOPIC,
            bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        self.detector = AnomalyDetector()
        self.buffer = []
        self.buffer_size = 100  # Number of data points to buffer before processing

    def process_message(self, message):
        """Process a single message"""
        try:
            data = message.value
            self.buffer.append(data)

            if len(self.buffer) >= self.buffer_size:
                self.process_buffer()

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def process_buffer(self):
        """Process buffered data for anomaly detection"""
        try:
            # Convert buffer to sequences
            sequences = self.prepare_sequences(self.buffer)
            
            # Detect anomalies
            anomalies, scores = self.detector.detect_anomalies(sequences)
            
            # Log results to MLflow
            with mlflow.start_run(run_name="anomaly_detection"):
                mlflow.log_metrics({
                    "anomaly_ratio": float(anomalies.mean()),
                    "mean_score": float(scores.mean()),
                    "max_score": float(scores.max())
                })
            
            # Clear buffer
            self.buffer = []
            
        except Exception as e:
            logger.error(f"Error processing buffer: {e}")

    def prepare_sequences(self, data):
        """Prepare data sequences for the model"""
        # Implementation depends on your specific data structure
        # This is a placeholder
        return np.array(data)

    def start_consuming(self):
        """Start consuming messages"""
        try:
            logger.info("Starting to consume messages...")
            for message in self.consumer:
                self.process_message(message)
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.consumer.close() 