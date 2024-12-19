import json
import logging
from kafka import KafkaProducer
from datetime import datetime
import numpy as np
from ..utils.config import Config

logger = logging.getLogger(__name__)

class IoTDataProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = Config.KAFKA_TOPIC

    def generate_sample_data(self):
        """Generate sample IoT sensor data"""
        return {
            'timestamp': datetime.now().isoformat(),
            'sensor_id': np.random.randint(1, 100),
            'temperature': np.random.normal(25, 5),
            'humidity': np.random.normal(60, 10),
            'pressure': np.random.normal(1013, 10)
        }

    def send_data(self, data=None):
        """Send data to Kafka topic"""
        if data is None:
            data = self.generate_sample_data()
        
        try:
            future = self.producer.send(self.topic, data)
            future.get(timeout=10)
            logger.info(f"Sent data to topic {self.topic}: {data}")
        except Exception as e:
            logger.error(f"Error sending data to Kafka: {e}")

    def close(self):
        """Close the producer connection"""
        self.producer.close() 