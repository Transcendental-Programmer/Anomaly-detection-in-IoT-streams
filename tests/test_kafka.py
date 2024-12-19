import pytest
import json
from src.data.kafka_producer import IoTDataProducer
from src.data.kafka_consumer import IoTDataConsumer
from kafka import KafkaAdminClient
from kafka.admin import NewTopic
import time

@pytest.fixture(scope="session")
def kafka_topic():
    """Create a test topic"""
    admin_client = KafkaAdminClient(bootstrap_servers='localhost:9092')
    topic = NewTopic(name='test_topic', num_partitions=1, replication_factor=1)
    admin_client.create_topics([topic])
    yield 'test_topic'
    admin_client.delete_topics(['test_topic'])

@pytest.fixture
def producer():
    """Create a producer instance"""
    producer = IoTDataProducer()
    yield producer
    producer.close()

@pytest.fixture
def consumer():
    """Create a consumer instance"""
    consumer = IoTDataConsumer()
    yield consumer
    consumer.consumer.close()

def test_producer_send_data(producer, kafka_topic):
    """Test sending data through producer"""
    test_data = {
        'sensor_id': 1,
        'temperature': 25.0,
        'humidity': 60.0,
        'pressure': 1013.0
    }
    producer.send_data(test_data)
    time.sleep(1)  # Give time for message to be sent

def test_consumer_receive_data(producer, consumer, kafka_topic):
    """Test receiving data through consumer"""
    test_data = producer.generate_sample_data()
    producer.send_data(test_data)
    
    # Get one message
    message = next(consumer.consumer)
    assert message.value is not None
    
    # Check message structure
    data = message.value
    assert 'sensor_id' in data
    assert 'temperature' in data
    assert 'humidity' in data
    assert 'pressure' in data 