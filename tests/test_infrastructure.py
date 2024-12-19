import pytest
import requests
import time
from pymongo import MongoClient
import pika

def is_responsive(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False

@pytest.mark.timeout(30)
def test_mongodb_connection(docker_services):
    """Test MongoDB container is running and accessible"""
    port = docker_services.port_for("mongodb", 27017)
    url = f"mongodb://localhost:{port}"
    
    # Wait for MongoDB to be responsive
    time.sleep(5)  # Give MongoDB time to start
    
    client = MongoClient(url)
    assert client.server_info()  # Will raise error if can't connect

@pytest.mark.timeout(30)
def test_rabbitmq_connection(docker_services):
    """Test RabbitMQ container is running and accessible"""
    port = docker_services.port_for("rabbitmq", 5672)
    
    # Wait for RabbitMQ to be responsive
    time.sleep(10)  # RabbitMQ needs more time to start
    
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost', port=port)
    )
    assert connection.is_open
    connection.close()

@pytest.mark.timeout(30)
def test_mlflow_ui_accessible(docker_services):
    """Test MLflow UI is accessible"""
    port = docker_services.port_for("mlflow", 5000)
    url = f"http://localhost:{port}"
    
    # Wait for MLflow to be responsive
    time.sleep(5)
    
    assert is_responsive(url)

@pytest.mark.timeout(30)
def test_grafana_ui_accessible(docker_services):
    """Test Grafana UI is accessible"""
    port = docker_services.port_for("grafana", 3000)
    url = f"http://localhost:{port}"
    
    # Wait for Grafana to be responsive
    time.sleep(5)
    
    assert is_responsive(url) 