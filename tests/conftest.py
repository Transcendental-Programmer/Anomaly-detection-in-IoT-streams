import pytest
import os
import tempfile
from pathlib import Path
import shutil

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def docker_compose_file(test_data_dir):
    """Get the docker-compose file path"""
    return os.path.join(os.path.dirname(__file__), "../docker/docker-compose.yml")

@pytest.fixture(scope="session")
def docker_compose_project_name():
    """Get unique project name for docker-compose"""
    return "anomaly_detection_test" 