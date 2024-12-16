import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    
    # Dataset categories
    DATASET_CATEGORIES = {
        'artificial': 'artificialNoAnomaly',
        'real': 'realTraffic',
        'aws': 'realAWSCloudwatch',
        'ad': 'realAdExchange',
        'cpu': 'realCPU',
        'machine': 'realMachine',
        'nyc': 'realNYCTaxi',
        'tweets': 'realTweets'
    }
    
    # Model parameters
    SEQUENCE_LENGTH = 24
    THRESHOLD = 0.1
    
    # Training parameters
    EPOCHS = 50
    BATCH_SIZE = 32