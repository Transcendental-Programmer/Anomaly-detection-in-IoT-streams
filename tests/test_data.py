import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.load_data import NABDataLoader
from src.data.preprocessor import TimeSeriesPreprocessor

@pytest.fixture
def sample_data():
    """Create sample time series data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='5T')
    data = {
        'timestamp': dates,
        'value': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def data_loader():
    """Create a NABDataLoader instance"""
    return NABDataLoader(base_dir="data")

class TestNABDataLoader:
    def test_list_categories(self, data_loader):
        """Test that list_categories returns expected categories"""
        categories = data_loader.list_categories()
        assert isinstance(categories, list)
        assert all(isinstance(cat, str) for cat in categories)
        assert set(categories).issubset({'artificial', 'real', 'aws', 'ad', 'cpu', 'machine', 'nyc', 'tweets'})

    def test_list_datasets(self, data_loader):
        """Test that list_datasets returns datasets for valid category"""
        # Test with specific category
        datasets = data_loader.list_datasets('artificial')
        assert isinstance(datasets, list)
        
        # Test with invalid category
        with pytest.raises(ValueError):
            data_loader.list_datasets('invalid_category')

class TestTimeSeriesPreprocessor:
    def test_preprocess_data(self, sample_data):
        """Test data preprocessing pipeline"""
        preprocessor = TimeSeriesPreprocessor()
        df, sequences = preprocessor.preprocess_data(sample_data)
        
        # Check that output DataFrame has expected columns
        assert 'scaled_value' in df.columns
        assert 'hour' in df.columns
        assert 'day' in df.columns
        assert 'day_of_week' in df.columns
        
        # Check sequences shape
        assert len(sequences.shape) == 3
        assert sequences.shape[1] == 24  # sequence length
        assert sequences.shape[2] == 1   # number of features

    def test_create_sequences(self):
        """Test sequence creation"""
        preprocessor = TimeSeriesPreprocessor()
        data = np.arange(100)
        seq_length = 24
        
        sequences = preprocessor.create_sequences(data, seq_length)
        
        # Check sequences shape
        assert sequences.shape == (76, 24, 1)  # (100 - 24, 24, 1)
        
        # Test with insufficient data
        with pytest.raises(ValueError):
            preprocessor.create_sequences(np.arange(10), seq_length) 