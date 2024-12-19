import pytest
import numpy as np
from src.models.anomaly_detector import AnomalyDetector
from src.models.model_evaluation import ModelEvaluator
from src.data.preprocessor import TimeSeriesPreprocessor

@pytest.fixture
def sample_sequences():
    """Create sample sequences for testing"""
    return np.random.rand(100, 24, 1)

@pytest.fixture
def model():
    """Create an AnomalyDetector instance"""
    return AnomalyDetector(sequence_length=24)

class TestAnomalyDetector:
    def test_model_initialization(self, model):
        """Test model architecture initialization"""
        assert model.sequence_length == 24
        assert model.model is not None
        
        # Check model layers
        layers = model.model.layers
        assert len(layers) == 5  # Input, LSTM, Dropout, LSTM, Dense
        
        # Check input shape
        assert model.model.input_shape == (None, 24, 1)
        
    def test_train(self, model, sample_sequences):
        """Test model training"""
        history = model.train(sample_sequences, epochs=2, batch_size=32)
        assert 'loss' in history.history
        assert len(history.history['loss']) == 2
        
    def test_detect_anomalies(self, model, sample_sequences):
        """Test anomaly detection"""
        # Train model first
        model.train(sample_sequences, epochs=2, batch_size=32)
        
        # Test detection
        anomalies, scores = model.detect_anomalies(sample_sequences, threshold=0.1)
        
        assert len(anomalies) == len(sample_sequences)
        assert len(scores) == len(sample_sequences)
        assert anomalies.dtype == bool
        assert np.all(np.isfinite(scores[~np.isnan(scores)]))

class TestModelEvaluator:
    def test_evaluation(self):
        """Test model evaluation"""
        model = AnomalyDetector(sequence_length=24)
        preprocessor = TimeSeriesPreprocessor()
        evaluator = ModelEvaluator(model, preprocessor)
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5T')
        data = {
            'timestamp': dates,
            'value': np.random.rand(100)
        }
        df = pd.DataFrame(data)
        
        # Train model
        _, sequences = preprocessor.preprocess_data(df)
        model.train(sequences, epochs=2, batch_size=32)
        
        # Test evaluation
        metrics = evaluator.evaluate(df)
        assert 'mean_anomaly_score' in metrics
        assert 'std_anomaly_score' in metrics
        assert 'anomaly_ratio' in metrics 