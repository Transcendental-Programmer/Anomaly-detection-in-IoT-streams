from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ProgbarLogger
import numpy as np
import streamlit as st
import tensorflow as tf
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

class AnomalyDetector:
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the LSTM model with proper input shape"""
        model = Sequential([
            Input(shape=(self.sequence_length, 1), name='input'),
            LSTM(64, return_sequences=True, name='lstm_1'),
            Dropout(0.2),
            LSTM(32, return_sequences=False, name='lstm_2'),
            Dropout(0.2),
            Dense(1, name='output')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def train(self, sequences, epochs=50, batch_size=32):
        """Train the model"""
        if sequences.shape[1] != self.sequence_length:
            raise ValueError(f"Input sequence length {sequences.shape[1]} does not match model sequence length {self.sequence_length}")
            
        X = sequences[:-1].reshape(-1, self.sequence_length, 1)
        y = sequences[1:, -1].reshape(-1, 1)
        
        return self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
    def detect_anomalies(self, sequences, threshold=0.1):
        """Detect anomalies based on prediction error"""
        # Reshape sequences if needed
        if len(sequences.shape) == 2:
            sequences = sequences.reshape(-1, self.sequence_length, 1)
        
        if len(sequences.shape) != 3:
            raise ValueError(f"Expected 3D input shape (samples, sequence_length, features), got shape {sequences.shape}")
        
        # Make predictions
        X = sequences[:-1]  # All sequences except the last one
        predictions = self.model.predict(X, verbose=0)
        
        # Get actual values (next value after each sequence)
        actual = sequences[1:, -1, 0]  # Get the last value of each sequence except the first
        
        # Ensure predictions and actual values have the same shape
        predictions = predictions.flatten()
        assert len(actual) == len(predictions), f"Shape mismatch: actual {actual.shape} vs predictions {predictions.shape}"
        
        # Calculate reconstruction error
        mse = np.power(actual - predictions, 2)
        
        # Mark anomalies where error exceeds threshold
        anomalies = mse > threshold
        
        # Calculate the original sequence length
        original_length = len(sequences)
        
        # Pad arrays to match original sequence length
        anomalies = np.pad(anomalies, (0, original_length - len(anomalies)), 
                           mode='constant', constant_values=False)
        mse = np.pad(mse, (0, original_length - len(mse)), 
                     mode='constant', constant_values=np.nan)
        
        logger.info(f"Original sequence length: {original_length}, "
                    f"Anomalies length: {len(anomalies)}, "
                    f"MSE length: {len(mse)}")
        
        return anomalies, mse
    
    def get_model_signature(self):
        """Get model signature for MLflow"""
        input_schema = Schema([
            TensorSpec(np.dtype(np.float32), (-1, self.sequence_length, 1), "input")
        ])
        
        output_schema = Schema([
            TensorSpec(np.dtype(np.float32), (-1, 1), "output")
        ])
        
        return ModelSignature(inputs=input_schema, outputs=output_schema)