import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, df):
        """Preprocess time series data"""
        # Ensure DataFrame has an index
        if df.index.empty:
            df.index = pd.RangeIndex(len(df))
            
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Ensure value column is numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Handle missing values
        df = df.dropna(subset=['value'])
        
        # Scale the value column
        scaled_values = self.scaler.fit_transform(df[['value']])
        df['scaled_value'] = scaled_values
        
        # Create sequences for time series analysis
        sequence_length = 24  # 2-hour sequences (assuming 5-min intervals)
        sequences = self.create_sequences(df['scaled_value'].values, sequence_length)
        
        return df, sequences
        
    def create_sequences(self, data, seq_length):
        """Create sequences for time series prediction"""
        if len(data) <= seq_length:
            raise ValueError(f"Data length ({len(data)}) must be greater than sequence length ({seq_length})")
            
        sequences = []
        for i in range(len(data) - seq_length):
            sequence = data[i:(i + seq_length)]
            sequences.append(sequence)
            
        sequences = np.array(sequences)
        
        # Ensure proper shape for LSTM (samples, sequence_length, features)
        if len(sequences.shape) == 2:
            sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
            
        return sequences