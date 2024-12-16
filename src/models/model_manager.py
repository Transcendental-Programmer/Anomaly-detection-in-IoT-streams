# src/models/model_manager.py
import joblib
from pathlib import Path
import json
import datetime

class ModelManager:
    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def save_model(self, model, model_name=None):
        """Save model and metadata"""
        if model_name is None:
            model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        model_dir = self.base_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model.model.save(model_dir / 'model')
        
        # Save metadata
        metadata = {
            'sequence_length': model.sequence_length,
            'created_at': datetime.datetime.now().isoformat(),
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
            
    def load_model(self, model_name):
        """Load model and metadata"""
        model_dir = self.base_dir / model_name
        
        if not model_dir.exists():
            raise ValueError(f"Model {model_name} not found")
            
        # Load metadata
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            
        # Load model
        from tensorflow.keras.models import load_model
        model = load_model(model_dir / 'model')
        
        return model, metadata