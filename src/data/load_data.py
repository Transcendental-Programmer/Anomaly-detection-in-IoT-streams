# src/data/load_data.py
import pandas as pd
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

class NABDataLoader:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.dataset_categories = {
            'artificial': 'artificialNoAnomaly',
            'real': 'realTraffic',
            'aws': 'realAWSCloudwatch',
            'ad': 'realAdExchange',
            'cpu': 'realCPU',
            'machine': 'realMachine',
            'nyc': 'realNYCTaxi',
            'tweets': 'realTweets'
        }

    def list_categories(self):
        """Returns available dataset categories"""
        return list(self.dataset_categories.keys())

    def list_datasets(self, category=None):
        """
        Lists available datasets, optionally filtered by category.
        
        Args:
            category (str, optional): Dataset category to filter by
            
        Returns:
            dict: Dictionary of category: [datasets] if no category specified,
                 or list of datasets for specified category
        """
        if category and category not in self.dataset_categories:
            raise ValueError(f"Invalid category: {category}")
            
        result = {}
        categories = [category] if category else self.dataset_categories.keys()
        
        for cat in categories:
            cat_dir = self.base_dir / self.dataset_categories[cat]
            if cat_dir.exists():
                datasets = [f.stem for f in cat_dir.glob("*.csv")]
                result[cat] = sorted(datasets)
                
        return result[category] if category else result

    def load_dataset(self, dataset_name, category):
        """
        Loads a specific dataset from a category.
        
        Args:
            dataset_name (str): Name of the dataset to load
            category (str): Category of the dataset
            
        Returns:
            pandas.DataFrame: The loaded dataset
        """
        if category not in self.dataset_categories:
            raise ValueError(f"Invalid category: {category}")
            
        file_path = self.base_dir / self.dataset_categories[category] / f"{dataset_name}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
            
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df