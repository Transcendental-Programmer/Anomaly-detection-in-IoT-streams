# setup.py
from setuptools import setup, find_packages

setup(
    name="anomaly_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'streamlit',
        'tensorflow',
        'scikit-learn',
        'plotly',
        'mlflow'
    ]
)