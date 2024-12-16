import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.load_data import NABDataLoader
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.anomaly_detector import AnomalyDetector
from src.models.model_evaluation import ModelEvaluator
from src.models.model_manager import ModelManager
import logging

logger = logging.getLogger(__name__)

class Dashboard:
    def __init__(self):
        logger.info("Initializing Dashboard...")
        self.data_loader = NABDataLoader()
        self.preprocessor = TimeSeriesPreprocessor()
        self.detector = None
        self.model_manager = ModelManager()
        
        # Initialize evaluator with None for model - we'll update it later
        self.evaluator = ModelEvaluator(model=None, preprocessor=self.preprocessor)
        self.mlflow_available = self.evaluator.mlflow_available

    def run(self):
        try:
            st.title("IoT Anomaly Detection Dashboard")
            
            # Initialize model with default sequence length if not already initialized
            if self.detector is None:
                default_sequence_length = 24  # Default sequence length
                self.detector = AnomalyDetector(sequence_length=default_sequence_length)
                self.evaluator.model = self.detector
            
            # Sidebar for dataset selection
            st.sidebar.header("Dataset Selection")
            
            # Category selection
            categories = self.data_loader.list_categories()
            selected_category = st.sidebar.selectbox(
                "Select Category",
                categories
            )
            
            # Dataset selection within category
            datasets = self.data_loader.list_datasets(selected_category)
            selected_dataset = st.sidebar.selectbox(
                "Select Dataset",
                datasets
            )
            
            # Load and process data
            if st.sidebar.button("Load Dataset"):
                try:
                    df = self.data_loader.load_dataset(selected_dataset, selected_category)
                    st.success(f"Loaded dataset: {selected_dataset}")
                    
                    # Display raw data
                    st.subheader("Raw Data Preview")
                    st.write(df.head())
                    
                    # Preprocess data
                    processed_df, sequences = self.preprocessor.preprocess_data(df)
                    st.session_state['sequences'] = sequences
                    st.session_state['processed_df'] = processed_df
                    
                    # Display preprocessing info
                    st.subheader("Preprocessing Info")
                    st.write(f"Number of sequences: {len(sequences)}")
                    st.write(f"Sequence length: {sequences.shape[1]}")
                    
                    # Reinitialize model if sequence length doesn't match
                    if sequences.shape[1] != self.detector.sequence_length:
                        self.detector = AnomalyDetector(sequence_length=sequences.shape[1])
                        self.evaluator.model = self.detector
                        st.info(f"Model reinitialized with sequence length: {sequences.shape[1]}")
                        
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
                    logger.error(f"Dataset loading error: {e}", exc_info=True)
                    return
            
            # Model Training Section
            st.header("Model Training")
            if 'sequences' in st.session_state:
                epochs = st.slider("Number of epochs", 10, 100, 50)
                batch_size = st.slider("Batch size", 16, 128, 32)
                
                if st.button("Train Model"):
                    try:
                        with st.spinner('Training model...'):
                            history = self.detector.train(
                                st.session_state['sequences'],
                                epochs=epochs,
                                batch_size=batch_size
                            )
                            st.success("Model trained successfully!")
                            
                            # Log metrics and model to MLflow
                            if self.mlflow_available:
                                try:
                                    metrics = {
                                        'final_loss': float(history.history['loss'][-1]),
                                        'val_loss': float(history.history.get('val_loss', [-1])[-1]),
                                        'epochs': epochs,
                                        'batch_size': batch_size
                                    }
                                    
                                    # Log model parameters
                                    model_params = {
                                        'sequence_length': self.detector.sequence_length,
                                        'model_architecture': str(self.detector.model.get_config())
                                    }
                                    
                                    self.evaluator.log_metrics(metrics, model_params)
                                    st.success("Metrics and model logged to MLflow!")
                                except Exception as e:
                                    st.warning(f"Failed to log to MLflow: {str(e)}")
                                    logger.warning(f"MLflow logging error: {e}", exc_info=True)
                                    
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        logger.error(f"Training error: {e}", exc_info=True)
            
            # Model Evaluation Section
            if self.detector and 'sequences' in st.session_state:
                st.header("Model Evaluation")
                threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.1)
                
                if st.button("Detect Anomalies"):
                    try:
                        anomalies, scores = self.detector.detect_anomalies(
                            st.session_state['sequences'],
                            threshold=threshold
                        )
                        
                        # Store results in session state
                        st.session_state['anomalies'] = anomalies
                        st.session_state['scores'] = scores
                        
                        # Display results
                        st.subheader("Detection Results")
                        st.write(f"Number of anomalies detected: {np.sum(anomalies)}")
                        
                        # Plot results
                        self.plot_results(
                            st.session_state['processed_df'],
                            anomalies,
                            scores
                        )
                        
                    except Exception as e:
                        st.error(f"Error during anomaly detection: {str(e)}")
                        logger.error(f"Detection error: {e}", exc_info=True)
            
            # Model Saving Section
            if self.detector:
                st.header("Save Model")
                model_name = st.text_input("Model Name (optional)")
                
                if st.button("Save Model"):
                    try:
                        self.model_manager.save_model(self.detector, model_name)
                        st.success("Model saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
                        logger.error(f"Model saving error: {e}", exc_info=True)
                        
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            logger.error(f"Dashboard error: {e}", exc_info=True)

    def plot_results(self, df, anomalies, scores):
        """Plot the detection results"""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Plot original data
        fig.add_trace(go.Scatter(
            x=df.index if 'timestamp' not in df else df['timestamp'],
            y=df['value'],
            name='Original Data'
        ))
        
        # Plot anomalies
        if np.any(anomalies):
            # Ensure anomalies array matches DataFrame length
            if len(anomalies) != len(df):
                logger.warning(f"Length mismatch: anomalies ({len(anomalies)}) vs df ({len(df)})")
                # Pad or truncate anomalies array to match df length
                if len(anomalies) < len(df):
                    anomalies = np.pad(anomalies, (0, len(df) - len(anomalies)), 
                                     mode='constant', constant_values=False)
                else:
                    anomalies = anomalies[:len(df)]
                
                # Do the same for scores
                if len(scores) < len(df):
                    scores = np.pad(scores, (0, len(df) - len(scores)),
                                  mode='constant', constant_values=np.nan)
                else:
                    scores = scores[:len(df)]
            
            # Create mask and get anomaly points
            anomaly_points = df[anomalies]
            
            # Plot anomaly points
            fig.add_trace(go.Scatter(
                x=anomaly_points.index if 'timestamp' not in anomaly_points else anomaly_points['timestamp'],
                y=anomaly_points['value'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
            
            # Plot anomaly scores
            fig.add_trace(go.Scatter(
                x=df.index if 'timestamp' not in df else df['timestamp'],
                y=scores,
                name='Anomaly Score',
                yaxis='y2',
                line=dict(color='orange', dash='dot')
            ))
        
        fig.update_layout(
            title='Anomaly Detection Results',
            xaxis_title='Timestamp',
            yaxis_title='Value',
            yaxis2=dict(
                title='Anomaly Score',
                overlaying='y',
                side='right',
                rangemode='tozero'  # Start from zero
            ),
            showlegend=True,
            height=600  # Make plot taller
        )
        
        st.plotly_chart(fig, use_container_width=True)