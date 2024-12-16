# scripts/start_services.py
import subprocess
import sys
import os
from pathlib import Path
import logging
from threading import Thread
from queue import Queue
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def stream_process_output(process, name, output_queue):
    """Stream process output to queue"""
    for line in iter(process.stdout.readline, b''):
        output_queue.put(f"[{name}] {line.decode().strip()}")

def start_services():
    output_queue = Queue()
    mlflow_process = None
    
    # Start MLflow
    logger.info("Starting MLflow server...")
    try:
        mlflow_process = subprocess.Popen(
            ["mlflow", "ui", "--port", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False
        )
        
        # Start thread to monitor MLflow output
        mlflow_thread = Thread(
            target=stream_process_output,
            args=(mlflow_process, "MLflow", output_queue)
        )
        mlflow_thread.daemon = True
        mlflow_thread.start()
        
    except Exception as e:
        logger.error(f"Error starting MLflow: {str(e)}")
        sys.exit(1)
    
    # Wait a bit for MLflow to start
    time.sleep(2)
    
    # Start Streamlit
    logger.info("Starting Streamlit dashboard...")
    try:
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        
        # Start thread to monitor Streamlit output
        streamlit_thread = Thread(
            target=stream_process_output,
            args=(streamlit_process, "Streamlit", output_queue)
        )
        streamlit_thread.daemon = True
        streamlit_thread.start()
        
    except Exception as e:
        logger.error(f"Error starting Streamlit: {str(e)}")
        if mlflow_process:
            mlflow_process.terminate()
        sys.exit(1)
    
    logger.info("All services are running. Press Ctrl+C to stop.")
    
    # Monitor and log process outputs
    try:
        while True:
            try:
                output = output_queue.get(timeout=1)
                logger.info(output)
            except:
                continue
    except KeyboardInterrupt:
        logger.info("\nShutting down services...")
        if mlflow_process:
            mlflow_process.terminate()
        streamlit_process.terminate()
        logger.info("Services stopped successfully.")

if __name__ == "__main__":
    start_services()