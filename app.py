import streamlit as st
from src.visualization.dashboard import Dashboard
import tensorflow as tf

# Suppress TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main():
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 