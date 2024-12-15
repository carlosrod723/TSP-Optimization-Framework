# main.py
import logging
from pathlib import Path
from utils.logger import setup_logger
from utils.config import load_config

def init_project():
    """Initialize project structure and configurations."""
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed',
        'algorithms',
        'models',
        'utils',
        'tests',
        'logs',
        'results/visualizations',
        'notebooks',
        'docs',
        'config',
        'scripts'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = setup_logger()
    logger.info("Project structure initialized successfully")
    
    return logger

if __name__ == "__main__":
    logger = init_project()
    logger.info("TSP Optimizer initialization complete")