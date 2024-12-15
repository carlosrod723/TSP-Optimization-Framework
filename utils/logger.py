import logging
import yaml
from pathlib import Path

def setup_logger(config_path: str = "config/default_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    log_config = config['logging_config']
    
    # Create logs directory if it doesn't exist
    Path(log_config['file']).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_config['format'],
        handlers=[
            logging.FileHandler(log_config['file']),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('tsp_optimizer')