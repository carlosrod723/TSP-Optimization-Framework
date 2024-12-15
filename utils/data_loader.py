# Import necessary libraries and packages
import numpy as np
import urllib.request
import tsplib95
from pathlib import Path

def download_tsplib_instance(name: str, save_dir: str = "data/raw") -> str:
    """Download a specific instance from TSPLIB."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    base_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
    filename = f"{name}.tsp"
    save_path = f"{save_dir}/{filename}"
    
    if not Path(save_path).exists():
        url = f"{base_url}{filename}"
        urllib.request.urlretrieve(url, save_path)
    
    return save_path