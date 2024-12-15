# large_scale_handlers/custom_data_loader.py

import numpy as np
from typing import List, Tuple, Union
import pandas as pd
import logging
from pathlib import Path

class CustomDataLoader:
    """Handles loading and preprocessing of custom TSP datasets."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_distance_matrix(
        self,
        file_path: str,
        file_type: str = 'auto'
    ) -> np.ndarray:
        """
        Load distance matrix from various file formats.
        
        Args:
            file_path: Path to the data file
            file_type: Type of file ('csv', 'npy', 'txt', or 'auto' for automatic detection)
        
        Returns:
            np.ndarray: Distance matrix
        """
        file_path = Path(file_path)
        
        if file_type == 'auto':
            file_type = file_path.suffix[1:]
        
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path).values
            elif file_type == 'npy':
                return np.load(file_path)
            elif file_type == 'txt':
                return np.loadtxt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def load_coordinates(
        self,
        file_path: str,
        file_type: str = 'auto'
    ) -> List[Tuple[float, float]]:
        """
        Load coordinates from file.
        
        Args:
            file_path: Path to the coordinates file
            file_type: Type of file ('csv', 'txt', or 'auto')
        
        Returns:
            List of coordinate tuples
        """
        try:
            if file_type == 'csv' or (file_type == 'auto' and file_path.endswith('.csv')):
                df = pd.read_csv(file_path)
                return list(zip(df.iloc[:, 0], df.iloc[:, 1]))
            else:
                data = np.loadtxt(file_path)
                return list(map(tuple, data))
        except Exception as e:
            self.logger.error(f"Error loading coordinates from {file_path}: {str(e)}")
            raise
    
    def coordinates_to_distance_matrix(
        self,
        coordinates: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Convert coordinates to distance matrix."""
        n = len(coordinates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances[i][j] = distance
                distances[j][i] = distance
                
        return distances
    
    def validate_distance_matrix(self, matrix: np.ndarray) -> bool:
        """
        Validate distance matrix properties.
        
        Args:
            matrix: Distance matrix to validate
            
        Returns:
            bool: True if matrix is valid
        """
        try:
            # Check if matrix is square
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Distance matrix must be square")
            
            # Check symmetry
            if not np.allclose(matrix, matrix.T):
                raise ValueError("Distance matrix must be symmetric")
            
            # Check diagonal
            if not np.allclose(np.diagonal(matrix), 0):
                raise ValueError("Diagonal elements must be zero")
            
            # Check for negative distances
            if (matrix < 0).any():
                raise ValueError("Negative distances are not allowed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Matrix validation failed: {str(e)}")
            return False