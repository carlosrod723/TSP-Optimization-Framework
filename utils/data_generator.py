import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class TSPInstance:
    """Class to hold a TSP problem instance."""
    size: int  # Number of cities
    distances: np.ndarray  # Distance matrix
    optimal_tour: List[int] = None  # Optional known optimal tour
    optimal_length: float = None  # Optional known optimal length
    name: str = None  # Instance identifier

class TSPDataGenerator:
    def __init__(self, seed: int = None):
        """Initialize the generator with optional random seed."""
        self.rng = np.random.RandomState(seed)
    
    def generate_euclidean_instance(
        self,
        size: int,
        min_coord: float = 0,
        max_coord: float = 100
    ) -> TSPInstance:
        """Generate a Euclidean TSP instance."""
        # Generate random city coordinates
        points = self.rng.uniform(min_coord, max_coord, size=(size, 2))
        
        # Calculate distance matrix
        distances = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i != j:
                    distances[i][j] = np.sqrt(
                        np.sum((points[i] - points[j]) ** 2)
                    )
        
        return TSPInstance(
            size=size,
            distances=distances,
            name=f"euclidean_{size}"
        )

    def generate_test_suite(
        self,
        sizes: List[int] = [10, 20, 50]  # Start with small sizes for testing
    ) -> List[TSPInstance]:
        """Generate a suite of test instances of varying sizes."""
        test_suite = []
        
        for size in sizes:
            instance = self.generate_euclidean_instance(size)
            test_suite.append(instance)
        
        return test_suite