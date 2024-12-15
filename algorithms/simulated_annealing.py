import numpy as np
from typing import List, Tuple
import random
import math
import time
from dataclasses import dataclass

@dataclass
class SAResult:
    """Results from Simulated Annealing algorithm."""
    path: List[int]
    total_distance: float
    iterations: int
    time_taken: float
    best_temperature: float

class SimulatedAnnealingTSP:
    def __init__(
        self,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.995,
        min_temp: float = 1e-8,
        max_iterations: int = 10000
    ):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
    
    def solve(self, distances: np.ndarray) -> SAResult:
        """
        Solve TSP using Simulated Annealing.
        """
        start_time = time.time()
        n = len(distances)
        
        # Generate initial solution
        current_path = self._generate_initial_solution(n)
        current_distance = self._calculate_path_length(distances, current_path)
        
        # Initialize best solution
        best_path = current_path.copy()
        best_distance = current_distance
        
        # Simulated annealing
        temperature = self.initial_temp
        iteration = 0
        best_temp = temperature
        
        while temperature > self.min_temp and iteration < self.max_iterations:
            # Generate neighbor solution
            neighbor_path = self._get_neighbor(current_path)
            neighbor_distance = self._calculate_path_length(distances, neighbor_path)
            
            # Calculate change in energy
            delta_e = neighbor_distance - current_distance
            
            # Accept or reject new solution
            if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
                current_path = neighbor_path
                current_distance = neighbor_distance
                
                # Update best solution if improved
                if current_distance < best_distance:
                    best_path = current_path.copy()
                    best_distance = current_distance
                    best_temp = temperature
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
        
        time_taken = time.time() - start_time
        
        return SAResult(
            path=best_path,
            total_distance=best_distance,
            iterations=iteration,
            time_taken=time_taken,
            best_temperature=best_temp
        )
    
    def _generate_initial_solution(self, n: int) -> List[int]:
        """Generate initial solution using greedy approach."""
        path = list(range(n))
        random.shuffle(path[1:])  # Keep 0 as start
        path.append(path[0])  # Complete the cycle
        return path
    
    def _get_neighbor(self, path: List[int]) -> List[int]:
        """Generate neighbor solution using 2-opt move."""
        n = len(path)
        neighbor = path.copy()
        
        # Select two random positions (excluding start/end)
        i = random.randint(1, n-3)
        j = random.randint(i+1, n-2)
        
        # Reverse the segment between i and j
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        
        return neighbor
    
    def _calculate_path_length(self, distances: np.ndarray, path: List[int]) -> float:
        """Calculate total path length."""
        return sum(
            distances[path[i]][path[i+1]]
            for i in range(len(path)-1)
        )