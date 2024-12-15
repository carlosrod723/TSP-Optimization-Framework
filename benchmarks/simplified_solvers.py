"""
Simplified versions of the Graphite solvers without external dependencies
"""

import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class SimplifiedResult:
    path: List[int]
    total_distance: float

class SimplifiedNearestNeighbourSolver:
    def solve(self, distance_matrix: np.ndarray) -> SimplifiedResult:
        """Solve TSP using nearest neighbor algorithm"""
        n = len(distance_matrix)
        unvisited = set(range(1, n))
        current = 0
        path = [current]
        total_distance = 0.0
        
        while unvisited:
            # Find nearest unvisited node
            next_node = min(unvisited, key=lambda x: distance_matrix[current][x])
            
            # Add to path
            path.append(next_node)
            total_distance += distance_matrix[current][next_node]
            unvisited.remove(next_node)
            current = next_node
        
        # Return to start
        path.append(0)
        total_distance += distance_matrix[current][0]
        
        return SimplifiedResult(path=path, total_distance=total_distance)

class SimplifiedBeamSearchSolver:
    def __init__(self, beam_width: int = 3):
        self.beam_width = beam_width
    
    def solve(self, distance_matrix: np.ndarray) -> SimplifiedResult:
        """Solve TSP using beam search"""
        n = len(distance_matrix)
        
        # Initialize beam with starting node
        beam = [(0.0, [0])]  # (cost, path)
        
        # Build tours
        while len(beam[0][1]) < n:
            candidates = []
            
            # Extend each path in beam
            for cost, path in beam:
                current = path[-1]
                
                # Try adding each unvisited node
                for next_node in range(n):
                    if next_node not in path:
                        new_path = path + [next_node]
                        new_cost = cost + distance_matrix[current][next_node]
                        candidates.append((new_cost, new_path))
            
            # Select best candidates for new beam
            beam = sorted(candidates, key=lambda x: x[0])[:self.beam_width]
        
        # Complete the best tour
        best_cost, best_path = beam[0]
        best_path.append(0)  # Return to start
        total_distance = best_cost + distance_matrix[best_path[-2]][0]
        
        return SimplifiedResult(path=best_path, total_distance=total_distance)