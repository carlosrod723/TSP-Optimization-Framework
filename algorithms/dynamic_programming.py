# Import necessary libraries and packages
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import logging

@dataclass
class DPResult:
    """Results from the Dynamic Programming algorithm."""
    path: List[int]
    total_distance: float
    optimal: bool

class DynamicProgrammingTSP:
    """
    Dynamic Programming implementation for TSP using Held-Karp algorithm.
    Guaranteed optimal solution for small to medium instances.
    """
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)
        
    def solve(self, distances: np.ndarray) -> DPResult:
        n = len(distances)
        if n > self.max_size:
            self.logger.warning(
                f"Problem size {n} exceeds DP limit {self.max_size}. "
                "Falling back to Beam Search."
            )
            return self._fallback_solver(distances)

        # Initialize the DP table
        # dp[(mask, pos)] gives shortest path visiting all vertices in mask
        # and ending at pos
        dp = {}
        backtrack = {}
        
        # Initialize base cases - paths from 0 to each city
        for i in range(1, n):
            dp[(1 | (1 << i), i)] = distances[0][i]
        
        # Iterate over all possible sets of cities
        for size in range(3, n + 1):
            for subset in self._get_combinations(range(1, n), size - 1):
                # Add starting city (0) to subset
                mask = 1  # Include city 0
                for city in subset:
                    mask |= (1 << city)
                
                # Try all possible last cities in subset
                for last in subset:
                    # Remove last city from consideration
                    prev_mask = mask ^ (1 << last)
                    min_dist = float('inf')
                    min_prev = -1
                    
                    # Try all possible cities before last
                    for prev in subset:
                        if prev == last:
                            continue
                        if prev_mask & (1 << prev):
                            curr_dist = dp[(prev_mask, prev)] + distances[prev][last]
                            if curr_dist < min_dist:
                                min_dist = curr_dist
                                min_prev = prev
                    
                    if min_prev != -1:
                        dp[(mask, last)] = min_dist
                        backtrack[(mask, last)] = min_prev
        
        # Find optimal tour
        min_dist = float('inf')
        min_last = -1
        final_mask = (1 << n) - 1  # All cities visited
        
        # Try all possible last cities before returning to 0
        for last in range(1, n):
            if (final_mask, last) in dp:
                curr_dist = dp[(final_mask, last)] + distances[last][0]
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_last = last
        
        # Reconstruct path
        if min_last == -1:
            self.logger.error("No valid tour found")
            return self._fallback_solver(distances)
            
        path = self._reconstruct_path(backtrack, final_mask, min_last, n)
        
        return DPResult(
            path=path,
            total_distance=min_dist,
            optimal=True
        )
    
    def _get_combinations(self, elements, size):
        """Generate all possible combinations of given size."""
        if size == 0:
            yield []
            return
        for i in range(len(elements)):
            for combo in self._get_combinations(elements[i + 1:], size - 1):
                yield [elements[i]] + combo
    
    def _reconstruct_path(
        self,
        backtrack: Dict,
        mask: int,
        last: int,
        n: int
    ) -> List[int]:
        """Reconstruct the optimal path from backtracking information."""
        path = [0]  # Start from city 0
        curr = last
        curr_mask = mask
        
        while curr != 0:
            path.append(curr)
            next_curr = backtrack.get((curr_mask, curr))
            if next_curr is None:
                break
            curr_mask ^= (1 << curr)
            curr = next_curr
            
        path.append(0)  # Return to start
        return path
    
    def _fallback_solver(self, distances: np.ndarray) -> DPResult:
        """Fallback to Beam Search for large instances."""
        from .beam_search import BeamSearchTSP
        solver = BeamSearchTSP(beam_width=5)
        result = solver.solve(distances)
        return DPResult(
            path=result.path,
            total_distance=result.total_distance,
            optimal=False
        )