# algorithms/beam_search.py

import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass
import heapq
import logging

@dataclass
class BeamSearchResult:
    """Results from the Beam Search algorithm."""
    path: List[int]
    total_distance: float
    iterations: int
    improvement_count: int

class EnhancedBeamSearchTSP:
    """
    Enhanced Beam Search with optimizations.
    """
    def __init__(
        self,
        min_beam_width: int = 3,
        max_beam_width: int = 10,
        max_iterations: int = 1000,
        look_ahead: int = 2
    ):
        self.min_beam_width = min_beam_width
        self.max_beam_width = max_beam_width
        self.max_iterations = max_iterations
        self.look_ahead = look_ahead
        self.logger = logging.getLogger(__name__)

    def solve(self, distances: np.ndarray) -> BeamSearchResult:
        """Solve TSP using enhanced beam search with local optimization."""
        n = len(distances)
        
        # Adapt parameters based on problem size
        beam_width = self._get_adaptive_beam_width(n)
        
        # Get initial solution using beam search
        initial_result = self._beam_search(distances, beam_width)
        
        # Apply local search optimization
        improved_path, improved_distance = self._apply_local_search(
            distances,
            initial_result.path,
            initial_result.total_distance
        )
        
        improvement_count = 1 if improved_distance < initial_result.total_distance else 0
        
        return BeamSearchResult(
            path=improved_path,
            total_distance=improved_distance,
            iterations=initial_result.iterations,
            improvement_count=improvement_count
        )

    def _get_adaptive_beam_width(self, n: int) -> int:
        """Calculate adaptive beam width based on problem size."""
        if n <= 20:
            return min(n, self.min_beam_width)
        elif n <= 50:
            return min(n // 3, self.max_beam_width)
        else:
            return self.max_beam_width

    def _beam_search(
        self,
        distances: np.ndarray,
        beam_width: int
    ) -> BeamSearchResult:
        """Core beam search implementation."""
        n = len(distances)
        start_node = 0
        
        # Initialize with starting node
        beam = [(
            [start_node],  # path
            0.0,          # cost
            set(range(1, n))  # unvisited
        )]
        
        best_path = None
        best_distance = float('inf')
        iterations = 0
        
        while beam and iterations < self.max_iterations:
            iterations += 1
            candidates = []
            
            for path, cost, unvisited in beam:
                current = path[-1]
                
                # If all cities visited, only consider returning to start
                if not unvisited:
                    if current != start_node:
                        new_cost = cost + distances[current][start_node]
                        candidates.append((
                            new_cost,
                            (path + [start_node], new_cost, unvisited)
                        ))
                    continue
                
                # Try visiting unvisited cities
                for next_city in unvisited:
                    score = self._evaluate_path(
                        distances,
                        path,
                        next_city,
                        unvisited
                    )
                    
                    new_cost = cost + distances[current][next_city]
                    candidates.append((
                        score,
                        (path + [next_city], new_cost, unvisited - {next_city})
                    ))
            
            if not candidates:
                break
            
            # Select best candidates
            candidates.sort(key=lambda x: x[0])
            beam = [c[1] for c in candidates[:beam_width]]
            
            # Update best solution if we have a complete tour
            for path, cost, unvisited in beam:
                if not unvisited and path[-1] == start_node:
                    if cost < best_distance:
                        best_distance = cost
                        best_path = path
        
        if best_path is None:
            # Fallback to nearest neighbor if no solution found
            best_path = self._nearest_neighbor(distances)
            best_distance = self._calculate_path_length(distances, best_path)
        
        return BeamSearchResult(
            path=best_path,
            total_distance=best_distance,
            iterations=iterations,
            improvement_count=0
        )

    def _evaluate_path(
        self,
        distances: np.ndarray,
        current_path: List[int],
        candidate: int,
        unvisited: Set[int]
    ) -> float:
        """Evaluate path quality with look-ahead."""
        score = distances[current_path[-1]][candidate]
        
        if not unvisited:
            return score + distances[candidate][0]
        
        # Look ahead
        current = candidate
        remaining = unvisited.copy()
        look_ahead_cost = 0
        discount = 0.7
        
        # Consider next few moves
        for i in range(min(self.look_ahead, len(remaining))):
            if not remaining:
                break
            next_city = min(remaining, key=lambda x: distances[current][x])
            look_ahead_cost += distances[current][next_city] * (discount ** i)
            current = next_city
            remaining.remove(next_city)
        
        # Estimate remaining cost
        if remaining:
            min_remaining = min(
                min(distances[x][y] for y in remaining | {0})
                for x in remaining
            )
            look_ahead_cost += min_remaining * (discount ** self.look_ahead)
        
        return score + look_ahead_cost

    def _nearest_neighbor(self, distances: np.ndarray) -> List[int]:
        """Get nearest neighbor solution."""
        n = len(distances)
        current = 0
        path = [current]
        unvisited = set(range(1, n))
        
        while unvisited:
            next_city = min(
                unvisited,
                key=lambda x: distances[current][x]
            )
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city
            
        path.append(0)
        return path

    def _calculate_path_length(self, distances: np.ndarray, path: List[int]) -> float:
        """Calculate total path length."""
        return sum(
            distances[path[i]][path[i + 1]]
            for i in range(len(path) - 1)
        )
    
    def _apply_local_search(
        self,
        distances: np.ndarray,
        path: List[int],
        initial_distance: float
    ) -> Tuple[List[int], float]:
        """
        Apply local search optimization using 2-opt and 3-opt moves.
        Returns improved path and its distance.
        """
        current_path = path[:]
        current_distance = initial_distance
        improvement_found = True
        
        while improvement_found:
            improvement_found = False
            
            # Try 2-opt improvements
            two_opt_path, two_opt_distance = self._apply_2opt(
                distances, current_path, current_distance
            )
            
            if two_opt_distance < current_distance:
                current_path = two_opt_path
                current_distance = two_opt_distance
                improvement_found = True
                continue
            
            # Try 3-opt improvements if 2-opt didn't help and path is small enough
            if len(path) <= 30:  # 3-opt is expensive, limit to smaller instances
                three_opt_path, three_opt_distance = self._apply_3opt(
                    distances, current_path, current_distance
                )
                
                if three_opt_distance < current_distance:
                    current_path = three_opt_path
                    current_distance = three_opt_distance
                    improvement_found = True
        
        return current_path, current_distance

    def _apply_2opt(
        self,
        distances: np.ndarray,
        path: List[int],
        current_distance: float
    ) -> Tuple[List[int], float]:
        """Apply 2-opt local search improvement."""
        best_path = path[:]
        best_distance = current_distance
        n = len(path)
        
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Calculate change in distance if we reverse path[i:j+1]
                change = (
                    distances[path[i-1]][path[j]] +
                    distances[path[i]][path[j+1]] -
                    distances[path[i-1]][path[i]] -
                    distances[path[j]][path[j+1]]
                )
                
                if change < -1e-10:  # Improvement found
                    new_path = (
                        path[:i] +
                        list(reversed(path[i:j+1])) +
                        path[j+1:]
                    )
                    new_distance = best_distance + change
                    
                    best_path = new_path
                    best_distance = new_distance
                    return best_path, best_distance  # Return first improvement
        
        return best_path, best_distance

    def _apply_3opt(
        self,
        distances: np.ndarray,
        path: List[int],
        current_distance: float
    ) -> Tuple[List[int], float]:
        """Apply 3-opt local search improvement."""
        best_path = path[:]
        best_distance = current_distance
        n = len(path)
        
        for i in range(1, n - 4):
            for j in range(i + 2, n - 2):
                for k in range(j + 2, n - 1):
                    for segment_config in range(8):  # Try all segment configurations
                        if segment_config == 0:
                            continue  # Skip original configuration
                        
                        new_path = self._get_3opt_path(
                            path, i, j, k, segment_config
                        )
                        new_distance = self._calculate_path_length(
                            distances, new_path
                        )
                        
                        if new_distance < best_distance:
                            best_path = new_path
                            best_distance = new_distance
                            return best_path, best_distance  # Return first improvement
        
        return best_path, best_distance

    def _get_3opt_path(
        self,
        path: List[int],
        i: int,
        j: int,
        k: int,
        config: int
    ) -> List[int]:
        """Generate a new path for 3-opt move based on configuration."""
        A = path[:i]
        B = path[i:j]
        C = path[j:k]
        D = path[k:]
        
        if config == 1:
            return A + B + list(reversed(C)) + D
        elif config == 2:
            return A + list(reversed(B)) + C + D
        elif config == 3:
            return A + list(reversed(B)) + list(reversed(C)) + D
        elif config == 4:
            return A + C + B + D
        elif config == 5:
            return A + C + list(reversed(B)) + D
        elif config == 6:
            return A + list(reversed(C)) + B + D
        else:  # config == 7
            return A + list(reversed(C)) + list(reversed(B)) + D