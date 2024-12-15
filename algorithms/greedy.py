import numpy as np
from typing import List, Tuple, Set, Dict
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import heapq
import time
import random

@dataclass
class GreedyResult:
    """Results from the Greedy algorithm."""
    path: List[int]
    total_distance: float
    improvements_made: int

class OptimizedGreedyTSP:
    def __init__(self, 
                 use_parallel: bool = True,
                 num_starts: int = 15,
                 use_2opt: bool = True,
                 use_3opt: bool = True,
                 max_no_improve: int = 300):
        self.logger = logging.getLogger(__name__)
        self.use_parallel = use_parallel
        self.num_starts = num_starts
        self.use_2opt = use_2opt
        self.use_3opt = use_3opt
        self.max_no_improve = max_no_improve
        
    def solve(self, distances: np.ndarray) -> GreedyResult:
        """Main solve method that interfaces with the testing framework."""
        n = len(distances)
        start_time = time.time()
        
        # Get best solution using limited strategies based on available time
        solutions = []
        
        # Use faster construction methods for time-sensitive cases
        if n <= 30:  # For smaller instances
            # Just use nearest neighbor with less randomization
            sol = self._alpha_nearest_neighbor(distances, 0, alpha=0.1)
            solutions.append(sol)
        else:
            # For larger instances, use multiple strategies
            alphas = [0.1, 0.2, 0.3]
            for alpha in alphas:
                sol = self._alpha_nearest_neighbor(distances, 0, alpha)
                solutions.append(sol)
            
            solutions.append(self._savings_construction(distances))
        
        # Process solutions with time awareness
        best_solution = None
        best_distance = float('inf')
        total_improvements = 0
        
        for solution in solutions:
            elapsed = time.time() - start_time
            if elapsed > 0.8:  # Time safety margin
                break
                
            current_solution = solution
            current_distance = self._calculate_tour_length(current_solution, distances)
            
            # Limited improvement attempts
            for _ in range(3):  # Limit improvement iterations
                new_solution, change = self._lin_kernighan_step(current_solution, distances)
                if change < -1e-10:
                    current_solution = new_solution
                    current_distance += change
                    total_improvements += 1
                else:
                    break
            
            if current_distance < best_distance:
                best_distance = current_distance
                best_solution = current_solution
        
        if best_solution is None:
            # Emergency fallback to simple nearest neighbor
            best_solution = self._alpha_nearest_neighbor(distances, 0, alpha=0)
            best_distance = self._calculate_tour_length(best_solution, distances)
        
        return GreedyResult(
            path=best_solution,
            total_distance=best_distance,
            improvements_made=total_improvements
        )

    def _calculate_tour_length(self, tour: List[int], distances: np.ndarray) -> float:
        """Calculate the total length of a tour."""
        return sum(distances[tour[i]][tour[i+1]] for i in range(len(tour)-1))

    def _alpha_nearest_neighbor(self, distances: np.ndarray, start: int, alpha: float = 0.3) -> List[int]:
        """Enhanced nearest neighbor with randomization."""
        n = len(distances)
        unvisited = set(range(n))
        path = [start]
        unvisited.remove(start)
        
        while unvisited:
            current = path[-1]
            candidates = [(distances[current][j], j) for j in unvisited]
            candidates.sort()  # Sort by distance
            
            # Select from top alpha% candidates randomly
            num_candidates = max(1, int(len(candidates) * alpha))
            selected = random.choice(candidates[:num_candidates])[1]
            
            path.append(selected)
            unvisited.remove(selected)
        
        path.append(start)  # Complete the tour
        return path

    def _savings_construction(self, distances: np.ndarray) -> List[int]:
        """Clarke-Wright savings algorithm."""
        n = len(distances)
        depot = 0  # Use first city as depot
        
        # Calculate savings for each pair
        savings = []
        for i in range(1, n):
            for j in range(i+1, n):
                saving = distances[i][depot] + distances[depot][j] - distances[i][j]
                savings.append((saving, i, j))
        
        savings.sort(reverse=True)  # Sort by savings value
        
        # Initialize routes
        routes = [{i} for i in range(1, n)]
        route_ends = {i: i for i in range(1, n)}  # Track ends of routes
        
        # Merge routes based on savings
        for saving, i, j in savings:
            if saving <= 0:
                break
                
            # Find routes containing i and j
            route_i = route_j = None
            for route in routes:
                if i in route:
                    route_i = route
                if j in route:
                    route_j = route
                    
            # Skip if cities are in same route
            if route_i is route_j:
                continue
                
            # Merge routes if possible
            if route_ends[i] == i and route_ends[j] == j:
                route_i.update(route_j)
                routes.remove(route_j)
                route_ends[i] = j
                route_ends[j] = i
        
        # Construct final tour
        tour = [depot]
        visited = {depot}
        current = depot
        
        while len(visited) < n:
            next_city = min(range(n), 
                          key=lambda x: float('inf') if x in visited else distances[current][x])
            tour.append(next_city)
            visited.add(next_city)
            current = next_city
            
        tour.append(depot)
        return tour

    def _lin_kernighan_step(self, tour: List[int], distances: np.ndarray) -> Tuple[List[int], float]:
        """Single step of Lin-Kernighan heuristic with time limit."""
        n = len(tour)
        best_gain = 0
        best_move = None
        
        for i in range(1, n-2):
            t1 = tour[i-1]
            t2 = tour[i]
            g0 = distances[t1][t2]  # Initial broken edge
            
            # Try to find improving sequence
            for j in range(i+1, n-1):
                t3 = tour[j]
                t4 = tour[j+1]
                g1 = distances[t2][t3]  # New edge
                
                if g0 > g1:  # Potential improvement
                    gain = g0 - g1
                    if gain > best_gain:
                        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                        best_gain = gain
                        best_move = new_tour
        
        if best_move:
            return best_move, -best_gain
        return tour, 0