# Import necessary libraries and packages
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
import logging

@dataclass
class BenchmarkResult:
    algorithm_name: str
    path_length: float
    computation_time: float
    solution_quality: float  # Compared to known optimal or best known
    memory_usage: float     # Peak memory usage in MB

class PerformanceTracker:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.baseline_results = {}
        self.results_history = []
        
    def measure_performance(
        self,
        algorithm: Callable,
        problem_instance: np.ndarray,
        algorithm_name: str,
        optimal_value: float = None
    ) -> BenchmarkResult:
        """Measure algorithm performance on a given problem instance."""
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # Run algorithm
        try:
            solution = algorithm(problem_instance)
            path_length = self.calculate_path_length(problem_instance, solution)
        except Exception as e:
            self.logger.error(f"Algorithm {algorithm_name} failed: {str(e)}")
            tracemalloc.stop()
            raise
        
        # Measure time and memory
        computation_time = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate solution quality
        solution_quality = (
            (optimal_value / path_length) * 100 if optimal_value
            else None
        )
        
        result = BenchmarkResult(
            algorithm_name=algorithm_name,
            path_length=path_length,
            computation_time=computation_time,
            solution_quality=solution_quality,
            memory_usage=peak / 1024 / 1024  # Convert to MB
        )
        
        self.results_history.append(result)
        return result
    
    @staticmethod
    def calculate_path_length(distance_matrix: np.ndarray, path: List[int]) -> float:
        """Calculate total path length for a given solution."""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += distance_matrix[path[i]][path[i + 1]]
        return total_distance
    
    def compare_to_baseline(self, result: BenchmarkResult) -> Dict[str, float]:
        """Compare result to baseline performance."""
        if result.algorithm_name not in self.baseline_results:
            self.baseline_results[result.algorithm_name] = result
            return {"improvement": 0.0}
        
        baseline = self.baseline_results[result.algorithm_name]
        return {
            "path_length_improvement": (
                (baseline.path_length - result.path_length) / baseline.path_length * 100
            ),
            "time_improvement": (
                (baseline.computation_time - result.computation_time) / baseline.computation_time * 100
            ),
            "memory_improvement": (
                (baseline.memory_usage - result.memory_usage) / baseline.memory_usage * 100
            )
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "summary": {
                "total_runs": len(self.results_history),
                "algorithms_tested": len(set(r.algorithm_name for r in self.results_history))
            },
            "detailed_results": {}
        }
        
        # Group results by algorithm
        for algorithm_name in set(r.algorithm_name for r in self.results_history):
            algorithm_results = [r for r in self.results_history if r.algorithm_name == algorithm_name]
            
            report["detailed_results"][algorithm_name] = {
                "avg_path_length": np.mean([r.path_length for r in algorithm_results]),
                "avg_computation_time": np.mean([r.computation_time for r in algorithm_results]),
                "avg_memory_usage": np.mean([r.memory_usage for r in algorithm_results]),
                "best_solution_quality": max([r.solution_quality for r in algorithm_results if r.solution_quality is not None], default=None)
            }
        
        return report