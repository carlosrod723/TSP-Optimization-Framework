import sys
import os
import time
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any

# Add paths for both implementations
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import implementations
from algorithms.greedy import OptimizedGreedyTSP
from algorithms.beam_search import EnhancedBeamSearchTSP
from benchmarks.simplified_solvers import SimplifiedNearestNeighbourSolver, SimplifiedBeamSearchSolver

class ComparisonBenchmark:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.problem_sizes = [10, 20, 50, 100]
        self.results = {}
    
    def run_comparison(self):
        """Run comparison between different TSP solvers"""
        print("\nRunning TSP Solver Comparison Benchmarks...")
        
        for size in self.problem_sizes:
            print(f"\nTesting problem size: {size}")
            
            # Generate symmetric distance matrix
            distance_matrix = np.random.rand(size, size)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            
            # Test implementations
            base_results = self._test_base_implementations(distance_matrix, size)
            our_results = self._test_our_implementations(distance_matrix, size)
            
            # Store results
            self.results[size] = {
                "base": base_results,
                "ours": our_results
            }
            
            # Print immediate results
            self._print_size_results(size)
        
        self._print_summary()
        self._save_results()
    
    def _test_base_implementations(
        self,
        distance_matrix: np.ndarray,
        size: int
    ) -> Dict[str, Any]:
        results = {}
        
        # Test their Greedy implementation
        print("Testing base Greedy implementation...")
        their_greedy = SimplifiedNearestNeighbourSolver()
        start_time = time.time()
        greedy_result = their_greedy.solve(distance_matrix)
        greedy_time = time.time() - start_time
        
        results["greedy"] = {
            "time": greedy_time,
            "total_distance": greedy_result.total_distance,
            "path_length": len(greedy_result.path)
        }
        
        # Test their Beam Search implementation
        print("Testing base Beam Search implementation...")
        their_beam = SimplifiedBeamSearchSolver()
        start_time = time.time()
        beam_result = their_beam.solve(distance_matrix)
        beam_time = time.time() - start_time
        
        results["beam"] = {
            "time": beam_time,
            "total_distance": beam_result.total_distance,
            "path_length": len(beam_result.path)
        }
        
        return results
    
    def _test_our_implementations(
        self,
        distance_matrix: np.ndarray,
        size: int
    ) -> Dict[str, Any]:
        results = {}
        
        # Test our Greedy implementation
        print("Testing our Greedy implementation...")
        our_greedy = OptimizedGreedyTSP()
        start_time = time.time()
        greedy_result = our_greedy.solve(distance_matrix)
        greedy_time = time.time() - start_time
        
        results["greedy"] = {
            "time": greedy_time,
            "total_distance": greedy_result.total_distance,
            "path_length": len(greedy_result.path)
        }
        
        # Test our Beam Search implementation
        print("Testing our Beam Search implementation...")
        our_beam = EnhancedBeamSearchTSP()
        start_time = time.time()
        beam_result = our_beam.solve(distance_matrix)
        beam_time = time.time() - start_time
        
        results["beam"] = {
            "time": beam_time,
            "total_distance": beam_result.total_distance,
            "path_length": len(beam_result.path)
        }
        
        return results
    
    def _print_size_results(self, size: int):
        print(f"\nResults for size {size}:")
        base = self.results[size]["base"]
        ours = self.results[size]["ours"]
        
        print("\nGreedy Algorithm:")
        print(f"  Base: {base['greedy']['total_distance']:.2f} ({base['greedy']['time']:.3f}s)")
        print(f"  Ours: {ours['greedy']['total_distance']:.2f} ({ours['greedy']['time']:.3f}s)")
        
        print("\nBeam Search:")
        print(f"  Base: {base['beam']['total_distance']:.2f} ({base['beam']['time']:.3f}s)")
        print(f"  Ours: {ours['beam']['total_distance']:.2f} ({ours['beam']['time']:.3f}s)")
    
    def _print_summary(self):
        print("\nOverall Performance Summary:")
        print("=" * 60)
        
        for size in self.problem_sizes:
            print(f"\nProblem Size: {size}")
            base = self.results[size]["base"]
            ours = self.results[size]["ours"]
            
            greedy_improvement = (
                (base["greedy"]["total_distance"] - ours["greedy"]["total_distance"]) /
                base["greedy"]["total_distance"] * 100
            )
            
            beam_improvement = (
                (base["beam"]["total_distance"] - ours["beam"]["total_distance"]) /
                base["beam"]["total_distance"] * 100
            )
            
            print(f"Greedy Improvement: {greedy_improvement:+.2f}%")
            print(f"Beam Search Improvement: {beam_improvement:+.2f}%")
    
    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"results/benchmark_comparison_{timestamp}.txt"
        
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("TSP Solver Implementation Comparison\n")
            f.write("=" * 60 + "\n\n")
            
            for size in self.problem_sizes:
                f.write(f"\nProblem Size: {size}\n")
                f.write("-" * 40 + "\n")
                
                base = self.results[size]["base"]
                ours = self.results[size]["ours"]
                
                f.write("\nGreedy Algorithm:\n")
                f.write(f"  Base: Distance={base['greedy']['total_distance']:.2f}, "
                       f"Time={base['greedy']['time']:.3f}s\n")
                f.write(f"  Ours: Distance={ours['greedy']['total_distance']:.2f}, "
                       f"Time={ours['greedy']['time']:.3f}s\n")
                
                f.write("\nBeam Search:\n")
                f.write(f"  Base: Distance={base['beam']['total_distance']:.2f}, "
                       f"Time={base['beam']['time']:.3f}s\n")
                f.write(f"  Ours: Distance={ours['beam']['total_distance']:.2f}, "
                       f"Time={ours['beam']['time']:.3f}s\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benchmark = ComparisonBenchmark()
    benchmark.run_comparison()