# tests/test_beam_search.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.beam_search import EnhancedBeamSearchTSP
from utils.data_generator import TSPDataGenerator
import time

def test_enhanced_beam_search():
    print("\nTesting Enhanced Beam Search TSP Solver...")
    
    # Generate test instances
    generator = TSPDataGenerator(seed=42)
    sizes = [10, 20, 50]
    
    for size in sizes:
        instance = generator.generate_euclidean_instance(size)
        print(f"\nTesting size {size}:")
        
        solver = EnhancedBeamSearchTSP()
        
        # Time the solution
        start_time = time.perf_counter()
        result = solver.solve(instance.distances)
        solve_time = time.perf_counter() - start_time
        
        # Detailed analysis
        print("\nSolution Analysis:")
        print(f"Path length: {len(result.path)} (expected {size + 1})")
        print(f"Initial distance: {result.total_distance:.2f}")
        print(f"Local search improvements: {result.improvement_count}")
        print(f"Execution time: {solve_time:.3f}s")
        print(f"Iterations: {result.iterations}")
        
        # Path verification
        print("\nPath Verification:")
        print(f"Start node: {result.path[0]}")
        print(f"End node: {result.path[-1]}")
        print(f"Unique cities: {len(set(result.path))}")
        
        # Basic correctness checks
        assert len(result.path) == size + 1, f"Invalid tour length: got {len(result.path)}, expected {size + 1}"
        assert result.path[0] == result.path[-1], "Tour doesn't return to start"
        assert len(set(result.path[:-1])) == size, "Not all cities are visited exactly once"
        
        # Performance checks
        assert result.total_distance > 0, "Invalid tour distance"
        assert solve_time >= 0, "Invalid solving time"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_enhanced_beam_search()