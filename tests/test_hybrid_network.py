# tests/test_hybrid_network.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.hybrid_pointer_network import HybridPointerNetworkTSP
from utils.data_generator import TSPDataGenerator
import time
import logging

def test_hybrid_solver():
    print("\nTesting Hybrid Pointer Network TSP Solver...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize solver and data generator
    solver = HybridPointerNetworkTSP()
    generator = TSPDataGenerator(seed=42)
    
    # Test different problem sizes
    sizes = [10, 20, 50]  # Changed from problem_sizes to sizes to match usage
    
    for size in sizes:  # Using the correct variable name
        print(f"\nTesting size {size}:")
        instance = generator.generate_euclidean_instance(size)
        
        # Time the solution
        start_time = time.perf_counter()
        result = solver.solve(instance.distances)
        solve_time = time.perf_counter() - start_time
        
        # Print results
        print("\nSolution Analysis:")
        print(f"Path length: {len(result.path)} (expected {size + 1})")
        print(f"Total distance: {result.total_distance:.2f}")
        print(f"Method used: {result.method_used}")
        print(f"Execution time: {solve_time:.3f}s")
        print(f"Improvements: {result.improvements}")
        
        # Verify solution
        print("\nSolution Verification:")
        print(f"Start node: {result.path[0]}")
        print(f"End node: {result.path[-1]}")
        print(f"Unique cities: {len(set(result.path))}")
        
        # Basic checks
        assert len(result.path) == size + 1, "Invalid tour length"
        assert result.path[0] == result.path[-1], "Tour doesn't return to start"
        assert len(set(result.path[:-1])) == size, "Not all cities visited exactly once"

if __name__ == "__main__":
    test_hybrid_solver()