# Import necessary libraries and packages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.hybrid_pointer_network import HybridPointerNetworkTSP
from utils.data_generator import TSPDataGenerator

def test_hybrid_solver():
    print("\nTesting Hybrid Pointer Network TSP Solver...")
    
    # Generate test instances
    generator = TSPDataGenerator(seed=42)
    small_instance = generator.generate_euclidean_instance(size=15)
    large_instance = generator.generate_euclidean_instance(size=50)
    
    # Initialize solver
    solver = HybridPointerNetworkTSP()
    
    # Test small instance (should use DP)
    print("\nTesting small instance (n=15)...")
    small_result = solver.solve(small_instance.distances)
    print(f"Path: {small_result.path}")
    print(f"Distance: {small_result.total_distance:.2f}")
    print(f"Method: {small_result.method_used}")
    
    # Test large instance (should use neural/fallback)
    print("\nTesting large instance (n=50)...")
    large_result = solver.solve(large_instance.distances)
    print(f"Path: {large_result.path}")
    print(f"Distance: {large_result.total_distance:.2f}")
    print(f"Method: {large_result.method_used}")

if __name__ == "__main__":
    test_hybrid_solver()