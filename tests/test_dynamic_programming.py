# Import necessary libraries and packages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.dynamic_programming import DynamicProgrammingTSP
from utils.data_generator import TSPDataGenerator

def test_dp_solver():
    print("\nTesting Dynamic Programming TSP Solver...")
    
    # Generate test instances
    generator = TSPDataGenerator(seed=42)
    small_instance = generator.generate_euclidean_instance(size=10)
    large_instance = generator.generate_euclidean_instance(size=30)
    
    # Initialize solver
    solver = DynamicProgrammingTSP(max_size=20)
    
    # Test small instance (should use DP)
    print("\nTesting small instance (n=10)...")
    small_result = solver.solve(small_instance.distances)
    print(f"Path: {small_result.path}")
    print(f"Distance: {small_result.total_distance:.2f}")
    print(f"Optimal: {small_result.optimal}")
    
    # Test large instance (should use fallback)
    print("\nTesting large instance (n=30)...")
    large_result = solver.solve(large_instance.distances)
    print(f"Path: {large_result.path}")
    print(f"Distance: {large_result.total_distance:.2f}")
    print(f"Optimal: {large_result.optimal}")
    
    # Basic checks
    assert small_result.optimal == True, "Small instance should be solved optimally"
    assert large_result.optimal == False, "Large instance should use fallback"
    assert len(small_result.path) == 11, "Path should visit all cities and return"
    assert small_result.path[0] == small_result.path[-1], "Path should return to start"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_dp_solver()