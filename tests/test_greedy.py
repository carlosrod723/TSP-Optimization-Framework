# Import necessary libraries and packages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.greedy import OptimizedGreedyTSP
from utils.data_generator import TSPDataGenerator

def test_greedy_solver():
    print("\nTesting Greedy TSP Solver...")
    
    # Generate a small test instance
    generator = TSPDataGenerator(seed=42)
    instance = generator.generate_euclidean_instance(size=10)
    
    # Test basic configuration
    print("\nTesting basic configuration...")
    basic_solver = OptimizedGreedyTSP(
        use_parallel=False,
        num_starts=1,
        use_2opt=False
    )
    basic_result = basic_solver.solve(instance.distances)
    
    # Test optimized configuration
    print("\nTesting optimized configuration...")
    optimized_solver = OptimizedGreedyTSP(
        use_parallel=True,
        num_starts=3,
        use_2opt=True
    )
    optimized_result = optimized_solver.solve(instance.distances)
    
    # Print results
    print(f"\nBasic Configuration:")
    print(f"Path: {basic_result.path}")
    print(f"Distance: {basic_result.total_distance:.2f}")
    print(f"Improvements: {basic_result.improvements_made}")
    
    print(f"\nOptimized Configuration:")
    print(f"Path: {optimized_result.path}")
    print(f"Distance: {optimized_result.total_distance:.2f}")
    print(f"Improvements: {optimized_result.improvements_made}")
    
    # Basic checks
    assert len(basic_result.path) == len(instance.distances) + 1, "Path should visit all cities and return"
    assert basic_result.path[0] == basic_result.path[-1], "Path should return to start"
    assert optimized_result.total_distance <= basic_result.total_distance, "Optimized version should not be worse"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_greedy_solver()