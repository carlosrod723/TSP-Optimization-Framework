# Import necessary libraries and packages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mainnet_simulator import MainnetSimulator
from utils.data_generator import TSPDataGenerator
from algorithms.greedy import GreedyTSP

def test_simulator():
    print("\nTesting Mainnet Simulator...")
    
    # Initialize components
    simulator = MainnetSimulator()
    data_generator = TSPDataGenerator(seed=42)
    solver = GreedyTSP()
    
    # Generate a test instance
    instance = data_generator.generate_euclidean_instance(size=20)
    
    # Run simulation
    print("\nRunning solver under mainnet conditions...")
    result = simulator.run_with_constraints(
        solver_func=solver.solve,
        problem_instance=instance.distances
    )
    
    # Print results
    print(f"\nSimulation Results:")
    print(f"Success: {result.success}")
    print(f"Execution Time: {result.execution_time:.3f}s")
    print(f"Memory Usage: {result.memory_usage:.2f}MB")
    print(f"CPU Usage: {result.cpu_usage:.2f}%")
    print(f"Solution Quality: {result.solution_quality:.2f}")
    if result.error_message:
        print(f"Error: {result.error_message}")

if __name__ == "__main__":
    test_simulator()