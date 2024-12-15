import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.simulated_annealing import SimulatedAnnealingTSP
from utils.data_generator import TSPDataGenerator

def test_simulated_annealing():
    print("\nTesting Simulated Annealing TSP Solver...")
    
    # Generate test instances
    generator = TSPDataGenerator(seed=42)
    sizes = [10, 20, 50, 100]
    
    for size in sizes:
        print(f"\nTesting size {size}:")
        instance = generator.generate_euclidean_instance(size)
        
        # Solve using SA
        solver = SimulatedAnnealingTSP()
        result = solver.solve(instance.distances)
        
        # Print results
        print(f"Distance: {result.total_distance:.2f}")
        print(f"Time taken: {result.time_taken:.3f}s")
        print(f"Iterations: {result.iterations}")
        print(f"Best temperature: {result.best_temperature:.2f}")
        
        # Verify solution
        assert len(result.path) == size + 1, "Invalid tour length"
        assert result.path[0] == result.path[-1], "Tour doesn't return to start"
        assert len(set(result.path[:-1])) == size, "Not all cities visited"

if __name__ == "__main__":
    test_simulated_annealing()