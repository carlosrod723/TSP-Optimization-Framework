# tests/large_scale_benchmark.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import logging
from datetime import datetime
from large_scale_handlers.large_instance_solver import LargeInstanceSolver
from large_scale_handlers.batch_processor import BatchProcessor
from large_scale_handlers.partition_handler import PartitionHandler
from large_scale_handlers.memory_optimizer import MemoryOptimizer
from utils.data_generator import TSPDataGenerator

def run_large_scale_benchmark():
    """Run comprehensive benchmarks for large-scale instances."""
    
    print("\nStarting Large-Scale TSP Benchmarks...")
    
    # Test sizes
    sizes = [1000, 2000, 5000, 10000]
    
    # Initialize components
    generator = TSPDataGenerator(seed=42)
    memory_optimizer = MemoryOptimizer()
    
    results = {}
    
    for size in sizes:
        print(f"\nTesting size {size}:")
        
        try:
            # Generate large instance
            print(f"Generating instance of size {size}...")
            instance = generator.generate_euclidean_instance(size)
            
            # Optimize memory usage
            print("Optimizing memory usage...")
            optimized_distances = memory_optimizer.optimize_matrix(instance.distances)
            
            # Initialize solver with appropriate parameters
            print("Initializing solver...")
            solver = LargeInstanceSolver(
                max_partition_size=min(1000, size // 4),  # Adjusted partition size
                num_threads=4
            )
            
            # Solve and time the solution
            print("Solving...")
            start_time = time.time()
            result = solver.solve(optimized_distances)
            solve_time = time.time() - start_time
            
            # Get memory stats
            memory_stats = memory_optimizer.get_memory_stats()
            
            # Verify result format
            if not hasattr(result, 'total_distance'):
                raise ValueError("Invalid result format")
            
            # Record results
            results[size] = {
                'time': solve_time,
                'distance': float(result.total_distance),
                'memory_peak': float(memory_stats.peak_usage_mb),
                'partitions': int(result.partitions_used),
                'method': str(result.method_used)
            }
            
            # Print immediate results
            print(f"\nResults for size {size}:")
            print(f"Solution time: {solve_time:.2f} seconds")
            print(f"Total distance: {result.total_distance:.2f}")
            print(f"Peak memory: {memory_stats.peak_usage_mb:.2f}MB")
            print(f"Partitions used: {result.partitions_used}")
            print(f"Method used: {result.method_used}")
            
        except Exception as e:
            print(f"Error testing size {size}: {str(e)}")
            import traceback
            print(traceback.format_exc())  # This will show the full error trace
            results[size] = {'error': str(e)}
            
            # Save detailed results
            save_results(results)
            
            print("\nBenchmark Complete!")
            return results

def save_results(results: dict):
    """Save benchmark results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"results/large_scale_benchmark_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("Large-Scale TSP Benchmark Results\n")
        f.write("================================\n\n")
        
        for size, result in sorted(results.items()):
            f.write(f"\nSize {size}:\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            else:
                f.write(f"Solution time: {result['time']:.2f} seconds\n")
                f.write(f"Total distance: {result['distance']:.2f}\n")
                f.write(f"Peak memory: {result['memory_peak']:.2f}MB\n")
                f.write(f"Partitions: {result['partitions']}\n")
                f.write(f"Method: {result['method']}\n")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run benchmarks
    results = run_large_scale_benchmark()