# tests/test_benchmark_suite.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.benchmark_suite import BenchmarkSuite
from algorithms.greedy import GreedyTSP
import logging

def test_benchmark_suite():
    print("\nTesting Benchmark Suite...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    suite = BenchmarkSuite()
    solver = GreedyTSP()
    
    # Run a small benchmark
    results = suite.run_benchmark(
        algorithm=solver,
        algorithm_name="greedy_tsp",
        problem_sizes=[10, 20],  # Small sizes for testing
        runs_per_size=2  # Few runs for testing
    )
    
    # Print results
    for size, result in results.items():
        print(f"\nResults for {size}:")
        print(f"Success rate: {result.successful_runs}/{result.total_runs}")
        print(f"Average time: {result.average_time:.3f}s")
        print(f"Average solution quality: {result.average_quality:.2f}")
        print(f"Max memory usage: {result.max_memory_usage:.2f}MB")
        print(f"Deregistration risk: {result.deregistration_risk}")

if __name__ == "__main__":
    test_benchmark_suite()