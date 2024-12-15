# Import necessary libraries and packages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.benchmark_suite import BenchmarkSuite
from algorithms.greedy import GreedyTSP
from algorithms.beam_search import BeamSearchTSP
import logging
from typing import Dict
from tabulate import tabulate

def run_comparison():
    print("\nRunning Algorithm Comparison...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize benchmark suite
    suite = BenchmarkSuite()
    
    # Initialize algorithms
    algorithms = {
        "greedy": GreedyTSP(),
        "beam_search_3": BeamSearchTSP(beam_width=3),
        "beam_search_5": BeamSearchTSP(beam_width=5)
    }
    
    # Test sizes
    sizes = [10, 20, 50]  # We can extend this later
    
    # Store results for comparison
    all_results = {}
    
    # Run benchmarks for each algorithm
    for name, algorithm in algorithms.items():
        print(f"\nBenchmarking {name}...")
        results = suite.run_benchmark(
            algorithm=algorithm,
            algorithm_name=name,
            problem_sizes=sizes,
            runs_per_size=5  # Number of runs per size
        )
        all_results[name] = results
    
    # Compare results
    print("\nComparison Results:")
    for size in sizes:
        print(f"\nProblem Size: {size}")
        
        # Prepare comparison table
        table_data = []
        headers = ["Algorithm", "Avg Time(s)", "Avg Quality", "Success Rate", "Memory(MB)"]
        
        for algo_name in algorithms.keys():
            result = all_results[algo_name][f"size_{size}"]
            table_data.append([
                algo_name,
                f"{result.average_time:.3f}",
                f"{result.average_quality:.2f}",
                f"{result.successful_runs}/{result.total_runs}",
                f"{result.max_memory_usage:.1f}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    run_comparison()