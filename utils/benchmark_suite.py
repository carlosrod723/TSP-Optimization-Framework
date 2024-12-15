# utils/benchmark_suite.py

import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np

from .mainnet_simulator import MainnetSimulator
from .data_generator import TSPDataGenerator
from .config_handler import MainnetConfig

@dataclass
class BenchmarkResults:
    """Stores results for a complete benchmark run."""
    algorithm_name: str
    problem_size: int
    total_runs: int
    successful_runs: int
    average_time: float
    average_quality: float
    max_memory_usage: float
    average_cpu_usage: float
    deregistration_risk: bool
    timestamp: str

class BenchmarkSuite:
    def __init__(self, results_dir: str = "results/benchmarks"):
        self.config = MainnetConfig()
        self.simulator = MainnetSimulator()
        self.data_generator = TSPDataGenerator(seed=42)
        self.logger = logging.getLogger(__name__)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self,
        algorithm: Any,
        algorithm_name: str,
        problem_sizes: List[int] = None,
        runs_per_size: int = None
    ) -> Dict[str, BenchmarkResults]:
        """Run comprehensive benchmarks for an algorithm."""
        
        # Use config values if not specified
        problem_sizes = problem_sizes or self.config.testing['test_sizes']
        runs_per_size = runs_per_size or self.config.testing['num_test_runs']
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for size in problem_sizes:
            self.logger.info(f"Benchmarking {algorithm_name} on size {size}")
            
            successful_runs = 0
            execution_times = []
            quality_scores = []
            memory_usages = []
            cpu_usages = []
            
            # Generate a set of instances for this size
            instances = [
                self.data_generator.generate_euclidean_instance(size)
                for _ in range(runs_per_size)
            ]

            # Run warm-up iterations
            for _ in range(self.config.testing['warmup_runs']):
                self.simulator.run_with_constraints(
                    algorithm.solve,
                    instances[0].distances
                )

            # Run actual benchmarks
            for instance in instances:
                result = self.simulator.run_with_constraints(
                    algorithm.solve,
                    instance.distances
                )

                if result.success:
                    successful_runs += 1
                    execution_times.append(result.execution_time)
                    quality_scores.append(result.solution_quality)
                    memory_usages.append(result.memory_usage)
                    cpu_usages.append(result.cpu_usage)

            # Calculate statistics
            benchmark_result = BenchmarkResults(
                algorithm_name=algorithm_name,
                problem_size=size,
                total_runs=runs_per_size,
                successful_runs=successful_runs,
                average_time=np.mean(execution_times) if execution_times else float('inf'),
                average_quality=np.mean(quality_scores) if quality_scores else float('inf'),
                max_memory_usage=max(memory_usages) if memory_usages else 0,
                average_cpu_usage=np.mean(cpu_usages) if cpu_usages else 0,
                deregistration_risk=self.simulator.would_be_deregistered(),
                timestamp=timestamp
            )

            results[f"size_{size}"] = benchmark_result
            
            # Save results to file
            self._save_results(algorithm_name, results, timestamp)

        return results

    def _save_results(
        self,
        algorithm_name: str,
        results: Dict[str, BenchmarkResults],
        timestamp: str
    ):
        """Save benchmark results to JSON file."""
        results_dict = {
            size: asdict(result)
            for size, result in results.items()
        }
        
        filename = f"{algorithm_name}_benchmark_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Saved benchmark results to {filepath}")