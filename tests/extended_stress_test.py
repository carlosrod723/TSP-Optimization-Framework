# tests/extended_stress_test.py

import sys
import os
import time
import logging
import psutil
import json
from datetime import datetime, timedelta
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import TSPDataGenerator
from utils.mainnet_simulator import MainnetSimulator
from utils.algorithm_selector import AlgorithmSelector

class ExtendedStressTest:
    def __init__(self, 
                 duration_hours: int = 24,
                 log_interval_minutes: int = 30):
        self.duration = duration_hours
        self.log_interval = log_interval_minutes
        self.generator = TSPDataGenerator(seed=42)
        self.simulator = MainnetSimulator()
        self.selector = AlgorithmSelector()
        
        # Enhanced metrics tracking
        self.metrics = {
            'performance_history': [],
            'resource_usage': [],
            'solution_quality': {},
            'response_times': {},
            'failure_points': [],
            'algorithm_distribution': {}
        }
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'results/stress_test_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_single_test(self, size: int, load: float, current_hour: float):
        """Run a single test case and record metrics."""
        try:
            # Generate problem instance
            instance = self.generator.generate_euclidean_instance(size)
            
            # Select algorithm
            start_time = time.time()
            algorithm = self.selector.select_algorithm(
                size,
                self.simulator.get_time_limit(size),
                load
            )
            
            # Run solution
            result = self.simulator.run_with_constraints(
                algorithm.algorithm.solve,
                instance.distances
            )
            
            # Record metrics
            execution_time = time.time() - start_time
            self.record_metrics(size, load, algorithm.name, result, execution_time)
            
            # Print immediate results
            print(f"\nTest at hour {current_hour:.1f}:")
            print(f"Size: {size}, Load: {load}")
            print(f"Algorithm: {algorithm.name}")
            print(f"Success: {result.success}")
            print(f"Time: {execution_time:.3f}s")
            print(f"Quality: {result.solution_quality:.2f}")
            
            if result.deregistration_risk:
                print("WARNING: Deregistration Risk Detected!")
                self.metrics['failure_points'].append({
                    'hour': current_hour,
                    'size': size,
                    'load': load
                })
                
        except Exception as e:
            self.logger.error(f"Error in test case (size={size}, load={load}): {str(e)}")
    
    def record_metrics(self, size, load, algorithm_name, result, execution_time):
        """Record detailed metrics for the test case."""
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'size': size,
            'load': load,
            'algorithm': algorithm_name,
            'success': result.success,
            'execution_time': execution_time,
            'quality': result.solution_quality,
            'memory_usage': result.memory_usage,
            'cpu_usage': result.cpu_usage
        }
        
        self.metrics['performance_history'].append(metrics_entry)
        
        # Update algorithm distribution
        if algorithm_name not in self.metrics['algorithm_distribution']:
            self.metrics['algorithm_distribution'][algorithm_name] = 0
        self.metrics['algorithm_distribution'][algorithm_name] += 1
        
        # Update solution quality tracking
        size_key = f"size_{size}"
        if size_key not in self.metrics['solution_quality']:
            self.metrics['solution_quality'][size_key] = []
        self.metrics['solution_quality'][size_key].append(result.solution_quality)
        
        # Update response times
        if size_key not in self.metrics['response_times']:
            self.metrics['response_times'][size_key] = []
        self.metrics['response_times'][size_key].append(execution_time)
    
    def should_generate_report(self) -> bool:
        """Determine if it's time to generate an interim report."""
        if not self.metrics['performance_history']:
            return False
        
        last_entry = datetime.fromisoformat(self.metrics['performance_history'][-1]['timestamp'])
        time_since_last = (datetime.now() - last_entry).total_seconds() / 60
        
        return time_since_last >= self.log_interval
    
    def generate_interim_report(self):
        """Generate and print interim report."""
        print("\nInterim Report:")
        print("==============")
        
        # Calculate success rate
        total_tests = len(self.metrics['performance_history'])
        successful_tests = sum(1 for m in self.metrics['performance_history'] if m['success'])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Failure Points: {len(self.metrics['failure_points'])}")
        
        # Algorithm distribution
        print("\nAlgorithm Usage:")
        for algo, count in self.metrics['algorithm_distribution'].items():
            percentage = (count / total_tests * 100) if total_tests > 0 else 0
            print(f"{algo}: {percentage:.1f}%")
    
    def generate_final_report(self, start_time: datetime, end_time: datetime):
        """Generate comprehensive final report."""
        report = {
            'test_duration': str(end_time - start_time),
            'total_tests': len(self.metrics['performance_history']),
            'metrics': self.metrics
        }
        
        # Save detailed report
        report_path = f'results/stress_test_report_{start_time.strftime("%Y%m%d_%H%M")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nFinal report saved to: {report_path}")
        self.print_summary(report)

    def run_extended_test(self):
        """Run the extended stress test with better progress tracking."""
        print(f"\nStarting {self.duration}-hour extended stress test...")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.duration)
        
        # Test configurations
        problem_sizes = [10, 20, 50, 100, 200]
        network_loads = [0.1, 0.3, 0.5, 0.8, 0.9]
        
        test_count = 0
        iteration = 0
        
        while datetime.now() < end_time:
            iteration += 1
            current_hour = (datetime.now() - start_time).total_seconds() / 3600
            
            print(f"\n=== Iteration {iteration} (Hour {current_hour:.1f}/{self.duration}) ===")
            print(f"Tests completed so far: {test_count}")
            
            # Run one batch of tests
            for size in problem_sizes:
                for load in network_loads:
                    self.run_single_test(size, load, current_hour)
                    test_count += 1
            
            # Generate interim report after each complete iteration
            self.generate_interim_report()
            
            # Add a delay between iterations to prevent overwhelming the system
            remaining_time = (end_time - datetime.now()).total_seconds()
            if remaining_time > 300:  # If more than 5 minutes remaining
                print("\nWaiting 5 minutes before next iteration...")
                time.sleep(300)
        
        # Generate final report
        self.generate_final_report(start_time, end_time)
        
        print(f"\nStress test completed.")
        print(f"Total iterations: {iteration}")
        print(f"Total tests run: {test_count}")
    
    def print_summary(self, report):
        """Print summary of test results."""
        print("\nTest Summary:")
        print("============")
        print(f"Duration: {report['test_duration']}")
        print(f"Total Tests: {report['total_tests']}")
        
        # Success rate
        successful_tests = sum(1 for m in self.metrics['performance_history'] if m['success'])
        success_rate = (successful_tests / report['total_tests'] * 100) if report['total_tests'] > 0 else 0
        print(f"Overall Success Rate: {success_rate:.2f}%")
        
        # Performance by size
        print("\nPerformance by Size:")
        for size_key in sorted(self.metrics['solution_quality'].keys()):
            qualities = self.metrics['solution_quality'][size_key]
            times = self.metrics['response_times'][size_key]
            print(f"\n{size_key}:")
            print(f"  Avg Quality: {np.mean(qualities):.2f}")
            print(f"  Avg Time: {np.mean(times):.3f}s")
            print(f"  Max Time: {np.max(times):.3f}s")

if __name__ == "__main__":
    # Run a shorter test first (1 hour) for initial validation
    test = ExtendedStressTest(duration_hours=1)
    test.run_extended_test()