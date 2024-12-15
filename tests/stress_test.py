# tests/stress_test.py

import sys
import os
import time
import logging
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import TSPDataGenerator
from utils.mainnet_simulator import MainnetSimulator
from utils.algorithm_selector import AlgorithmSelector
from algorithms.greedy import OptimizedGreedyTSP
from algorithms.beam_search import EnhancedBeamSearchTSP
from algorithms.dynamic_programming import DynamicProgrammingTSP

class StressTest:
    def __init__(self, duration_hours: int = 168):  # 168 hours = 7 days
        self.duration = duration_hours
        self.generator = TSPDataGenerator(seed=42)
        self.simulator = MainnetSimulator()
        self.selector = AlgorithmSelector()
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        self.setup_logging()
    
    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        log_file = f"logs/stress_test_{timestamp}.log"
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def run_stress_test(self):
        """Run extended stress test simulating mainnet conditions."""
        print(f"\nStarting {self.duration}-hour stress test...")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.duration)
        
        results = {
            'total_problems': 0,
            'successful_solutions': 0,
            'deregistration_risks': 0,
            'sizes_tested': set(),
            'performance_by_size': {},
            'algorithm_usage': {}
        }
        
        # Test configurations
        problem_sizes = [10, 20, 50, 100, 200]
        network_loads = [0.3, 0.5, 0.8]
        
        current_time = start_time
        hour = 0
        
        while current_time < end_time:
            print(f"\nHour {hour}/{self.duration}")
            self.logger.info(f"Testing hour {hour}")
            
            for size in problem_sizes:
                for load in network_loads:
                    self.logger.info(f"Testing size {size} with load {load}")
                    
                    # Generate problem instance
                    instance = self.generator.generate_euclidean_instance(size)
                    
                    # Select and run algorithm
                    try:
                        algorithm = self.selector.select_algorithm(
                            size,
                            available_time=self.simulator.get_time_limit(size),
                            network_load=load
                        )
                        
                        result = self.simulator.run_with_constraints(
                            algorithm.algorithm.solve,
                            instance.distances
                        )
                        
                        # Update statistics
                        self.update_results(results, size, algorithm.name, result)
                        
                    except Exception as e:
                        self.logger.error(f"Error testing size {size}: {str(e)}")
                        results['deregistration_risks'] += 1
            
            hour += 1
            current_time = start_time + timedelta(hours=hour)
            
            # Generate interim report
            if hour % 24 == 0:
                self.generate_report(results, start_time, current_time)
        
        # Generate final report
        self.generate_final_report(results, start_time, end_time)
        return results
    
    def update_results(self, results, size, algorithm_name, run_result):
        """Update test results with new data."""
        results['total_problems'] += 1
        results['sizes_tested'].add(size)
        
        if run_result.success:
            results['successful_solutions'] += 1
        
        if run_result.deregistration_risk:
            results['deregistration_risks'] += 1
        
        # Update size-specific stats
        if size not in results['performance_by_size']:
            results['performance_by_size'][size] = {
                'total': 0,
                'successful': 0,
                'avg_time': 0,
                'max_time': 0
            }
        
        size_stats = results['performance_by_size'][size]
        size_stats['total'] += 1
        size_stats['successful'] += int(run_result.success)
        size_stats['avg_time'] = (
            (size_stats['avg_time'] * (size_stats['total'] - 1) + 
             run_result.execution_time) / size_stats['total']
        )
        size_stats['max_time'] = max(
            size_stats['max_time'],
            run_result.execution_time
        )
        
        # Update algorithm usage stats
        if algorithm_name not in results['algorithm_usage']:
            results['algorithm_usage'][algorithm_name] = 0
        results['algorithm_usage'][algorithm_name] += 1

        # Add immediate result output
        print(f"\nResult for size {size} with {algorithm_name}:")
        print(f"Success: {run_result.success}")
        print(f"Time: {run_result.execution_time:.3f}s")
        print(f"Memory: {run_result.memory_usage:.2f}MB")
        print(f"Quality: {run_result.solution_quality:.2f}")
        if run_result.deregistration_risk:
            print("WARNING: Deregistration Risk!")
    
    def generate_report(self, results, start_time, current_time):
        """Generate interim report."""
        duration = current_time - start_time
        success_rate = (
            results['successful_solutions'] / results['total_problems'] * 100
            if results['total_problems'] > 0 else 0
        )
        
        report = (
            f"\nInterim Report ({duration.total_seconds()/3600:.1f} hours)\n"
            f"Total Problems: {results['total_problems']}\n"
            f"Success Rate: {success_rate:.2f}%\n"
            f"Deregistration Risks: {results['deregistration_risks']}\n\n"
            "Algorithm Usage:\n"
        )
        
        for algo, count in results['algorithm_usage'].items():
            report += f"{algo}: {count} times\n"
        
        self.logger.info(report)
        print(report)
    
    def generate_final_report(self, results, start_time, end_time):
        """Generate comprehensive final report."""
        # Similar to interim report but more detailed
        # Save to file for client review
        report_path = f"results/stress_test_{start_time.strftime('%Y%m%d_%H%M')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("TSP Optimizer Stress Test Results\n")
            f.write("================================\n\n")
            
            # Test duration
            duration = end_time - start_time
            f.write(f"Test Duration: {duration.total_seconds()/3600:.1f} hours\n")
            f.write(f"Start Time: {start_time}\n")
            f.write(f"End Time: {end_time}\n\n")
            
            # Overall statistics
            f.write("Overall Performance:\n")
            f.write(f"Total Problems: {results['total_problems']}\n")
            success_rate = (
                results['successful_solutions'] / results['total_problems'] * 100
                if results['total_problems'] > 0 else 0
            )
            f.write(f"Success Rate: {success_rate:.2f}%\n")
            f.write(f"Deregistration Risks: {results['deregistration_risks']}\n\n")
            
            # Performance by size
            f.write("Performance by Problem Size:\n")
            for size, stats in sorted(results['performance_by_size'].items()):
                f.write(f"\nSize {size}:\n")
                size_success_rate = (
                    stats['successful'] / stats['total'] * 100
                    if stats['total'] > 0 else 0
                )
                f.write(f"  Success Rate: {size_success_rate:.2f}%\n")
                f.write(f"  Average Time: {stats['avg_time']:.3f}s\n")
                f.write(f"  Maximum Time: {stats['max_time']:.3f}s\n")
            
            # Algorithm usage
            f.write("\nAlgorithm Usage:\n")
            for algo, count in results['algorithm_usage'].items():
                percentage = count / results['total_problems'] * 100
                f.write(f"{algo}: {count} times ({percentage:.1f}%)\n")

if __name__ == "__main__":
    # Run a shorter test first (1 hour)
    stress_test = StressTest(duration_hours=1)
    results = stress_test.run_stress_test()