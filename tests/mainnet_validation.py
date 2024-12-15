# tests/mainnet_validation.py
import sys
import os
import time
import logging
from typing import Dict, Any
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.algorithm_selector import AlgorithmSelector
from utils.data_generator import TSPDataGenerator
from utils.mainnet_simulator import MainnetSimulator

class MainnetValidator:
    def __init__(self, test_duration_hours: int = 168):  # 7 days
        self.selector = AlgorithmSelector()
        self.generator = TSPDataGenerator()
        self.simulator = MainnetSimulator()
        self.test_duration = test_duration_hours
        self.logger = logging.getLogger(__name__)
        
    def run_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation simulating 7 days of mainnet operation.
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.test_duration)
        
        results = {
            'total_problems': 0,
            'successful_solutions': 0,
            'average_quality': 0,
            'performance_by_size': {},
            'deregistration_risks': [],
            'execution_times': []
        }
        
        # Simulate different problem sizes and network conditions
        problem_sizes = [10, 20, 50, 100, 200]
        network_loads = [0.3, 0.5, 0.8]  # Different network conditions
        
        for hour in range(self.test_duration):
            current_time = start_time + timedelta(hours=hour)
            self.logger.info(f"Testing hour {hour}/{self.test_duration}")
            
            # Test each problem size under different conditions
            for size in problem_sizes:
                for load in network_loads:
                    result = self._test_problem(size, load)
                    self._update_results(results, result, size)
                    
                    if result.get('deregistration_risk'):
                        self.logger.warning(
                            f"Deregistration risk detected at hour {hour} "
                            f"for size {size} under load {load}"
                        )
        
        self._generate_report(results, start_time, end_time)
        return results
    
    def _test_problem(self, size: int, network_load: float) -> Dict[str, Any]:
        """Test a single problem instance."""
        instance = self.generator.generate_euclidean_instance(size)
        
        # Get time limit from mainnet conditions
        available_time = self.simulator.get_time_limit(size)
        
        # Select algorithm
        selection = self.selector.select_algorithm(
            size,
            available_time,
            network_load
        )
        
        # Run solution under mainnet constraints
        result = self.simulator.run_with_constraints(
            selection.algorithm.solve,
            instance.distances,
            time_limit=available_time
        )
        
        return {
            'success': result.success,
            'execution_time': result.execution_time,
            'solution_quality': result.solution_quality,
            'deregistration_risk': result.deregistration_risk,
            'algorithm_used': selection.name
        }
    
    def _update_results(
        self,
        results: Dict[str, Any],
        test_result: Dict[str, Any],
        size: int
    ):
        """Update aggregate results with new test result."""
        results['total_problems'] += 1
        results['successful_solutions'] += int(test_result['success'])
        results['execution_times'].append(test_result['execution_time'])
        
        if str(size) not in results['performance_by_size']:
            results['performance_by_size'][str(size)] = {
                'total': 0,
                'successful': 0,
                'avg_quality': 0,
                'avg_time': 0
            }
        
        size_stats = results['performance_by_size'][str(size)]
        size_stats['total'] += 1
        size_stats['successful'] += int(test_result['success'])
        size_stats['avg_quality'] = (
            (size_stats['avg_quality'] * (size_stats['total'] - 1) +
             test_result['solution_quality']) / size_stats['total']
        )
        size_stats['avg_time'] = (
            (size_stats['avg_time'] * (size_stats['total'] - 1) +
             test_result['execution_time']) / size_stats['total']
        )
    
    def _generate_report(
        self,
        results: Dict[str, Any],
        start_time: datetime,
        end_time: datetime
    ):
        """Generate comprehensive validation report."""
        report_path = f"results/mainnet_validation_{start_time.strftime('%Y%m%d_%H%M')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("Mainnet Validation Report\n")
            f.write("========================\n\n")
            f.write(f"Test Period: {start_time} to {end_time}\n")
            f.write(f"Total Problems: {results['total_problems']}\n")
            f.write(f"Success Rate: {results['successful_solutions']/results['total_problems']*100:.2f}%\n\n")
            
            f.write("Performance by Problem Size:\n")
            for size, stats in results['performance_by_size'].items():
                f.write(f"\nSize {size}:\n")
                f.write(f"  Success Rate: {stats['successful']/stats['total']*100:.2f}%\n")
                f.write(f"  Average Quality: {stats['avg_quality']:.2f}\n")
                f.write(f"  Average Time: {stats['avg_time']:.3f}s\n")
            
            f.write("\nDeregistration Risk Assessment:\n")
            risk_rate = len(results['deregistration_risks']) / results['total_problems']
            f.write(f"Risk Rate: {risk_rate*100:.2f}%\n")