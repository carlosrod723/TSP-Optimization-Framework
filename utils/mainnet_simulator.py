# utils/mainnet_simulator.py

import time
import random
import logging
import psutil
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
from .config_handler import MainnetConfig

@dataclass
class SimulationResult:
    """Results from a simulation run."""
    success: bool
    execution_time: float
    memory_usage: float
    cpu_usage: float
    solution_quality: float
    deregistration_risk: bool = False
    error_message: Optional[str] = None

class MainnetSimulator:
    def __init__(self, config_path: str = "config/mainnet_config.yaml"):
        """Initialize the mainnet simulator with configuration."""
        self.config = MainnetConfig(config_path)
        self.logger = logging.getLogger(__name__)
        self.consecutive_failures = 0
        
    def get_time_limit(self, problem_size: int) -> float:
        """
        Get the time limit for a given problem size based on mainnet conditions.
        """
        if problem_size <= self.config.problem_sizes['small']:
            return self.config.time_limits.small_instance
        elif problem_size <= self.config.problem_sizes['medium']:
            return self.config.time_limits.medium_instance
        else:
            return self.config.time_limits.large_instance
    
    def simulate_network_delay(self):
        """Simulate network latency."""
        delay = random.uniform(
            self.config.network.min_latency,
            self.config.network.max_latency
        ) / 1000.0  # Convert ms to seconds
        time.sleep(delay)
        
    def check_resource_usage(self) -> tuple[float, float]:
        """Monitor current CPU and memory usage."""
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        cpu_usage = psutil.cpu_percent()
        return memory_usage, cpu_usage
    
    def run_with_constraints(
        self,
        solver_func: Callable,
        problem_instance: np.ndarray,
        time_limit: Optional[float] = None,
        optimal_value: Optional[float] = None
    ) -> SimulationResult:
        """
        Run a TSP solver under mainnet conditions.
        
        Args:
            solver_func: The TSP solver function to test
            problem_instance: The TSP problem instance
            time_limit: Optional override for time limit
            optimal_value: Known optimal solution value (if available)
        """
        # Simulate network delay before starting
        self.simulate_network_delay()
        
        # Get time limit if not provided
        if time_limit is None:
            time_limit = self.get_time_limit(len(problem_instance))
        
        start_time = time.perf_counter()
        
        try:
            # Run solver with time limit
            result = solver_func(problem_instance)
            
            execution_time = time.perf_counter() - start_time
            memory_usage, cpu_usage = self.check_resource_usage()
            
            # Check if execution time exceeded limit
            if execution_time > time_limit:
                self.consecutive_failures += 1
                return SimulationResult(
                    success=False,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    solution_quality=float('inf'),
                    deregistration_risk=self.would_be_deregistered(),
                    error_message="Time limit exceeded"
                )
            
            # Calculate solution quality
            solution_quality = (
                result.total_distance / optimal_value 
                if optimal_value 
                else 1.0
            )
            
            # Check if solution meets quality threshold
            success = self.config.performance.solution_quality_threshold >= solution_quality
            
            # Update consecutive failures counter
            if success:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            
            return SimulationResult(
                success=success,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                solution_quality=solution_quality,
                deregistration_risk=self.would_be_deregistered()
            )
            
        except Exception as e:
            self.consecutive_failures += 1
            return SimulationResult(
                success=False,
                execution_time=time.perf_counter() - start_time,
                memory_usage=0,
                cpu_usage=0,
                solution_quality=float('inf'),
                deregistration_risk=self.would_be_deregistered(),
                error_message=str(e)
            )
    
    def would_be_deregistered(self) -> bool:
        """Check if current performance would lead to deregistration."""
        return (
            self.consecutive_failures >= 
            self.config.performance.max_consecutive_failures
        )