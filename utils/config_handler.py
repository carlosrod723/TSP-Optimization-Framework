# Import necessary libraries and packages
import yaml
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MainnetConfig:
    """Mainnet configuration parameters."""
    
    class TimeLimits:
        def __init__(self, config: Dict):
            self.small_instance = config['small_instance']
            self.medium_instance = config['medium_instance']
            self.large_instance = config['large_instance']
    
    class ResourceLimits:
        def __init__(self, config: Dict):
            self.max_memory_mb = config['max_memory_mb']
            self.max_cpu_usage_percent = config['max_cpu_usage_percent']
    
    class Performance:
        def __init__(self, config: Dict):
            self.solution_quality_threshold = config['solution_quality_threshold']
            self.min_success_rate = config['min_success_rate']
            self.max_consecutive_failures = config['max_consecutive_failures']
            
    class Network:
        def __init__(self, config: Dict):
            self.min_latency = config['min_latency']
            self.max_latency = config['max_latency']
            self.packet_loss_rate = config['packet_loss_rate']
            self.connection_timeout = config['connection_timeout']
    
    def __init__(self, config_path: str = "config/mainnet_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['mainnet']
        
        self.time_limits = self.TimeLimits(config['time_limits'])
        self.resource_limits = self.ResourceLimits(config['resource_limits'])
        self.performance = self.Performance(config['performance'])
        self.problem_sizes = config['problem_sizes']
        self.network = self.Network(config['network'])
        self.testing = config['testing']
    
    def get_time_limit(self, problem_size: int) -> float:
        """Get time limit for a given problem size."""
        if problem_size <= self.problem_sizes['small']:
            return self.time_limits.small_instance
        elif problem_size <= self.problem_sizes['medium']:
            return self.time_limits.medium_instance
        return self.time_limits.large_instance
    
    def is_within_resource_limits(self, memory_usage: float, cpu_usage: float) -> bool:
        """Check if resource usage is within limits."""
        return (memory_usage <= self.resource_limits.max_memory_mb and 
                cpu_usage <= self.resource_limits.max_cpu_usage_percent)
    
    def is_solution_acceptable(self, solution_quality: float) -> bool:
        """Check if solution quality meets threshold."""
        return solution_quality <= self.performance.solution_quality_threshold