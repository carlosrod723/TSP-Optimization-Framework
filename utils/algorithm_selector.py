import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass
import numpy as np

from algorithms.greedy import OptimizedGreedyTSP
from algorithms.beam_search import EnhancedBeamSearchTSP
from algorithms.dynamic_programming import DynamicProgrammingTSP
from .config_handler import MainnetConfig

@dataclass
class AlgorithmSelection:
    """Result of algorithm selection process."""
    algorithm: Any
    name: str
    estimated_time: float
    expected_quality: float
    reason: str

class AlgorithmSelector:
    """
    Intelligent algorithm selector for TSP problems.
    Chooses the best algorithm based on problem characteristics
    and network conditions.
    """
    
    def __init__(self, config_path: str = "config/mainnet_config.yaml"):
        self.config = MainnetConfig(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize algorithms
        self.algorithms = {
            'greedy': OptimizedGreedyTSP(),
            'beam_search': EnhancedBeamSearchTSP(),
            'dynamic_programming': DynamicProgrammingTSP()
        }
        
        # Performance characteristics based on testing
        self.performance_profiles = {
            'greedy': {
                'max_reliable_size': float('inf'),
                'time_factor': 0.001,  # seconds per city
                'quality_factor': 1.3   # typical ratio to optimal
            },
            'beam_search': {
                'max_reliable_size': 100,
                'time_factor': 0.05,
                'quality_factor': 1.1
            },
            'dynamic_programming': {
                'max_reliable_size': 20,
                'time_factor': 0.1,
                'quality_factor': 1.0
            }
        }
    
    def select_algorithm(
        self,
        problem_size: int,
        available_time: float,
        network_load: float = 0.5
    ) -> AlgorithmSelection:
        """
        Select the best algorithm based on current conditions.
        """
        print(f"\nAlgorithm Selection Debug:")
        print(f"Problem size: {problem_size}")
        print(f"Available time: {available_time}")
        print(f"Network load: {network_load}")

        # Try for Dynamic Programming
        if problem_size <= 15 and available_time >= 1.0:
            print("Selecting Dynamic Programming (small instance)")
            return AlgorithmSelection(
                algorithm=self.algorithms['dynamic_programming'],
                name='dynamic_programming',
                estimated_time=problem_size * self.performance_profiles['dynamic_programming']['time_factor'],
                expected_quality=self.performance_profiles['dynamic_programming']['quality_factor'],
                reason="Small problem size allows for optimal solution"
            )
        
        # Try for Beam Search
        if problem_size <= 50 and available_time >= 0.5:
            estimated_time = problem_size * self.performance_profiles['beam_search']['time_factor']
            if estimated_time <= available_time * 0.8:
                print("Selecting Beam Search (medium instance)")
                return AlgorithmSelection(
                    algorithm=self.algorithms['beam_search'],
                    name='beam_search',
                    estimated_time=estimated_time,
                    expected_quality=self.performance_profiles['beam_search']['quality_factor'],
                    reason="Good balance of quality and speed for medium size"
                )
        
        # Fallback to Greedy
        print("Selecting Greedy (fallback)")
        return AlgorithmSelection(
            algorithm=self.algorithms['greedy'],
            name='greedy',
            estimated_time=problem_size * self.performance_profiles['greedy']['time_factor'],
            expected_quality=self.performance_profiles['greedy']['quality_factor'],
            reason="Fast solution required for large problem or time constraint"
        )
    
    def _get_selection_reason(
        self,
        name: str,
        size: int,
        est_time: float,
        limit: float
    ) -> str:
        """Generate explanation for algorithm selection."""
        if name == 'dynamic_programming' and size <= 20:
            return "Small problem size allows for optimal solution"
        elif name == 'beam_search' and size <= 100:
            return "Good balance of quality and speed for medium size"
        elif name == 'greedy':
            return "Fast solution required for large problem or time constraint"
        return "Selected based on time and quality requirements"
    
    def _get_fallback_selection(
        self,
        size: int,
        available_time: float
    ) -> AlgorithmSelection:
        """Provide fallback algorithm when no algorithm meets constraints."""
        return AlgorithmSelection(
            algorithm=self.algorithms['greedy'],
            name='greedy',
            estimated_time=size * self.performance_profiles['greedy']['time_factor'],
            expected_quality=self.performance_profiles['greedy']['quality_factor'],
            reason="Fallback selection due to constraints"
        )