# large_scale_handlers/batch_processor.py

import numpy as np
from typing import List, Dict, Optional, Generator, Tuple
import logging
from dataclasses import dataclass
import psutil
import time
from pathlib import Path

@dataclass
class BatchResult:
    """Results from batch processing."""
    batch_id: int
    partial_path: List[int]
    batch_distance: float
    processing_time: float
    memory_usage: float

class BatchProcessor:
    """
    Handles processing of large TSP instances in batches.
    Manages memory efficiently for very large datasets.
    """
    
    def __init__(
        self,
        max_batch_size: int = 1000,
        memory_threshold_mb: int = 4096,
        temp_storage_path: str = "temp_batches"
    ):
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold_mb
        self.temp_path = Path(temp_storage_path)
        self.temp_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def process_large_instance(
        self,
        distance_matrix: np.ndarray,
        save_intermediates: bool = True
    ) -> Generator[BatchResult, None, None]:
        """
        Process large TSP instance in batches.
        
        Args:
            distance_matrix: Full distance matrix
            save_intermediates: Whether to save intermediate results
            
        Yields:
            BatchResult for each processed batch
        """
        n = len(distance_matrix)
        num_batches = (n + self.max_batch_size - 1) // self.max_batch_size
        
        try:
            for batch_id in range(num_batches):
                # Process batch
                batch_result = self._process_batch(
                    distance_matrix,
                    batch_id
                )
                
                # Save intermediate results if requested
                if save_intermediates:
                    self._save_batch_result(batch_result)
                
                yield batch_result
                
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            raise
        finally:
            # Cleanup temporary files
            if not save_intermediates:
                self._cleanup_temp_files()
    
    def _process_batch(
        self,
        distance_matrix: np.ndarray,
        batch_id: int
    ) -> BatchResult:
        """Process a single batch of the problem."""
        start_time = time.time()
        n = len(distance_matrix)
        
        # Calculate batch boundaries
        start_idx = batch_id * self.max_batch_size
        end_idx = min(start_idx + self.max_batch_size, n)
        
        try:
            # Extract batch submatrix
            batch_matrix = distance_matrix[
                start_idx:end_idx,
                start_idx:end_idx
            ]
            
            # Solve batch
            from algorithms.greedy import OptimizedGreedyTSP
            solver = OptimizedGreedyTSP()
            result = solver.solve(batch_matrix)
            
            # Adjust indices to global coordinates
            global_path = [i + start_idx for i in result.path]
            
            processing_time = time.time() - start_time
            memory_usage = self._get_current_memory()
            
            return BatchResult(
                batch_id=batch_id,
                partial_path=global_path,
                batch_distance=result.total_distance,
                processing_time=processing_time,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_id}: {str(e)}")
            raise
    
    def _save_batch_result(self, result: BatchResult):
        """Save intermediate batch results to disk."""
        save_path = self.temp_path / f"batch_{result.batch_id}.npy"
        try:
            np.save(save_path, {
                'path': result.partial_path,
                'distance': result.batch_distance,
                'time': result.processing_time,
                'memory': result.memory_usage
            })
        except Exception as e:
            self.logger.error(f"Error saving batch result: {str(e)}")
    
    def load_saved_results(self) -> List[BatchResult]:
        """Load all saved batch results."""
        results = []
        for file_path in sorted(self.temp_path.glob("batch_*.npy")):
            try:
                data = np.load(file_path, allow_pickle=True).item()
                batch_id = int(file_path.stem.split('_')[1])
                results.append(BatchResult(
                    batch_id=batch_id,
                    partial_path=data['path'],
                    batch_distance=data['distance'],
                    processing_time=data['time'],
                    memory_usage=data['memory']
                ))
            except Exception as e:
                self.logger.error(f"Error loading batch result: {str(e)}")
        
        return sorted(results, key=lambda x: x.batch_id)
    
    def merge_batch_results(
        self,
        results: List[BatchResult],
        distance_matrix: np.ndarray
    ) -> Tuple[List[int], float]:
        """Merge batch results into complete solution."""
        complete_path = []
        total_distance = 0
        
        for i, result in enumerate(results):
            if i > 0:
                # Add connection between batches
                prev_end = complete_path[-1]
                curr_start = result.partial_path[0]
                total_distance += distance_matrix[prev_end][curr_start]
            
            complete_path.extend(
                result.partial_path[:-1] if i < len(results)-1 
                else result.partial_path
            )
            total_distance += result.batch_distance
        
        return complete_path, total_distance
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            for file_path in self.temp_path.glob("batch_*.npy"):
                file_path.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {str(e)}")