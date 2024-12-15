# large_scale_handlers/memory_optimizer.py

import numpy as np
from typing import Dict, Optional, Tuple
import logging
import psutil
from dataclasses import dataclass
import gc
import os

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    current_usage_mb: float
    peak_usage_mb: float
    available_mb: float
    swap_usage_mb: float

class MemoryOptimizer:
    """
    Handles memory optimization for large TSP instances.
    Implements memory-efficient data structures and disk-based operations.
    """
    
    def __init__(
        self,
        max_memory_mb: int = 4096,
        use_mmap: bool = True,
        compression: bool = True
    ):
        self.max_memory = max_memory_mb
        self.use_mmap = use_mmap
        self.compression = compression
        self.logger = logging.getLogger(__name__)
        self.temp_files = []
    
    def optimize_matrix(
        self,
        distance_matrix: np.ndarray,
        precision: str = 'float32'
    ) -> np.ndarray:
        """
        Optimize memory usage of distance matrix.
        
        Args:
            distance_matrix: Original distance matrix
            precision: Data type precision ('float32' or 'float64')
        
        Returns:
            Memory-optimized matrix
        """
        # Convert to memory-efficient data type
        optimized = distance_matrix.astype(precision)
        
        if self.compression:
            # Apply compression if enabled
            optimized = self._compress_matrix(optimized)
        
        if self.use_mmap and optimized.nbytes > self.max_memory * 1024 * 1024:
            # Use memory mapping for large matrices
            return self._create_mmap_matrix(optimized)
        
        return optimized
    
    def _compress_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Compress matrix to reduce memory usage."""
        # Exploit symmetry for symmetric matrices
        if np.allclose(matrix, matrix.T):
            # Store only upper triangle
            tri = np.triu(matrix)
            return tri
        return matrix
    
    def _create_mmap_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Create memory-mapped matrix."""
        temp_file = f"temp_matrix_{id(matrix)}.mmap"
        self.temp_files.append(temp_file)
        
        fp = np.memmap(
            temp_file,
            dtype=matrix.dtype,
            mode='w+',
            shape=matrix.shape
        )
        
        # Copy data to memory-mapped file
        fp[:] = matrix[:]
        return fp
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        vm = psutil.virtual_memory()
        
        return MemoryStats(
            current_usage_mb=memory_info.rss / 1024 / 1024,
            peak_usage_mb=memory_info.rss / 1024 / 1024,  # Changed from peak to rss
            available_mb=vm.available / 1024 / 1024,
            swap_usage_mb=psutil.swap_memory().used / 1024 / 1024
        )

    def optimize_memory_usage(self):
        """Optimize current memory usage."""
        # Force garbage collection
        gc.collect()
        
        # Clear memory cache if needed
        if self.get_memory_stats().current_usage_mb > self.max_memory * 0.9:
            self._clear_memory_cache()
    
    def _clear_memory_cache(self):
        """Clear memory cache."""
        try:
            if hasattr(os, 'sync'):
                os.sync()
            gc.collect()
        except Exception as e:
            self.logger.error(f"Error clearing memory cache: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.error(f"Error removing temp file {temp_file}: {str(e)}")
        self.temp_files.clear()