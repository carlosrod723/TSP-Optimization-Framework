# large_scale_handlers/partition_handler.py

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.cluster import KMeans
import networkx as nx

@dataclass
class Partition:
    """Represents a partition of the original problem."""
    indices: List[int]           # Original indices of nodes in this partition
    distances: np.ndarray        # Distance matrix for this partition
    center: Tuple[float, float]  # Geometric center of partition
    boundary_nodes: List[int]    # Nodes that connect to other partitions

class PartitionHandler:
    """
    Handles sophisticated partitioning of large TSP instances.
    Implements multiple partitioning strategies and boundary management.
    """
    
    def __init__(
        self,
        max_partition_size: int = 1000,
        overlap_percentage: float = 0.1,
        strategy: str = 'kmeans'
    ):
        self.max_partition_size = max_partition_size
        self.overlap = overlap_percentage
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
    def create_partitions(
        self,
        distance_matrix: np.ndarray,
        coordinates: Optional[List[Tuple[float, float]]] = None
    ) -> List[Partition]:
        """
        Create optimal partitions based on problem structure.
        
        Args:
            distance_matrix: Complete distance matrix
            coordinates: Optional coordinate list for geometric partitioning
        
        Returns:
            List of Partition objects
        """
        if self.strategy == 'kmeans' and coordinates is not None:
            return self._kmeans_partitioning(distance_matrix, coordinates)
        elif self.strategy == 'spectral':
            return self._spectral_partitioning(distance_matrix)
        else:
            return self._geometric_partitioning(distance_matrix)
    
    def _kmeans_partitioning(
        self,
        distance_matrix: np.ndarray,
        coordinates: List[Tuple[float, float]]
    ) -> List[Partition]:
        """Partition using K-means clustering."""
        n = len(distance_matrix)
        n_clusters = max(2, n // self.max_partition_size)
        
        # Convert coordinates to numpy array
        coord_array = np.array(coordinates)
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42
        ).fit(coord_array)
        
        # Create partitions based on clusters
        partitions = []
        for i in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            
            # Add overlap nodes from nearby clusters
            extended_indices = self._add_overlap_nodes(
                cluster_indices,
                distance_matrix
            )
            
            partition = Partition(
                indices=extended_indices.tolist(),
                distances=distance_matrix[
                    np.ix_(extended_indices, extended_indices)
                ],
                center=tuple(kmeans.cluster_centers_[i]),
                boundary_nodes=self._find_boundary_nodes(
                    extended_indices,
                    distance_matrix
                )
            )
            partitions.append(partition)
        
        return partitions
    
    def _spectral_partitioning(
        self,
        distance_matrix: np.ndarray
    ) -> List[Partition]:
        """Partition using spectral clustering."""
        # Create graph from distance matrix
        G = nx.from_numpy_array(distance_matrix)
        
        # Calculate number of partitions needed
        n = len(distance_matrix)
        n_partitions = max(2, n // self.max_partition_size)
        
        # Perform spectral clustering
        try:
            partition_labels = nx.spectral_clustering(
                G,
                n_clusters=n_partitions
            )
        except Exception:
            self.logger.warning(
                "Spectral clustering failed, falling back to geometric"
            )
            return self._geometric_partitioning(distance_matrix)
        
        # Create partitions
        partitions = []
        for i in range(n_partitions):
            partition_indices = np.where(partition_labels == i)[0]
            extended_indices = self._add_overlap_nodes(
                partition_indices,
                distance_matrix
            )
            
            partition = Partition(
                indices=extended_indices.tolist(),
                distances=distance_matrix[
                    np.ix_(extended_indices, extended_indices)
                ],
                center=self._calculate_center(
                    extended_indices,
                    distance_matrix
                ),
                boundary_nodes=self._find_boundary_nodes(
                    extended_indices,
                    distance_matrix
                )
            )
            partitions.append(partition)
        
        return partitions
    
    def _geometric_partitioning(
        self,
        distance_matrix: np.ndarray
    ) -> List[Partition]:
        """Simple geometric partitioning based on distance matrix."""
        n = len(distance_matrix)
        partition_size = self.max_partition_size
        partitions = []
        
        for i in range(0, n, partition_size):
            end_idx = min(i + partition_size, n)
            base_indices = np.arange(i, end_idx)
            
            # Add overlap nodes
            extended_indices = self._add_overlap_nodes(
                base_indices,
                distance_matrix
            )
            
            partition = Partition(
                indices=extended_indices.tolist(),
                distances=distance_matrix[
                    np.ix_(extended_indices, extended_indices)
                ],
                center=self._calculate_center(
                    extended_indices,
                    distance_matrix
                ),
                boundary_nodes=self._find_boundary_nodes(
                    extended_indices,
                    distance_matrix
                )
            )
            partitions.append(partition)
        
        return partitions
    
    def _add_overlap_nodes(
        self,
        base_indices: np.ndarray,
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """Add overlapping nodes to partition."""
        n = len(distance_matrix)
        overlap_size = int(len(base_indices) * self.overlap)
        
        if overlap_size == 0:
            return base_indices
            
        # Find closest nodes to partition
        mask = np.ones(n, dtype=bool)
        mask[base_indices] = False
        
        min_distances = np.min(
            distance_matrix[base_indices][:, mask],
            axis=0
        )
        closest_indices = np.argsort(min_distances)[:overlap_size]
        
        # Add overlap nodes
        extended_indices = np.concatenate([
            base_indices,
            np.where(mask)[0][closest_indices]
        ])
        
        return np.unique(extended_indices)
    
    def _find_boundary_nodes(
        self,
        partition_indices: List[int],
        distance_matrix: np.ndarray
    ) -> List[int]:
        """Identify boundary nodes in partition."""
        n = len(distance_matrix)
        partition_set = set(partition_indices)
        boundary = []
        
        for idx in partition_indices:
            # Check connections to nodes outside partition
            for j in range(n):
                if j not in partition_set and distance_matrix[idx][j] > 0:
                    boundary.append(idx)
                    break
        
        return boundary
    
    def _calculate_center(
        self,
        indices: List[int],
        distance_matrix: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate geometric center of partition."""
        # Use MDS to get approximate coordinates
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed')
        
        sub_matrix = distance_matrix[np.ix_(indices, indices)]
        coordinates = mds.fit_transform(sub_matrix)
        
        # Return center as tuple
        center = coordinates.mean(axis=0)
        return (float(center[0]), float(center[1]))