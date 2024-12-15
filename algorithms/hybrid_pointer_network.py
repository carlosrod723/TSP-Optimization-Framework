# algorithms/hybrid_pointer_network.py

import torch
import numpy as np
from typing import List, Optional, Tuple
import logging
from dataclasses import dataclass
from .greedy import OptimizedGreedyTSP
from models.pointer_network.model import PointerNetwork

@dataclass
class HybridResult:
    path: List[int]
    total_distance: float
    method_used: str
    improvements: int

class HybridPointerNetworkTSP:
    def __init__(
        self,
        hidden_size: int = 128,
        model_path: Optional[str] = None,
        use_cuda: bool = torch.cuda.is_available()
    ):
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.hidden_size = hidden_size
        self.model = PointerNetwork(
            input_size=2,  # x, y coordinates
            hidden_size=hidden_size
        ).to(self.device)
        self.greedy_solver = OptimizedGreedyTSP()
        self.logger = logging.getLogger(__name__)
        
        if model_path:
            self._load_model(model_path)
        
        self.model.eval()
    
    def solve(self, distances: np.ndarray) -> HybridResult:
        """Solve TSP using hybrid approach."""
        n = len(distances)
        
        try:
            # Try neural approach first
            if n <= 100:  # Size limit for neural approach
                coords = self._get_coords(distances)
                neural_solution = self._solve_neural(coords)
                
                if neural_solution is not None:
                    path, distance = neural_solution
                    return HybridResult(
                        path=path,
                        total_distance=distance,
                        method_used='neural',
                        improvements=0
                    )
        except Exception as e:
            self.logger.warning(f"Neural solution failed: {str(e)}")
        
        # Fallback to traditional approach
        greedy_result = self.greedy_solver.solve(distances)
        
        return HybridResult(
            path=greedy_result.path,
            total_distance=greedy_result.total_distance,
            method_used='traditional',
            improvements=greedy_result.improvements_made
        )
    
    def _solve_neural(self, coords: np.ndarray) -> Optional[Tuple[List[int], float]]:
        """Solve using neural network approach."""
        with torch.no_grad():
            inputs = torch.FloatTensor(coords).unsqueeze(0).to(self.device)
            outputs = self.model(inputs)
            
            # Convert outputs to tour
            tour = outputs.max(2)[1].squeeze().cpu().numpy()
            
            # Calculate tour length
            tour_length = sum(
                np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]])
                for i in range(len(tour) - 1)
            )
            
            return tour, tour_length
    
    def _get_coords(self, distances: np.ndarray) -> np.ndarray:
        """Convert distance matrix to 2D coordinates using MDS."""
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        return mds.fit_transform(distances)
    
    def _load_model(self, path: str):
        """Load pre-trained model weights."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )