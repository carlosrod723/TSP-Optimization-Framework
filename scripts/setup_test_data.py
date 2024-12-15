# Import necessary libraries and packages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import TSPDataGenerator
from pathlib import Path
import numpy as np
import json
import tsplib95

def setup_test_data():
    # Create data directories if they don't exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # 1. Generate synthetic instances
    generator = TSPDataGenerator(seed=42)
    synthetic_instances = []
    
    # Generate instances of different sizes
    sizes = [10, 20, 50, 100, 200]
    for size in sizes:
        # Generate both Euclidean and asymmetric instances
        euclidean = generator.generate_euclidean_instance(size)
        asymmetric = generator.generate_asymmetric_instance(size)
        
        # Save instances
        for instance in [euclidean, asymmetric]:
            instance_path = f"data/processed/{instance.name}.npy"
            np.save(instance_path, instance.distances)
            synthetic_instances.append({
                "name": instance.name,
                "size": instance.size,
                "path": instance_path,
                "type": "synthetic"
            })
    
    # 2. Download some TSPLIB instances
    tsplib_instances = []
    test_instances = ["a280", "berlin52", "eil51", "st70"]  # Common test instances
    
    for name in test_instances:
        try:
            problem = tsplib95.load(f"https://raw.githubusercontent.com/mastqe/tsplib/master/data/{name}.tsp")
            
            # Convert to distance matrix
            n = len(problem.node_coords)
            distances = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x1, y1 = problem.node_coords[i+1]
                        x2, y2 = problem.node_coords[j+1]
                        distances[i][j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Save instance
            instance_path = f"data/processed/{name}.npy"
            np.save(instance_path, distances)
            
            tsplib_instances.append({
                "name": name,
                "size": n,
                "path": instance_path,
                "type": "tsplib",
                "optimal_value": problem.optimal_value if hasattr(problem, 'optimal_value') else None
            })
            
        except Exception as e:
            print(f"Error downloading {name}: {str(e)}")
    
    # Save metadata
    metadata = {
        "synthetic_instances": synthetic_instances,
        "tsplib_instances": tsplib_instances
    }
    
    with open("data/processed/instances_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {len(synthetic_instances)} synthetic instances")
    print(f"Downloaded {len(tsplib_instances)} TSPLIB instances")
    print("Metadata saved to data/processed/instances_metadata.json")

if __name__ == "__main__":
    setup_test_data()