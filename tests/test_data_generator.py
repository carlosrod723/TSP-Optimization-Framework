# Import necessary libraries and packages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import TSPDataGenerator
import numpy as np

def test_data_generator():
    # Initialize generator
    generator = TSPDataGenerator(seed=42)
    
    # Generate a small instance
    instance = generator.generate_euclidean_instance(size=5)
    
    # Basic checks
    print("\nTesting data generator...")
    print(f"Instance size: {instance.size}")
    print(f"Distance matrix shape: {instance.distances.shape}")
    print("\nDistance matrix:")
    print(instance.distances)
    
    # Generate test suite
    test_suite = generator.generate_test_suite(sizes=[5, 10])
    print(f"\nGenerated test suite with {len(test_suite)} instances")
    for inst in test_suite:
        print(f"Instance: {inst.name}, Size: {inst.size}")

if __name__ == "__main__":
    test_data_generator()