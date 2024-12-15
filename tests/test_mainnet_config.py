import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_handler import MainnetConfig

def test_config():
    print("\nTesting Mainnet Configuration...")
    
    config = MainnetConfig()
    
    # Test time limits
    print(f"\nTime Limits:")
    print(f"Small instance (15 nodes): {config.get_time_limit(15):.2f}s")
    print(f"Medium instance (35 nodes): {config.get_time_limit(35):.2f}s")
    print(f"Large instance (80 nodes): {config.get_time_limit(80):.2f}s")
    
    # Test performance thresholds
    print(f"\nPerformance Thresholds:")
    print(f"Solution quality threshold: {config.performance.solution_quality_threshold}")
    print(f"Minimum success rate: {config.performance.min_success_rate}")
    
    print("\nConfiguration loaded successfully!")

if __name__ == "__main__":
    test_config()