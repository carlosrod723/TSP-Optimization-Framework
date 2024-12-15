# tests/run_mainnet_validation.py

import sys
import os
import time
import logging
import signal
from contextlib import contextmanager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mainnet_validation import MainnetValidator

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Test timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def run_validation_tests():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\nStarting Mainnet Validation Tests...")
    print("Running quick validation check...")
    
    validator = MainnetValidator(test_duration_hours=1)
    
    problem_sizes = [10, 15, 20]
    network_loads = [0.3]
    
    for size in problem_sizes:
        print(f"\nTesting Problem Size: {size}")
        for load in network_loads:
            print(f"  Network Load: {load}")
            print(f"  Selecting algorithm...")
            
            try:
                with time_limit(10):  # Set 10-second timeout
                    start_time = time.time()
                    result = validator._test_problem(size, load)
                    elapsed = time.time() - start_time
                    
                    print(f"    Selected: {result['algorithm_used']}")
                    print(f"    Success: {result['success']}")
                    print(f"    Time: {result['execution_time']:.3f}s")
                    print(f"    Quality: {result['solution_quality']:.2f}")
                    print(f"    Total test time: {elapsed:.3f}s")
                    
                    if result['deregistration_risk']:
                        print("    WARNING: Deregistration Risk!")
                        
            except TimeoutException:
                print("    ERROR: Test timed out after 10 seconds!")
            except Exception as e:
                print(f"    ERROR: Test failed - {str(e)}")
                print(f"    Error type: {type(e)}")

if __name__ == "__main__":
    run_validation_tests()