import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.algorithm_selector import AlgorithmSelector

def test_algorithm_selector():
    print("\nTesting Algorithm Selector...")
    
    selector = AlgorithmSelector()
    
    # Test cases
    test_cases = [
        {
            'size': 10,
            'time': 1.0,
            'load': 0.3,
            'expected': 'dynamic_programming'
        },
        {
            'size': 50,
            'time': 2.0,
            'load': 0.5,
            'expected': 'beam_search'
        },
        {
            'size': 200,
            'time': 0.5,
            'load': 0.8,
            'expected': 'greedy'
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting with size={case['size']}, time={case['time']}s, load={case['load']}")
        
        selection = selector.select_algorithm(
            case['size'],
            case['time'],
            case['load']
        )
        
        print(f"Selected: {selection.name}")
        print(f"Estimated time: {selection.estimated_time:.3f}s")
        print(f"Expected quality factor: {selection.expected_quality:.2f}")
        print(f"Reason: {selection.reason}")
        
        assert selection.name == case['expected'], \
            f"Expected {case['expected']}, got {selection.name}"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_algorithm_selector()
    