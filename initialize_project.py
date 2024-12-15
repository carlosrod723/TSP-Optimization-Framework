# Import necessary libraries and packages
import os
from pathlib import Path

def create_project_structure():
    """Create the full project directory structure and initial files."""
    
    # Define the directory structure
    directories = [
        'data/raw',
        'data/processed',
        'algorithms',
        'models/pointer_network/checkpoints',
        'utils',
        'tests',
        'logs',
        'results/visualizations',
        'notebooks',
        'docs',
        'scripts'
    ]
    
    # Create directories
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create initial files
    initial_files = {
        'algorithms/__init__.py': '',
        'algorithms/greedy.py': '"""Implementation of the Greedy TSP solver."""\n',
        'algorithms/dynamic_programming.py': '"""Implementation of the Dynamic Programming TSP solver."""\n',
        'algorithms/beam_search.py': '"""Implementation of the Beam Search TSP solver."""\n',
        
        'models/__init__.py': '',
        'models/pointer_network/__init__.py': '',
        
        'utils/__init__.py': '',
        'utils/data_generator.py': '"""Utilities for generating TSP test data."""\n',
        'utils/visualizer.py': '"""Utilities for visualizing TSP solutions."""\n',
        'utils/metrics.py': '"""Performance metrics calculation utilities."""\n',
        
        'tests/__init__.py': '',
        'tests/test_greedy.py': '"""Test cases for the Greedy TSP solver."""\n',
        'tests/test_utils.py': '"""Test cases for utility functions."""\n',
        
        'docs/README.md': '# TSP Optimizer Documentation\n\nThis directory contains project documentation.',
        'scripts/run_tests.sh': '#!/bin/bash\npytest tests/',
        
        '.gitignore': '''
venv/
__pycache__/
*.pyc
.DS_Store
logs/*.log
results/*.json
*.egg-info/
dist/
build/
'''.strip()
    }
    
    for file_path, content in initial_files.items():
        file = Path(file_path)
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(content)
    
    # Make scripts executable
    scripts_dir = Path('scripts')
    if scripts_dir.exists():
        for script in scripts_dir.glob('*.sh'):
            script.chmod(0o755)
    
    print("Project structure initialized successfully!")

if __name__ == "__main__":
    create_project_structure()