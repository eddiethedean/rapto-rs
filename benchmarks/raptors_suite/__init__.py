"""
Benchmark suite for Raptors.
"""
import sys
from pathlib import Path

# Add the python directory to sys.path if not already there
# This ensures ASV can find the raptors module during discovery
_raptors_path = Path(__file__).parent.parent.parent / "python"
if str(_raptors_path) not in sys.path:
    sys.path.insert(0, str(_raptors_path))

