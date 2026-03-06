"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
VALLEX_ROOT = os.path.dirname(PROJECT_ROOT)


def setup():
    """Expose project paths and create local output directories."""
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    if VALLEX_ROOT not in sys.path:
        sys.path.append(VALLEX_ROOT)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


setup()
