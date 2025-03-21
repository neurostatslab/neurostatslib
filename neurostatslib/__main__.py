import sys
from .cli import neurostatslib

if __name__ == "__main__":
    """
    Run neurostatslib as a python module
    e.g. python -m neurostatslib
    """
    sys.exit(neurostatslib())
