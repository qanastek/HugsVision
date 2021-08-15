import os

__all__ = []

# For each file in directory
for filename in os.listdir(os.path.dirname(__file__)):

    # Get filename
    filename = os.path.basename(filename)

    # Check if is a python script
    if filename.endswith(".py") and not filename.startswith("__"):
        __all__.append(filename[:-3])

from . import *