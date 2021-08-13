import os

from . import dataio
from . import inference
from . import nnet
from . import utils

with open(os.path.join(os.path.dirname(__file__), "version.txt")) as f:
    version = f.read().strip()

__version__ = version