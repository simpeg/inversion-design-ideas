"""
Ideas for inversion framework.
"""
from ._version import __version__
from .objective_function import Combo, Objective, Scaled
from .data_misfit import DataMisfit
from .simulation import Simulation
from .regularization import TikhonovZero

__all__ = [
    "Combo",
    "DataMisfit",
    "Objective",
    "Scaled",
    "Simulation",
    "TikhonovZero",
    "__version__",
]
