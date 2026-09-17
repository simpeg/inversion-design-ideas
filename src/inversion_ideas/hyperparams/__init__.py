"""
Submodule for custom hyperparameter classes.
"""

from ._multiplier import CooledMultiplier
from ._sensitivity_weights import SensitivityWeights

__all__ = ["CooledMultiplier", "SensitivityWeights"]
