"""
Regularization classes.
"""

from ._general import TikhonovZero
from ._mesh_based import Smallness, Flatness

__all__ = [
    "Smallness",
    "Flatness",
    "TikhonovZero",
]
