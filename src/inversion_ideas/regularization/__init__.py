"""
Regularization classes.
"""

from ._general import TikhonovZero
from ._mesh_based import Flatness, Smallness

__all__ = [
    "Flatness",
    "Smallness",
    "TikhonovZero",
]
