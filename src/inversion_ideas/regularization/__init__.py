"""
Regularization classes.
"""

from ._general import TikhonovZero
from ._mesh_based import Flatness, Smallness, SparseSmallness

__all__ = [
    "Flatness",
    "Smallness",
    "SparseSmallness",
    "TikhonovZero",
]
