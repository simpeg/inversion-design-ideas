"""
Regularization classes.
"""

from ._general import TikhonovZero
from ._mesh_based import Flatness, Smallness, SparseFlatness, SparseSmallness

__all__ = [
    "Flatness",
    "Smallness",
    "SparseFlatness",
    "SparseSmallness",
    "TikhonovZero",
]
