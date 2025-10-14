"""
Regularization classes.
"""

from ._general import TikhonovZero
from ._mesh_based import Flatness, Smallness
from ._sparse import SparseSmallness

__all__ = [
    "Flatness",
    "Smallness",
    "TikhonovZero",
]
