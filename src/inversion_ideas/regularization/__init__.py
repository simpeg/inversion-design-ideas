"""
Regularization classes.
"""

from ._general import SimpleSmallness
from ._mesh_based import Flatness, Smallness, SparseSmallness

__all__ = [
    "Flatness",
    "SimpleSmallness",
    "Smallness",
    "SparseSmallness",
]
