"""
Regularization classes.
"""

from ._general import SimpleFlatness, SimpleSmallness
from ._mesh_based import Flatness, Smallness, SparseSmallness

__all__ = [
    "Flatness",
    "SimpleFlatness",
    "SimpleSmallness",
    "Smallness",
    "SparseSmallness",
]
