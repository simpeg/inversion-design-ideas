"""
Regularization classes.
"""

from ._general import SimpleSmallness, TikhonovFirst
from ._mesh_based import Flatness, Smallness, SparseSmallness

__all__ = [
    "Flatness",
    "SimpleSmallness",
    "Smallness",
    "SparseSmallness",
    "TikhonovFirst",
]
