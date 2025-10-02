"""
Regularization classes.
"""

from ._general import TikhonovZero
from ._mesh_based import Smallness, Smoothness

__all__ = [
    "Smallness",
    "Smoothness",
    "TikhonovZero",
]
