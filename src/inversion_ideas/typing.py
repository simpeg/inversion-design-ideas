"""
Custom types used for type hints.
"""

from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator

Model: TypeAlias = npt.NDArray[np.float64]
"""
Type alias to represent models in the inversion framework as 1D arrays.
"""

Preconditioner: TypeAlias = npt.NDArray[np.float64] | sparray | LinearOperator
"""
Type for static preconditioners.

Static preconditioners can either be a dense matrix, a sparse matrix or
a ``LinearOperator``.
"""


class SparseRegularization(Protocol):
    """
    Protocol to define sparse regularizations that can be used with a IRLS algorithm.
    """

    irls: bool

    def update_irls(self, model: Model) -> None:
        raise NotImplementedError

    def activate_irls(self, model_previous: Model) -> None:
        raise NotImplementedError


@runtime_checkable
class HasDiagonal(Protocol):
    """
    Protocol to define an object that implements the ``diagonal`` method.
    """

    def diagonal(self) -> npt.NDArray: ...
