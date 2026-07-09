"""
Custom types used for type hints.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.sparse import bsr_array, coo_array, csc_array, csr_array, dia_array
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from .base import MinimizerResult

SparseArray: TypeAlias = bsr_array | coo_array | csc_array | csr_array | dia_array
"""
Type alias to represent sparse arrays.
"""

Model: TypeAlias = npt.NDArray[np.float64]
"""
Type alias to represent models in the inversion framework as 1D arrays.
"""


Preconditioner: TypeAlias = npt.NDArray[np.float64] | SparseArray | LinearOperator
"""
Type for preconditioners.

Preconditioners can either be a dense matrix, a sparse matrix or a ``LinearOperator``.
"""


@runtime_checkable
class CanBeUpdated(Protocol):
    """
    Protocol for objects that can be updated.
    """

    def update(self, model: Model) -> None:
        raise NotImplementedError


class SparseRegularization(Protocol):
    """
    Protocol to define sparse regularizations that can be used with a IRLS algorithm.
    """

    irls: bool

    def update_irls(self, model: Model) -> None:
        raise NotImplementedError

    def activate_irls(self, model_previous: Model) -> None:
        raise NotImplementedError


class Log(Protocol):
    """
    Protocol to define inversion and minimizer logs.
    """

    def update(self, iteration: int, model: Model) -> None:
        raise NotImplementedError

    def get_minimizer_callback(self) -> Callable[["MinimizerResult"], None]:
        raise NotImplementedError


@runtime_checkable
class HasDiagonal(Protocol):
    """
    Protocol to define abstract array-like objects that has a ``diagonal`` method.
    """

    def diagonal(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError
