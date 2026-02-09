"""
Custom types used for type hints.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from .base import MinimizerResult

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


class Log(Protocol):
    """
    Protocol to define inversion and minimizer logs.
    """

    def update(self, iteration: int, model: Model) -> None:
        raise NotImplementedError

    def get_minimizer_callback(self) -> Callable[["MinimizerResult"], None]:
        raise NotImplementedError
