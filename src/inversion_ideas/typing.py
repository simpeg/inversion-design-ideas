"""
Custom types used for type hints.
"""

from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator

from .base.model import MultiModel

Model: TypeAlias = npt.NDArray[np.float64] | MultiModel
"""
Type alias to represent models in the inversion framework as a 1D array or a MultiModel.
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
