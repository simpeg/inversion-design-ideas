"""
Custom types used for type hints.
"""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator

Model: TypeAlias = npt.NDArray[np.float64]
"""
Type alias to represent models in the inversion framework as 1D arrays.
"""

Preconditioner: TypeAlias = (
    npt.NDArray[np.float64]
    | sparray
    | LinearOperator
    | Callable[[Model], npt.NDArray[np.float64] | sparray | LinearOperator]
)
"""
Type for possible preconditioners.

A preconditioner can either be _static_
(a dense or sparse matrix, or a ``LinearOperator``),
or _dynamic_ as a callable that takes a model and returns a static preconditioner
(a dense or sparse matrix or a ``LinearOperator``).
"""
