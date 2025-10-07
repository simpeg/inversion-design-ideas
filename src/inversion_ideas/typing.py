"""
Custom types used for type hints.
"""

from typing import TypeAlias

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
