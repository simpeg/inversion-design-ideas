"""
Custom :class:`scipy.sparse.linalg.LinearOperator` classes.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator


class Identity(LinearOperator):
    """
    Identity matrix.

    Represent the identity matrix of size ``n`` through
    a :class:`scipy.sparse.linalg.LinearOperator`, to avoid storing a large diagonal
    array full of ones.

    Parameters
    ----------
    n : int
        Size of the square matrix.
    dtype : dtype, optional
        Data type used for the linear operator.
    """

    def __init__(self, n: int, dtype=np.float64):
        self._n = n
        super().__init__(shape=(n, n), dtype=dtype)

    @property
    def n(self):
        """
        Size of the identity matrix.
        """
        return self._n

    def _matvec(self, x):
        # TODO: check if we should be copying here
        return x

    def _rmatvec(self, x):
        # TODO: check if we should be copying here
        return x

    def diagonal(self):
        """
        Diagonal of the matrix.
        """
        return np.ones(self.n, dtype=self.dtype)
