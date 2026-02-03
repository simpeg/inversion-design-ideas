"""
Special linear operators.

Define classes for custom linear operators.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import sparray


class BlockColumnMatrix(LinearOperator):
    r"""
    Represents a block column matrix.

    This ``LinearOperator`` represents a matrix with a dense or sparse block that
    occupies all rows, and columns to each side of it are all zeros.

    .. math::

        \textbf{M} =
        \begin{bmatrix}
            \dots & 0 & a_{11} & \dots  & a_{1M}  & 0 & \dots\\
            \dots & 0 & \vdots & \ddots & \vdots  & 0 & \dots\\
            \dots & 0 & a_{N1} & \dots  & a_{NM}  & 0 & \dots\\
        \end{bmatrix}

    Parameters
    ----------
    block : 2D array, LinearOperator or sparray
        Matrix or linear operator that will be used
    index_start : int
        Row index where the first column of the ``block`` matrix should be located in
        the large block matrix.
    n_cols : int
        Total number of columns of the large block matrix.
    """

    def __init__(
        self,
        block: npt.NDArray | LinearOperator | sparray,
        index_start: int,
        n_cols: tuple[int, int],
    ):
        # TODO: raise error if the block matrix has more columns than n_cols
        shape = (block.shape[0], n_cols)
        super().__init__(shape=shape, dtype=block.dtype)

        self.block = block
        self._slice = slice(index_start, index_start + block.shape[1])

    def _matvec(self, x):
        """
        Dot product between the matrix and a vector.
        """
        x_subset = x[self._slice]
        return self.block @ x_subset

    def _rmatvec(self, x):
        """
        Dot product between the transposed matrix and a vector.
        """
        out = np.zeros(self.shape[1], dtype=self.dtype)
        out[self._slice] = self.block.T @ x
        return out

    def toarray(self):
        """
        Return a dense ndarray representation of this blocked matrix.
        """
        # TODO: raise error if the block is not an array or if it doesn't have a toarray
        # method.
        matrix = np.zeros(self.shape, dtype=self.dtype)
        matrix[:, self._slice] = self.block
        return matrix
