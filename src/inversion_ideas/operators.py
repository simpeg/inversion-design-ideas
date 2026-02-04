"""
Special linear operators.

Define classes for custom linear operators.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator


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

        self._block = block
        self._index_start = index_start
        self._slice = slice(index_start, index_start + block.shape[1])

    @property
    def block(self):
        return self._block

    @property
    def index_start(self):
        return self._index_start

    @property
    def slice(self):
        return self._slice

    def _matvec(self, x):
        """
        Dot product between the matrix and a vector.
        """
        x_subset = x[self.slice]
        return self.block @ x_subset

    def _rmatvec(self, x):
        """
        Dot product between the transposed matrix and a vector.
        """
        out = np.zeros(self.shape[1], dtype=self.dtype)
        out[self.slice] = self.block.T @ x
        return out

    def toarray(self):
        """
        Return a dense ndarray representation of this blocked matrix.
        """
        # TODO: raise error if the block is not an array or if it doesn't have a toarray
        # method.
        matrix = np.zeros(self.shape, dtype=self.dtype)
        matrix[:, self.slice] = self.block
        return matrix

    def get_column(self, column_index) -> npt.NDArray:
        """
        Get the j-th column of the matrix.

        Parameters
        ----------
        column_index : int
            Index for the desired column.

        Returns
        -------
        column : array
            The j-th column of the matrix.
        """
        # TODO: raise error if we cannot extract column from block
        # (sparse array, linear operator).
        axis = 1
        if not (0 <= column_index < self.shape[axis]):
            msg = (
                f"index {column_index} is out of bounds for axis {axis} "
                f"with size {self.shape[axis]}"
            )
            raise IndexError(msg)
        if self.slice.start <= column_index < self.slice.stop:
            return self.block[:, column_index - self.slice.start]
        return np.zeros(self.shape[0], dtype=self.dtype)

    def __getitem__(self, indices):
        # TODO: raise error if block cannot be indexed
        # TODO: raise error if slices? or support them.

        row, column = indices

        # Sanity checks for indices
        axis = 0
        if row >= 0:
            if not (0 <= row < self.shape[axis]):
                msg = (
                    f"index {row} is out of bounds for axis {axis} "
                    f"with size {self.shape[axis]}"
                )
                raise IndexError(msg)
        else:
            if row < -self.shape[0]:
                msg = (
                    f"index {row} is out of bounds for axis {axis} "
                    f"with size {self.shape[axis]}"
                )
                raise IndexError(msg)
            row += self.shape[0]

        axis = 1
        if column >= 0:
            if not (0 <= column < self.shape[axis]):
                msg = (
                    f"index {column} is out of bounds for axis {axis} "
                    f"with size {self.shape[axis]}"
                )
                raise IndexError(msg)
        else:
            if column < -self.shape[axis]:
                msg = (
                    f"index {column} is out of bounds for axis {axis} "
                    f"with size {self.shape[axis]}"
                )
                raise IndexError(msg)
            column += self.shape[axis]

        if self.slice.start <= column < self.slice.stop:
            return self.block[row, column - self.slice.start]
        return np.astype(np.float64(0.0), self.dtype)
