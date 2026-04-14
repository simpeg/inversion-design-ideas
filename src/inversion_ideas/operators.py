"""
Custom LinearOperator classes and utilities.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import dia_array
from scipy.sparse.linalg import LinearOperator

from .typing import HasDiagonal, SparseArray


def get_diagonal(operator: npt.NDArray | SparseArray | LinearOperator):
    r"""
    Extract diagonal of a linear operator.

    Extracts the main diagonal of a square linear operator. If the operator is a dense
    or sparse array with a ``diagonal`` method, the method will be used. For
    :class:`~scipy.sparse.linalg.LinearOperator`s or any other operator that doesn't
    implement the ``diagonal`` method, the diagonal will be computed by ``N``
    projections using the unit vectors of the standard basis of :math:`\mathcal{R}^N`.

    .. important::

        Extracting the diagonal of a :class:`~scipy.sparse.linalg.LinearOperator`
        requires computing multiple dot products. This can be quite expensive for large
        operators.

    Parameters
    ----------
    operator : (n, n) array, sparse array or LinearOperator
        Square linear operator from which the diagonal will be extracted.

    Returns
    -------
    diagonal : (n,) array
        1D array containing the diagonal of the linear operator.

    Notes
    -----
    Consider a :math:`N \times N` linear operator :math:`A`, and let
    :math:`\{ e_i \}_1^N` be the standard basis for :math:`\mathcal{R}^N`.
    The :math:`i`-th diagonal element of :math:`A` (:math:`a_{ii}`) can be obtained as:

    .. math::

        a_{ii} = e_i^T A e_i


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import aslinearoperator

    Build a linear operator from a matrix, and extract its diagonal:

    >>> a_matrix = np.array(
    ...     [
    ...      [1., 2., 3.],
    ...      [4., 5., 6.],
    ...      [7., 8., 9.],
    ...     ]
    ... )
    >>> linop = aslinearoperator(a_matrix)
    >>> get_diagonal(linop)
    array([1., 5., 9.])
    """
    shape = operator.shape
    if shape[0] != shape[1]:
        msg = (
            f"Invalid operator '{operator}' with shape '{shape}'. "
            "It must be a square linear operator."
        )
        raise ValueError(msg)

    if isinstance(operator, HasDiagonal):
        return operator.diagonal()

    n, _ = shape
    basis = _get_standard_basis(n)
    diagonal = np.fromiter((e.T @ operator @ e for e in basis), dtype=operator.dtype)
    return diagonal


def _get_standard_basis(ndim: int, dtype=np.float64):
    r"""
    Generate the unit vectors of the standard basis of :math:`\mathcal{R}^N`.

    Parameters
    ----------
    ndim: int
        Number of dimensions of the vector space.
    dtype : dtype, optional
        Data type of the yielded arrays.

    Yields
    ------
    (n,) array
        Array representing the i-th unit vector.
    """
    for i in range(ndim):
        vector = np.zeros(ndim, dtype=dtype)
        vector[i] = 1
        yield vector


class BlockSquareMatrix(LinearOperator):
    r"""
    Operator that represents a square matrix with a non-zero block surrounded by zeros.

    Use this class to represent hessian matrices that are built from smaller matrices
    that operate only on a subset of the entire model. Only a block of that hessian will
    be non-zero (the one related to the relevant model elements for that objective
    function), while the rest of the matrix will be filled of zeros.

    Parameters
    ----------
    block : array, sparse array or LinearOperator
        Square block matrix.
    slice_matrix : dia_array
        Diagonal array to represent the slicing matrix.

    Notes
    -----
    This ``LinearOperator`` represents square matrices like:

    .. math::

        \mathbf{H} = \begin{bmatrix}
            0 & 0          & 0\\
            0 & \mathbf{B} & 0\\
            0 & 0          & 0
            \end{bmatrix}

    where :math:`\mathbf{B}` is a non-zero square block matrix that sits in the diagonal
    of :math:`\mathbf{H}`. The matrix :math:`\mathbf{H}` can be built as:

    .. math::

        \mathbf{H} = \mathbf{s}^T \mathbf{B} \mathbf{s}

    where :math:`\mathbf{s}` is the *slicing matrix*.
    """

    def __init__(
        self,
        block: npt.NDArray | LinearOperator | SparseArray,
        slice_matrix: dia_array,
    ):
        if block.shape[0] != block.shape[1]:
            msg = (
                f"Invalid block matrix '{block}' with shape '{block.shape}'. "
                "It must be a square matrix."
            )
            raise ValueError(msg)

        if slice_matrix.shape[0] != block.shape[1]:
            msg = (
                f"Invalid block '{block}' and slice_matrix '{slice_matrix}' with "
                f"shapes '{block.shape}' and {slice_matrix.shape}."
            )
            raise ValueError(msg)

        if slice_matrix.shape[1] <= block.shape[1]:
            # TODO: improve msg
            msg = "block is larger than slice matrix"
            raise ValueError(msg)

        _, full_size = slice_matrix.shape
        shape = (full_size, full_size)
        super().__init__(shape=shape, dtype=block.dtype)

        self._block = block
        self._slice_matrix = slice_matrix

    @property
    def block(self):
        return self._block

    @property
    def slice_matrix(self) -> dia_array:
        return self._slice_matrix

    def _matvec(self, x):
        """
        Dot product between the matrix and a vector.
        """
        result = self.slice_matrix.T @ (self.block @ (self.slice_matrix @ x))
        return result

    def _rmatvec(self, x):
        """
        Dot product between the transposed matrix and a vector.
        """
        result = self.slice_matrix @ (self.block.T @ (self.slice_matrix.T @ x))
        return result

    def diagonal(self):
        """
        Diagonal of the matrix.
        """
        if not isinstance(self.block, HasDiagonal):
            # TODO: Add msg
            raise TypeError()
        diagonal = self.block.diagonal()
        return self.slice_matrix.T @ diagonal
