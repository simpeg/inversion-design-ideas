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


class ExpandedSquareMatrix(LinearOperator):
    r"""
    Represents a square matrix expanded with zeros.

    Use this class to represent hessian matrices that are built from smaller matrices
    that operate only on a subset of the entire model. Only a portion of that hessian
    will be non-zero (the one related to the relevant model elements for that objective
    function), while the rest of the matrix will be filled of zeros.

    Parameters
    ----------
    matrix : array, sparse array or LinearOperator
        Square matrix to be expanded
    shape : tuple of int
        Shape of the expanded matrix.
    slice_ : slice or list of slices
        Slice(s) used to slice portions of the square matrix. The start and stop should
        be positive or zero, start should be lower than stop, and step should be None.

    Notes
    -----
    Given a square matrix :math:`\mathbf{A}`, this
    ``LinearOperator`` represents an *expanded* version of :math:`\mathbf{A}` like:

    .. math::

        \mathbf{H} = \begin{bmatrix}
            0 & 0          & 0\\
            0 & \mathbf{A} & 0\\
            0 & 0          & 0
        \end{bmatrix},

    where :math:`\mathbf{A}` sits in the diagonal of :math:`\mathbf{H}`.
    The ``slice_`` argument is used to specify the location of the matrix
    :math:`\mathbf{A}` within the expanded :math:`\mathbf{H}` matrix.

    When passing multiple ``slice_``s, it can represent more complex expanded
    matrix like:

    .. math::

        \mathbf{H} = \begin{bmatrix}
            0 & 0                & 0 & 0                 & 0 & 0      & 0 & 0                   & 0 & 0                  & 0 \\
            0 & \mathbf{A}_{1,1} & 0 & \mathbf{A}_{2,1}  & 0 & \dots  & 0 & \mathbf{A}_{n-1,1}  & 0 & \mathbf{A}_{n,1}   & 0 \\
            0 & 0                & 0 & 0                 & 0 & 0      & 0 & 0                   & 0 & 0                  & 0 \\
            0 & \vdots           & 0 & \vdots            & 0 & \ddots & 0 & \vdots              & 0 & \vdots             & 0 \\
            0 & 0                & 0 & 0                 & 0 & 0      & 0 & 0                   & 0 & 0                  & 0 \\
            0 & \mathbf{A}_{1,n} & 0 & \mathbf{A}_{2,n}  & 0 & \dots  & 0 & \mathbf{A}_{n-1,n}  & 0 & \mathbf{A}_{n,n}   & 0 \\
            0 & 0                & 0 & 0                 & 0 & 0      & 0 & 0                   & 0 & 0                  & 0
        \end{bmatrix},

    where each :math:`\mathbf{A}_{i,j}` is a square submatrix of :math:`\mathbf{A}`.
    """

    def __init__(
        self,
        matrix: npt.NDArray | LinearOperator | SparseArray,
        shape: tuple[int],
        slice_: slice | list[slice],
    ):
        if len(shape) != 2:
            # TODO: add msg
            raise ValueError()

        if shape[0] != shape[1]:
            # TODO: add msg
            raise ValueError()

        if matrix.ndim != 2:
            # TODO: add msg
            raise ValueError()

        if matrix.shape[0] != matrix.shape[1]:
            # TODO: add msg
            raise ValueError()

        # TODO: add sanity checks for slices
        #   - start >= 0 and stop > 0
        #   - step is None
        #   - start < stop

        super().__init__(shape=shape, dtype=matrix.dtype)
        self._matrix = matrix
        self._slice = slice_

        if self.reduced_size != matrix.shape[0]:
            # TODO: add msg
            raise ValueError()

    @property
    def matrix(self):
        return self._matrix

    @property
    def slice_(self):
        return self._slice

    @property
    def slices(self):
        if isinstance(self.slice_, slice):
            return [self.slice_]
        return self._slice

    @property
    def reduced_size(self):
        # TODO: choose better name
        return sum(s.stop - s.start for s in self.slices)

    @property
    def slicer_matrix(self):
        if not hasattr(self, "_slicer_matrix"):
            shape = (self.reduced_size, self.shape[0])
            slicer_matrix = Zero(shape=shape, dtype=self.dtype)
            offset = 0
            for slice_ in self.slices:
                slicer_matrix += _Slicer(
                    start=slice_.start, stop=slice_.stop, shape=shape, offset=offset
                )
                offset += slice_.stop - slice_.start
            if isinstance(slicer_matrix, Zero):
                msg = "No slices were found!"
                raise ValueError(msg)
            self._slicer_matrix = slicer_matrix
        return self._slicer_matrix

    def _matvec(self, x):
        """
        Dot product between the matrix and a vector.
        """
        return self.slicer_matrix.T @ (self.matrix @ (self.slicer_matrix @ x))

    def _rmatvec(self, x):
        """
        Dot product between the transposed matrix and a vector.
        """
        return self.slicer_matrix @ (self.matrix.T @ (self.slicer_matrix.T @ x))

    def _matmat(self, x):
        """
        Dot product between the matrix and a vector.
        """
        raise NotImplementedError()

    def diagonal(self):
        """
        Diagonal of the matrix.
        """
        raise NotImplementedError()

    def toarray(self):
        """
        Return expanded matrix as a dense array.

        .. warning::

            This method could create a very large 2D array that might not fit in memory.
        """
        raise NotImplementedError()


class _Slicer(LinearOperator):
    """ """

    def __init__(
        self, start: int, stop: int, shape: tuple[int, int], offset: int | None = None
    ):
        if shape[0] < start - stop:
            # TODO: add msg
            # The size of the slice cannot be larger than the output array.
            raise ValueError()

        if shape[0] > shape[1]:
            # TODO: add msg.
            # The sliced array cannot be larger than the array that will be sliced.
            raise ValueError()

        # TODO: add sanity checks for offset

        super().__init__(shape=shape, dtype=np.float64)
        self._start = start
        self._stop = stop
        self._offset = offset

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def slice(self):
        return slice(self._start, self._stop)

    @property
    def offset(self) -> int:
        if self._offset is None:
            return 0
        return self._offset

    def _matvec(self, x):
        """
        Slices the array.
        """
        result = np.zeros(self.shape[0], dtype=x.dtype)
        out_slice = slice(self.offset, self.offset + self.slice.stop - self.slice.start)
        result[out_slice] = x[self.slice]
        return result

    def _rmatvec(self, x):
        """
        Expand the array.
        """
        result = np.zeros(self.shape[1], dtype=x.dtype)
        out_slice = slice(self.offset, self.offset + self.slice.stop - self.slice.start)
        result[self.slice] = x[out_slice]
        return result

    def _matmat(self, x):
        """
        Dot product between the matrix and a vector.
        """
        pass
        # raise NotImplementedError()

    def toarray(self):
        # Idea: represent this as a sparse diagonal array and use the toarray method to
        # return a dense representation.
        raise NotImplementedError()


class Zero(LinearOperator):
    """
    Null linear operator.
    """

    def _matvec(self, x):
        return np.zeros(shape=self.shape[0], dtype=self.dtype)

    def _adjoint(self):
        new_shape = (self.shape[1], self.shape[0])
        return Zero(shape=new_shape, dtype=self.dtype)

    def _matmat(self, x):
        return np.zeros(shape=(self.shape[0], x.shape[1]), dtype=self.dtype)
