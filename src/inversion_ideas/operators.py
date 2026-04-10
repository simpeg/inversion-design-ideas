"""
Custom LinearOperator classes and utilities.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator

from inversion_ideas.typing import HasDiagonal, SparseArray


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
