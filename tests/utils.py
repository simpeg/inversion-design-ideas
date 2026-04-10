"""
Test utilities.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator

from inversion_ideas.base import Simulation
from inversion_ideas.typing import SparseArray
from inversion_ideas.utils import cache_on_model


def assert_equal_linear_operators(
    a: NDArray | SparseArray | LinearOperator,
    b: NDArray | SparseArray | LinearOperator,
    to_dense=False,
    seed=None,
    **kwargs,
):
    """
    Check if two linear operators are the same.

    If ``a`` and ``b`` are ``LinearOperator``s, they will be compared by computing the
    dot product with random arrays. Only the ``matvec`` and ``rmatvec`` will be tested.

    Parameters
    ----------
    a, b : array, sparse array, or LinearOperator
        Arrays or linear operators that will be tested.
    to_dense : bool, optional
        If True, sparse arrays will be converted to dense arrays for testing.
        Use False for big matrices that can be too large to fit in memory.
    seed : int or None, optional
        Random seed used to define a random vector to test ``LinearOperator``s.
        This argument will be ignored if ``a`` and ``b`` are not ``LinearOperator``s.
    **kwargs : dict
        Extra keyword arguments that will be passed to
        :func:`numpy.testing.assert_equal`.
    """
    if to_dense:
        if isinstance(a, sparray):
            a = a.toarray()
        if isinstance(b, sparray):
            b = b.toarray()
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        np.testing.assert_equal(a, b, **kwargs)
    else:
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        # matvec
        rng = np.random.default_rng(seed=seed)
        vector = rng.uniform(size=a.shape[1])
        np.testing.assert_equal(a @ vector, b @ vector, **kwargs)
        # rmatvec
        vector = rng.uniform(size=a.shape[0])
        np.testing.assert_equal(a.T @ vector, b.T @ vector, **kwargs)


def assert_allclose_linear_operators(
    a: NDArray | SparseArray | LinearOperator,
    b: NDArray | SparseArray | LinearOperator,
    to_dense=False,
    seed=None,
    **kwargs,
):
    """
    Check if two linear operators are close enough.

    If ``a`` and ``b`` are ``LinearOperator``s, they will be compared by computing the
    dot product with random arrays. Only the ``matvec`` and ``rmatvec`` will be tested.

    Parameters
    ----------
    a, b : array, sparse array, or LinearOperator
        Arrays or linear operators that will be tested.
    to_dense : bool, optional
        If True, sparse arrays will be converted to dense arrays for testing.
        Use False for big matrices that can be too large to fit in memory.
    seed : int or None, optional
        Random seed used to define a random vector to test ``LinearOperator``s.
        This argument will be ignored if ``a`` and ``b`` are not ``LinearOperator``s.
    **kwargs : dict
        Extra keyword arguments that will be passed to
        :func:`numpy.testing.assert_allclose`.
    """
    if to_dense:
        if isinstance(a, sparray):
            a = a.toarray()
        if isinstance(b, sparray):
            b = b.toarray()
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        np.testing.assert_allclose(a, b, **kwargs)
    else:
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        # matvec
        rng = np.random.default_rng(seed=seed)
        vector = rng.uniform(size=a.shape[1])
        np.testing.assert_allclose(a @ vector, b @ vector, **kwargs)
        # rmatvec
        vector = rng.uniform(size=a.shape[0])
        np.testing.assert_allclose(a.T @ vector, b.T @ vector, **kwargs)


class LinearRegressor(Simulation):
    r"""
    Linear regressor.

    .. math::

        \mathbf{y} = \mathbf{X} \cdot \mathbf{m}
    """

    def __init__(self, X, linop=False, cache=True):
        self.X = X
        self.linop = linop
        self.cache = cache

    @property
    def n_params(self) -> int:
        return self.X.shape[1]

    @property
    def n_data(self) -> int:
        return self.X.shape[0]

    @cache_on_model
    def __call__(self, model) -> NDArray[np.float64]:
        return self.X @ model

    def jacobian(self, model) -> NDArray[np.float64] | LinearOperator:  # noqa: ARG002
        if self.linop:
            linear_op = LinearOperator(
                shape=(self.n_data, self.n_params),
                matvec=lambda model: self.X @ model,
                rmatvec=lambda model: self.X.T @ model,
                dtype=np.float64,
            )
            return linear_op
        return self.X
