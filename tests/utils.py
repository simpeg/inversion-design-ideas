"""
Test utilities.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import dia_array, sparray
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from inversion_ideas.base import Objective
from inversion_ideas.typing import Model, SparseArray


class Dummy(Objective):
    r"""
    Dummy objective function.

    Define a dummy objective function as:

    .. math::

        \phi(\mathbf{m}) = \mathbf{m}^T \mathbf{A}^T \mathbf{A} \mathbf{m},

    where :math:`\mathbf{A}` is a random ``(n, \n_params)`` matrix.

    It's gradient will therefore be:

    .. math::

        \nabla\phi(\mathbf{m}) = \mathbf{A}^T \mathbf{A} \mathbf{m},

    and its Hessian:

    .. math::

        \bar{\bar{\nabla}}\phi(\mathbf{m}) = \mathbf{A}^T \mathbf{A}.

    Parameters
    ----------
    n_params : int
        Number of parameters for the objective function.
    seed : int or numpy.random.Generator or numpy.random.RandomState or None, optional
        Random seed used to define the :math:`\mathbf{A}` matrix.
    hessian_type : {"dense", "sparse", "linop"}, optional
        Type of Hessian matrix: "dense" matrix, "sparse" matrix or "linop" as in
        a ``LinearOperator``.
    """

    def __init__(self, n_params, seed=None, hessian_type="dense"):
        self._n_params = n_params
        rng = np.random.default_rng(seed=seed)
        self.a_matrix = rng.uniform(size=(n_params, n_params))
        if hessian_type not in ("dense", "sparse", "linop"):
            msg = f"Invalid hessian_type '{hessian_type}'."
            raise ValueError(msg)
        self.hessian_type = hessian_type

    @property
    def n_params(self):
        return self._n_params

    def __call__(self, model):
        return float(model.T @ self.a_matrix.T @ self.a_matrix @ model)

    def gradient(self, model):
        return self.a_matrix.T @ self.a_matrix @ model

    def hessian(self, model):  # noqa: ARG002
        match self.hessian_type:
            case "dense":
                hessian = self.a_matrix.T @ self.a_matrix
            case "sparse":
                a_sparse = dia_array(self.a_matrix)
                hessian = a_sparse.T @ a_sparse
            case "linop":
                a_linop = aslinearoperator(self.a_matrix)
                hessian = a_linop.T @ a_linop
            case _:
                msg = f"Invalid hessian_type '{self.hessian_type}'."
                raise ValueError(msg)
        return hessian


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


def derivative_test(
    function: Callable[[Model], float | NDArray[np.float64]],
    derivative: Callable[[Model], NDArray[np.float64] | SparseArray | LinearOperator],
    model: Model,
    delta_m: Model,
    **kwargs,
):
    r"""
    Check implementation of the derivative of a function.

    Compare the value of a given function with an approximation of it using a first
    order Taylor series expansion.

    Parameters
    ----------
    function : callable
        Function that will be tested. It must take a ``model`` array as argument, and
        return either a float or an array.
    derivative : callable
        Derivative of the ``function``. It must take a ``model`` array as argument,
        and return a dense or sparse array, or a
        :class:`~scipy.sparse.linalg.LinearOperator`.
    model : (n_params) array
        Array with model values that will be used to perform the test.
    delta_m : (n_params) array
        Array of small perturbation in the model space that will be used in the Taylor
        series approximation.
    **kwargs :
        Extra arguments passed to :func:`numpy.testing.assert_allclose` when comparing
        the value of the function and its first order approximation.

    Raises
    ------
    AssertionError :
        If the function and the approximation are not close enough for the given model
        and perturbation vectors.

    Notes
    -----
    This function will test the implementation of the derivative of a given function
    by evaluating the function and comparing it with an approximation using a first
    order Taylor series expansion.
    Since the derivative plays a part in that approximation, the comparison allows us to
    check if the derivative is correctly implemented for that particular function.

    Consider a `scalar field <https://en.wikipedia.org/wiki/Scalar_field>`__
    :math:`f: \mathbb{R}^M \rightarrow \mathbb{R}`, a model :math:`\mathbf{m}` and a
    perturbation vector :math:`\Delta\mathbf{m}` with a magnitude significantly smaller
    than the one of :math:`\mathbf{m}`.
    We can approximate :math:`f(\mathbf{m} + \Delta\mathbf{m})` using a first-order
    Taylor series expansion:

    .. math::

        f(\mathbf{m} + \Delta\mathbf{m}) =
        f(\mathbf{m}) + \nabla f(\mathbf{m}) \cdot \Delta\mathbf{m} + \Delta_r,

    where :math:`\nabla f(\mathbf{m})` is the gradient of :math:`f` evaluated on the
    same model :math:`\mathbf{m}`, and :math:`\Delta_r` is the discrepancy between the
    function and the first order approximation.

    Consider now a `vector field <https://en.wikipedia.org/wiki/Vector_field>`__
    :math:`\mathbf{f}: \mathbb{R}^M \rightarrow \mathbb{R}^N`.
    Analogously, we can approximate :math:`\mathbf{f}(\mathbf{m} + \Delta\mathbf{m})`
    using a first-order Taylor series expansion:

    .. math::

        \mathbf{f}(\mathbf{m} + \Delta\mathbf{m}) =
        \mathbf{f}(\mathbf{m})
        + \mathbf{J}_\mathbf{f}(\mathbf{m}) \cdot \Delta\mathbf{m} + \Delta_r,

    where :math:`\mathbf{J}_\mathbf{f}(\mathbf{m})` is the Jacobian matrix of the
    vector field :math:`\mathbf{f}` evaluated in the model :math:`\mathbf{m}`,
    and :math:`\Delta_r` is also the discrepancy between the function and its
    approximation.

    If the perturbation :math:`\Delta\mathbf{m}` is sufficiently small, the discrepancy
    :math:`\Delta_r` will also be small in both cases.

    """
    # Approximate the function using first order Taylor series
    approximation = function(model) + derivative(model) @ delta_m

    # Evaluate the function
    expected = function(model + delta_m)

    # Perform derivative test
    try:
        np.testing.assert_allclose(approximation, expected, **kwargs)
    except AssertionError as e:
        msg = (
            f"Failed derivative test for function '{function}' and "
            f"derivative '{derivative}' with:"
            f"\nmodel:    {model}"
            f"\ndelta_m:  {delta_m}"
            "\n"
        )
        raise AssertionError(msg + str(e)) from None
