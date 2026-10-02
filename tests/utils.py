"""
Test utilities.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import dia_array, sparray
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from inversion_ideas.base import Objective, Simulation
from inversion_ideas.decorators import cache_on_model
from inversion_ideas.typing import SparseArray
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


class NonLinearRegressor(Simulation):
    r"""
    Non-linear regressor simulation.

    Parameters
    ----------
    a_matrix : (n_data, n_params) array
        The :math:`\mathbf{A}` matrix.
    b_matrix : (n_data, n_params) array
        The :math:`\mathbf{A}` matrix.
    build_jacobian : bool, optional
        Whether the Jacobian matrix will be created as a dense matrix (True) or as a
        :class:`~scipy.sparse.linalg.LinearOperator` (False). Default to True.
    cache : bool, optional
        Whether to cache the results of the ``__call__`` method for the last model
        vector or not. Default to True.

    Notes
    -----
    Implements a simple non-linear simulation as a non-linear regressor in the form:

    .. math::

        \mathbf{y} = \mathbf{A} \cdot \mathbf{m}^2 + \mathbf{B} \cdot \mathbf{m}

    where :math:`\mathbf{y}` is the predicted data, :math:`\mathbf{m}` is the model
    vector, and :math:`\mathbf{A}` and :math:`\mathbf{B}` are two
    (``n_data``, ``n_params``) matrices.
    """

    def __init__(self, a_matrix, b_matrix, *, build_jacobian=True, cache=True):
        if a_matrix.shape != b_matrix.shape:
            raise ValueError()
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.build_jacobian = build_jacobian
        self.cache = cache

    @classmethod
    def create_random(cls, n_data: int, n_params: int, *, seed=None, **kwargs):
        """Create a non-linear regressor with random matrices.

        Parameters
        ----------
        n_data : int
            Number of data values that the simulation will generate.
        n_params : int
            Number of elements in the model vector.
        seed : int or None, optional
            Random seed or random state used to generate the matrix.
        **kwargs
            Keyword arguents passed to
            the constructor of :class:`~inversion_ideas.LinearRegressor`.
        """
        shape = (n_data, n_params)
        rng = np.random.default_rng(seed=seed)
        a_matrix = rng.uniform(low=-1.0, high=1.0, size=shape)
        b_matrix = rng.uniform(low=-1.0, high=1.0, size=shape)
        return cls(a_matrix, b_matrix, **kwargs)

    @property
    def n_params(self) -> int:
        return self.a_matrix.shape[1]

    @property
    def n_data(self) -> int:
        return self.a_matrix.shape[0]

    @cache_on_model
    def __call__(self, model) -> NDArray[np.float64]:
        return self.a_matrix @ model**2 + self.b_matrix @ model

    def jacobian(self, model) -> NDArray[np.float64] | LinearOperator:
        jacobian = (
            2 * self.a_matrix * model  # element-wise operation
            + self.b_matrix
        )
        if not self.build_jacobian:
            return aslinearoperator(jacobian)
        return jacobian
      
      
def derivative_test(
    function: Callable[[Model], float | NDArray[np.float64]],
    derivative: Callable[[Model], NDArray[np.float64] | SparseArray | LinearOperator],
    model: Model,
    delta_m: Model,
    **kwargs,
):
    r"""
    Validate the implementation of the derivative of a function.

    Compare the value of a given function with an approximation of it using a first
    order Taylor series expansion.
    If the discrepancy is not small enough, an :exc:`AssertionError` will be raised.

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


def derivative_convergence_test(
    function: Callable[[Model], float | NDArray[np.float64]],
    derivative: Callable[[Model], NDArray[np.float64] | SparseArray | LinearOperator],
    model: Model,
    delta_m: Model,
    factors: list[float] | NDArray[np.float64] | None = None,
):
    r"""
    Validate the implementation of the derivative of a function through convergence.

    Tests the implementation of the derivative of the function by  performing a
    convergence test of a first-order Taylor series expansion of the function.
    If the convergence test fails, an :exc:`AssertionError` will be raised.

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
        series approximation. This vector will be scaled using the ``factor`` during the
        convergence test.
    factors : list of float, array of float or None
        List of factors that will be used to scale the ``delta_m`` vector.
        If None, a default set of seven factors will be used as a logspace spanning from
        ``1e-6`` to ``1.0``.

    Raises
    ------
    AssertionError :
        If the convergence test fails.

    Notes
    -----
    This function will test the implementation of the derivative of a given function
    by performing a convergence test of a first-order Taylor series expansion of such
    function.

    Consider a `scalar field <https://en.wikipedia.org/wiki/Scalar_field>`__
    :math:`f: \mathbb{R}^M \rightarrow \mathbb{R}`, a model :math:`\mathbf{m}` a
    perturbation vector :math:`\Delta\mathbf{m}` with a magnitude significantly smaller
    than the one of :math:`\mathbf{m}`,
    and a scaling factor :math:`\alpha \in \mathbb{R}`.
    We can approximate :math:`f(\mathbf{m} + \alpha\Delta\mathbf{m})` using a
    first-order Taylor series expansion:

    .. math::

        f(\mathbf{m} + \alpha \Delta\mathbf{m}) =
        f(\mathbf{m}) + \nabla f(\mathbf{m}) \cdot \alpha \Delta\mathbf{m} + \Delta_r,

    where :math:`\nabla f(\mathbf{m})` is the gradient of :math:`f` evaluated on the
    same model :math:`\mathbf{m}`, and :math:`\Delta_r` is the discrepancy between the
    function and the first order approximation.

    Consider now a `vector field <https://en.wikipedia.org/wiki/Vector_field>`__
    :math:`\mathbf{f}: \mathbb{R}^M \rightarrow \mathbb{R}^N`.
    Analogously, we can approximate
    :math:`\mathbf{f}(\mathbf{m} + \alpha\Delta\mathbf{m})`
    using a first-order Taylor series expansion:

    .. math::

        \mathbf{f}(\mathbf{m} + \alpha\Delta\mathbf{m}) =
        \mathbf{f}(\mathbf{m})
        + \mathbf{J}_\mathbf{f}(\mathbf{m}) \cdot \alpha\Delta\mathbf{m} + \Delta_r,

    where :math:`\mathbf{J}_\mathbf{f}(\mathbf{m})` is the Jacobian matrix of the
    vector field :math:`\mathbf{f}` evaluated in the model :math:`\mathbf{m}`,
    and :math:`\Delta_r` is also the discrepancy between the function and its
    approximation.

    The discrepances :math:`\Delta_r` should get smaller as the scaling factor
    :math:`\alpha` decreases.
    This test will check that the discrepances increases with the ``factors``, and error
    out if such condition doesn't hold.
    """
    # Define factors if None or sort them if a list is passed
    if factors is None:
        factors = np.logspace(-6, 0, 7)
    else:
        factors = factors.copy()
        factors.sort()

    # Cache the value of function and derivative on the model to avoid recomputing them
    function_m = function(model)
    derivative_m = derivative(model)

    # Compute the absolute value of the discrepances for each factor
    discrepances = []
    for factor in factors:
        approximation = function_m + derivative_m @ (factor * delta_m)
        abs_diff = np.abs(function(model + (factor * delta_m)) - approximation)
        discrepances.append(abs_diff)
    discrepances = np.array(discrepances)

    # Check if the discrepances decrease with the factor
    try:
        assert np.all(discrepances[:-1] <= discrepances[1:])
    except AssertionError as e:
        msg = (
            f"Failed convergence test on derivatives for function '{function}' and "
            f"derivative '{derivative}' with:"
            f"\nmodel:    {model}"
            f"\ndelta_m:  {delta_m}"
            "\n"
        )
        raise AssertionError(msg + str(e)) from None
