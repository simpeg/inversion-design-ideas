"""
Utility functions.
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags_array
from scipy.sparse.linalg import LinearOperator
from simpeg.typing import RandomSeed

from .base import Objective
from .errors import ConvergenceError
from .typing import Model, SparseArray

__all__ = [
    "Counter",
    "get_logger",
    "get_sensitivity_weights",
]


def _create_logger():
    """
    Create custom logger.
    """
    logger = logging.getLogger("inversions")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    formatter = logging.Formatter("[{levelname}] {asctime} | {message}", style="{")
    handler.setFormatter(formatter)
    return logger


LOGGER = _create_logger()


def get_logger():
    r"""
    Get the default event logger.

    The logger records events and relevant information while setting up simulations and
    inversions. By default the logger will stream to stderr and using the INFO level.

    Returns
    -------
    logger : :class:`logging.Logger`
        The logger object for SimPEG.

    Examples
    --------
    Send an info message to the logger:

    >>> get_logger().info("Testing!")

    Change logging level:

    >>> import logging
    >>> logger = get_logger()
    >>> logger.setLevel("DEBUG")
    """
    return LOGGER


def get_sensitivity_weights(
    jacobian: npt.NDArray[np.float64] | SparseArray,
    *,
    data_weights: npt.NDArray[np.float64] | None = None,
    volumes: npt.NDArray[np.float64] | None = None,
    vmin: float | None = 1e-12,
):
    r"""
    Compute sensitivity weights.

    Parameters
    ----------
    jacobian : (n_data, n_params) array or sparray
        Jacobian matrix used to compute sensitivity weights. It must be a dense or
        sparse array.
    data_weights : (n_data,) array or None, optional
        Array with data weights used to compute the sensitivty weights.
        Can use the :attr:`inversion_ideas.DataMisfit.weights` property.
    volumes : (n_params) array or None, optional
        Array with the volumes of the active cells. Sensitivity weights are
        divided by the volumes to account for sensitivity changes due to cell sizes.
    vmin : float or None, optional
        Minimum value used for clipping.

    Notes
    -----
    Given a Jacobian matrix :math:`\mathbf{J}`:

    .. math::

        \mathbf{J} = \begin{bmatrix}
        J_{11} & \cdots & J_{1M} \\
        \vdots & \ddots & \vdots \\
        J_{N1} & \cdots & J_{NM}
        \end{bmatrix}


    a data weights vector :math:`\mathbf{w}`:

    .. math::

        \mathbf{w} = \left[ w_1, \dots, w_N \right],

    and a cell volumes vector :math:`\mathbf{V}`:

    .. math::

        \mathbf{V} = \left[ V_1, \dots, V_M \right],

    the :math:`j` -th component of the initial sensitivity weights vector
    :math:`\hat{\mathbf{s}}` is defined as:

    .. math::

        \hat{s}_j = \frac{1}{V_j} \sqrt{ \sum\limits_{i=0}^N w_i J_{ij}^2 }.


    .. note::

        If ``data_weights`` is ``None``, each one of the :math:`w_i` elements will be
        equal to one.

    .. important::

        If ``volumes`` is ``None``, then each one of the :math:`V_j` elements will be
        equal to one.

    Each one of the components of the sensitivity weights vector :math:`\mathbf{s}` is
    obtained by normalizing the components of :math:`\hat{\mathbf{s}}` by its maximum
    value:

    .. math::

        s_j = \frac{\hat{s}_j}{\max({\hat{\mathbf{s}}})}

    .. important::

        If ``vmin`` is passed, then all sensitivity weights below or equal to that value
        will be assigned the same ``vmin`` value.

    """
    if isinstance(jacobian, LinearOperator):
        msg = (
            f"Invalid jacobian '{jacobian}' of type {type(jacobian).__name__}. "
            "It must be a dense or sparse array."
        )
        raise TypeError(msg)
    if data_weights is not None and data_weights.ndim != 1:
        msg = (
            f"Invalid data_weights array with '{data_weights.ndim}' dimensions. "
            "It must be a 1D array with data weights."
        )
        raise ValueError(msg)

    if isinstance(jacobian, np.ndarray):
        # Compute sensitivity weights using np.einsum for dense arrays.
        # This way we avoid allocating any other large matrix.
        sensitivty_weights = np.sqrt(
            np.einsum("ij,ij->j", jacobian, jacobian)
            if data_weights is None
            else np.einsum("i,ij,ij->j", data_weights, jacobian, jacobian)
        )
    else:
        # Compute the square matrix for sparse arrays (and any other type)
        matrix = (
            diags_array(np.sqrt(data_weights)) @ jacobian
            if data_weights is not None
            else jacobian
        )
        sensitivty_weights = np.sqrt(np.sum(matrix**2, axis=0))

    if volumes is not None:
        sensitivty_weights /= volumes

    # Normalize it by maximum value
    sensitivty_weights /= sensitivty_weights.max()

    # Clip to vmin
    if vmin is not None:
        sensitivty_weights[sensitivty_weights < vmin] = vmin

    return sensitivty_weights


class Counter:
    """
    Simple counter callable class.

    Count how many times the object gets called.

    Parameters
    ----------
    initial_value : int, optional
        Initial value used in the counts.
    """

    def __init__(self, initial_value=0):
        self._initial_value = initial_value
        self._counts = initial_value

    @property
    def initial_value(self):
        """
        Initial value for the counter.
        """
        return self._initial_value

    @property
    def counts(self) -> int:
        """
        Return current amount of counts.
        """
        return self._counts

    def reset(self):
        """
        Reset counter to the initial value.
        """
        self._counts = self.initial_value

    def __call__(self, *args, **kwargs):  # noqa: ARG002
        """
        Increase ``counts`` by one.

        Parameters
        ----------
        *args :
            Position-based arguments that will be ignored.
        **kwargs :
            Keyword arguments that will be ignored.
        """
        self._counts += 1


def get_largest_eigenvalue(
    matrix: npt.NDArray[np.float64] | SparseArray | LinearOperator,
    *,
    n_iters: int = 4,
    seed: RandomSeed | None = None,
    atol: float = 1e-16,
) -> float:
    r"""
    Estimate largest absolute eigenvalue for a diagonadiagonalizable matrix.

    Use the power iteration method to estimate the eigenvector associated with the
    largest absolute eigenvalue for the given matrix.

    Parameters
    ----------
    matrix : (n,n) array or (n,n) sparray or (n,n) LinearOperator
        Square matrix or :class:`~scipy.sparse.linalg.LinearOperator` which eigenvalue
        will be estimated.
    n_iters : int, optional
        Number of iterations used in the power iteration method.
    seed : int or numpy.random.Generator or None, optional
        Random seed used to define a random initial guess for the eigenvector.
    atol : float, optional
        Absolute tolerance that will be passed to :func:`numpy.isclose` when checking if
        the norm of the eigenvector is small enought that it can be considered equal
        to zero. If a null eigenvector is found during the iterations, a
        :exc:`~inversion_ideas.errors.ConvergenceError` will be raised.

    Returns
    -------
    float
        Estimated largest absolute eigenvalue of the ``matrix`` argument.

    Raises
    ------
    inversion_ideas.errors.ConvergenceError :
        If the norm of the eigenvector falls below the ``atol`` argument, and therefore
        is considered to be the null vector.

    References
    ----------
    https://en.wikipedia.org/wiki/Power_iteration

    Notes
    -----
    Given a diagonalizable matrix :math:`\mathbf{A}`, we can estimate the eigenvector
    related to its largest absolute eigenvalue :math:`|\lambda|` through the power
    iteration method.
    In a nutshell, we start with a random unit vector :math:`\mathbf{b}_0`, and
    estimate that eigenvector through a recursive relation:

    .. math::

        \mathbf{b}_{k+1} = \frac{
            \mathbf{A}\mathbf{b}_k
        }{
            \lVert \mathbf{A}\mathbf{b}_k \rVert
        }

    Once we obtained the :math:`\mathbf{b}_k` eigenvector, we can get the absolute value
    of its corresponding eigenvalue through:

    .. math::

        |\lambda| = \frac{
            \lVert \mathbf{A} \mathbf{b}_k \rVert
        }{
            \lVert \mathbf{b}_k \rVert
        }

    If the eigenvector :math:`\mathbf{b}_k` is a unit vector (normalized by its
    magnitude), then we can simplify the last expression to:

    .. math::

        |\lambda| = \lVert \mathbf{A} \mathbf{b}_k \rVert

    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.array(
    ...     [
    ...         [1., 0., 0.],
    ...         [0., 2., 0.],
    ...         [0., 0., 3.],
    ...     ]
    ... )
    >>> get_largest_eigenvalue(matrix, seed=42) # doctest: +NUMBER
    3.0

    """
    # Check if matrix is a 2D array or operator
    if matrix.ndim != 2:
        msg = (
            f"Invalid matrix '{matrix}' with '{matrix.ndim}' dimensions. "
            "It must be a 2D dense or sparse array, or a LinearOperator."
        )
        raise ValueError(msg)

    # Check if matrix is square
    shape = matrix.shape
    if shape[0] != shape[1]:
        msg = (
            f"Invalid non-square matrix with shape {matrix.shape}. "
            "It must be a diagonalizable square matrix."
        )
        raise ValueError(msg)

    # Estimate eigenvector related to largest eigenvalue
    rng = np.random.default_rng(seed=seed)
    eigenvector = rng.random(size=shape[0])
    eigenvector /= np.linalg.norm(eigenvector)

    for _ in range(n_iters):
        next_vector = matrix @ eigenvector
        norm = np.linalg.norm(next_vector)
        if np.isclose(norm, 0.0, atol=atol):
            msg = (
                "Power method to estimate the largest eigenvalue "
                " was interrupted after producing the null vector."
            )
            raise ConvergenceError(msg)
        eigenvector = next_vector / norm

    # Estimate eigenvalue from eigenvector
    dot_product = matrix @ eigenvector
    eigenvalue_abs = np.linalg.norm(dot_product)

    return float(eigenvalue_abs)


def estimate_tradeoff_parameter(
    objective_one: Objective,
    objective_two: Objective,
    model: Model,
    **kwargs,
):
    r"""
    Estimate balanced value for the trade-off parameter between two objective functions.

    Make use of the power iteration method implemented in
    :func:`~inversion_ideas.utils.get_largest_eigenvalue` to estimate the largeset
    absolute eigenvalue of the Hessian matrices of the two objective functions for a
    given model.
    And then return the ratio between the eigenvalue of the first and second Hessian
    matrices.

    Parameters
    ----------
    objective_one, objective_two : base.Objective
        First and second objective functions.
        The Hessian matrices of the two of them will be computed and then used to
        calculate their largest absolute eigenvalues.
    model : (n_params,) array
        Model array used to compute the Hessian matrices.
    **kwargs :
        Keyword arguments that will be passed to
        :func:`~inversion_ideas.utils.get_largest_eigenvalue`.

    Returns
    -------
    float :
        The ratio between the largest eigenvalues of the Hessian of ``objective_one``
        and ``objective_two``.

    Notes
    -----
    Consider two objective functions :math:`\phi_1(\mathbf{m})` and
    :math:`\phi_2(\mathbf{m})`, and we desire to estimate a *balanced* value for a
    trade-off parameter :math:`\beta` that acts as a multiplier of
    :math:`\phi_2(\mathbf{m})`:

    .. math::

        \phi_1(\mathbf{m}) + \beta \, \phi_2(\mathbf{m})

    This function will estimate the largest absolute eigenvalues
    :math:`|\lambda_1|` and
    :math:`|\lambda_2|` for the Hessian matrices
    :math:`\nabla^2 \phi_1(\mathbf{m})` and
    :math:`\nabla^2 \phi_2(\mathbf{m})`, respectively, for a given model
    :math:`\mathbf{m}`.

    And finally estimate a balanced value for trade-off parameter :math:`\beta` as:

    .. math::

        \beta = \frac{|\lambda_1|}{|\lambda_2|}
    """
    # Check if both objective have the same n_params
    if objective_one.n_params != objective_two.n_params:
        msg = (
            f"Invalid objective functions '{objective_one}' and '{objective_two}' with "
            f"different n_params '{objective_one.n_params}' and "
            f"'{objective_two.n_params}', respectively."
        )
        raise ValueError(msg)

    # Estimate the largest eigenvalue for the hessian of each objective function
    hessian_one = objective_one.hessian(model)
    eigenvalue_one = get_largest_eigenvalue(hessian_one, **kwargs)
    hessian_two = objective_two.hessian(model)
    eigenvalue_two = get_largest_eigenvalue(hessian_two, **kwargs)

    # Compute the ratio between the eigenvalues
    return eigenvalue_one / eigenvalue_two
