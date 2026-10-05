"""
Utility functions.
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags_array
from scipy.sparse.linalg import LinearOperator

from .typing import SparseArray

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
    jacobian : (n_data, n_params) array or sparse array
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
