"""
Utility functions.
"""

import logging

import numpy as np
import numpy.typing as npt

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
    jacobian: npt.NDArray[np.float64],
    *,
    data_weights: npt.NDArray[np.float64] | SparseArray | None = None,
    volumes: npt.NDArray[np.float64] | None = None,
    vmin: float | None = 1e-12,
):
    """
    Compute sensitivity weights.

    Parameters
    ----------
    jacobian : (n_data, n_params) array
        Jacobian matrix used to compute sensitivity weights.
    data_weights : (n_data, n_data) array or None, optional
        Data weights matrix used to compute the sensitivity weights.
    volumes : (n_params) array
        Array with the volumes of the active cells. Sensitivity weights are
        divided by the volumes to account for sensitivity changes due to cell sizes.
    vmin : float or None, optional
        Minimum value used for clipping.

    Notes
    -----
    """
    matrix = data_weights @ jacobian if data_weights is not None else jacobian
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
        self._counts = initial_value

    @property
    def counts(self) -> int:
        """
        Return current amount of counts.
        """
        return self._counts

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
