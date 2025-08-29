"""
Code utilities.
"""

import functools
import hashlib
import logging

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags_array

from .base import Objective

__all__ = [
    "cache_on_model",
    "get_logger",
]

LOGGER = logging.Logger("inversions")
LOGGER.addHandler(logging.StreamHandler())


def _create_logger():
    """
    Create custom logger.
    """
    logger = logging.getLogger("inversions")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("{levelname}: {message}", style="{")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
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
    """
    return LOGGER


def cache_on_model(func):
    """
    Cache the last result of a method within the instance using the model hash.

    .. important::

        Use this decorator only for methods that take the ``model`` as the first
        argument.

    .. important::

        The instance needs to have a ``cache`` bool attribute. If True, the result
        of the decorated method will be cached. If False, no caching will be performed.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> class MyClass:
    >>>
    >>>     def __init__(self):
    >>>         self.cache = True
    >>>
    >>>     @cache_on_model
    >>>     def squared(self, model) -> float:
    >>>         return (self.model ** 2).sum()
    >>>
    >>>
    >>> sq = MyClass()
    >>> model = np.array([1.0, 2.0, 3.0])
    >>> print(squared(model))  # perform the computation
    np.float64(14.0)
    >>> print(squared(model))  # access the cached result
    np.float64(14.0)

    >>> model_new = np.array([4.0, 5.0, 6.0])
    >>> squared(model_new)  # perform a new computation
    """
    # Define attribute name for the model hash
    model_hash_attr = "_model_hash"

    # Define attribute name for the cached result using the hash of the function
    cache_attr = f"_cache_{hash(func)}"

    @functools.wraps(func)
    def wrapper(self, model, *args, **kwargs):
        if not hasattr(self, "cache"):
            msg = f"Missing 'cache' attribute in {self}"
            raise AttributeError(msg)

        if self.cache:
            model_hash = hashlib.sha256(model)
            if (
                hasattr(self, model_hash_attr)
                and getattr(self, model_hash_attr).digest() == model_hash.digest()
            ):
                return getattr(self, cache_attr)

            result = func(self, model, *args, **kwargs)
            setattr(self, cache_attr, result)
            setattr(self, model_hash_attr, model_hash)
        else:
            result = func(self, model, *args, **kwargs)
        return result

    return wrapper


def get_jacobi_preconditioner(
    objective_function: Objective, model: npt.NDArray[np.float64]
):
    r"""
    Obtain a Jacobi preconditioner from an objective function.

    Parameters
    ----------
    objective_function : Objective
        Objective function from which the preconditioner will be built.
    model : (n_params) array
        Model used to build the preconditioner.

    Returns
    -------
    diag_array
        Preconditioner as a sparse diagonal array.

    Notes
    -----
    Given an objective function :math:`\phi(\mathbf{m})`, this function builds the
    Jacobi preconditioner :math:`\mathbf{P}(\mathbf{m})` as the inverse of the diagonal
    of the Hessian of :math:`\phi(\mathbf{m})`:

    .. math::

        \mathbf{P}(\mathbf{m}) = \text{diag}[ \bar{\bar{\nabla}} \phi(\mathbf{m}) ]^{-1}

    where :math:`\bar{\bar{\nabla}} \phi(\mathbf{m})` is the Hessian of
    :math:`\phi(\mathbf{m})`.
    """
    hessian_diag = objective_function.hessian_diagonal(model)

    # Compute inverse only for non-zero elements
    zeros = hessian_diag == 0.0
    hessian_diag[~zeros] **= -1

    return diags_array(hessian_diag)
