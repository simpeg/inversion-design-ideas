"""
Code utilities.
"""

import functools
import hashlib
import logging


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
