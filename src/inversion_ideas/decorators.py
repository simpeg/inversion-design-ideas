"""
Decorators for functions, methods and classes.
"""

import functools
import hashlib

from ._utils import array_to_str
from .utils import get_logger


def cache_on_model(func):
    """
    Cache the last result of a method within the instance using the model hash.

    .. important::

        Use this decorator only for methods that take the ``model`` as the first
        argument.

    .. hint::

        If the instance has a ``cache`` bool attribute, useres can enable or disable
        caching within the decorated method. If ``cache=True``, the result
        of the decorated method will be cached. If ``cache=False``, no caching will
        be performed.

    See also
    --------
    Use :func:`functools.cache` for caching multiple results.

    Examples
    --------
    Let's decorate a method that will return the cached result when called again with
    the same ``model`` argument:

    >>> import numpy as np
    >>>
    >>> class MyClass:
    ...
    ...     @cache_on_model
    ...     def squared(self, model) -> float:
    ...         return (model ** 2).sum()
    >>>
    >>> sq = MyClass()

    When calling the method with a given model we'll perform the computation:

    >>> model = np.array([1.0, 2.0, 3.0])
    >>> print(sq.squared(model))  # perform the computation
    14.0

    Next time we call it with the same model, it'll return the cached value:

    >>> print(sq.squared(model))  # access the cached result
    14.0

    When calling the method with a different model, we'll trigger a new computation. And
    the new result will be cached:

    >>> model_new = np.array([4.0, 5.0, 6.0])
    >>> print(sq.squared(model_new))  # perform a new computation
    77.0

    .. important::

        Only the last value gets cached to be cautious about memory usage.

    Users could control wether to cache or not through the ``cache`` attribute:

    >>> class MyClass:
    ...
    ...     def __init__(self, cache):
    ...         self.cache = cache
    ...
    ...     @cache_on_model
    ...     def squared(self, model) -> float:
    ...         return (model ** 2).sum()


    The following instance caches the result:

    >>> sq_cache = MyClass(cache=True)
    >>> model = np.array([1.0, 2.0, 3.0])
    >>> sq_cache.squared(model)  # perform the computation
    >>> sq_cache.squared(model)  # returns cached object

    This one does not:

    >>> sq_no_cache = MyClass(cache=False)
    >>> model = np.array([1.0, 2.0, 3.0])
    >>> sq_no_cache.squared(model)  # perform the computation
    >>> sq_no_cache.squared(model)  # perform the computation

    """
    # Define attribute name for the cached result using the hash of the function
    cache_attr = f"_cache_{hash(func)}"

    @functools.wraps(func)
    def wrapper(self, model, *args, **kwargs):
        if hasattr(self, "cache") and not self.cache:
            # Return result without caching
            return func(self, model, *args, **kwargs)

        model_hash = hashlib.sha256(model)

        # Return cached object if the model hash matches with the cached one
        if hasattr(self, cache_attr):
            model_hash_cached, cached_result = getattr(self, cache_attr)
            if model_hash_cached.digest() == model_hash.digest():
                # -- Debug log --
                msg = (
                    f"Returning cached object '{array_to_str(cached_result)}' "
                    f"after calling '{func}' with model with hash "
                    f"'{model_hash_cached.hexdigest()}'."
                )
                if args:
                    msg += f" With args: '{args}'."
                if kwargs:
                    msg += f" With kwargs: '{kwargs}'."
                get_logger().debug(msg)
                # ---
                return cached_result

        # Compute new result and cache it
        result = func(self, model, *args, **kwargs)
        setattr(self, cache_attr, (model_hash, result))
        # -- Debug log --
        msg = (
            f"Computed new result '{array_to_str(result)}' after "
            f"calling '{func}' with model with hash '{model_hash.hexdigest()}'. "
            "Cached the result into the object."
        )
        if args:
            msg += f" With args: '{args}'."
        if kwargs:
            msg += f" With kwargs: '{kwargs}'."
        get_logger().debug(msg)
        # ---
        return result

    return wrapper


def debug(func):
    """
    Add a debug entry into the logger through a decorator.

    Use this decorator on methods and functions. When such method or function gets
    called, it will add an entry into the logger as a DEBUG level.
    """
    logger = get_logger()

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        msg = f"Called '{func}'"
        if args:
            msg += f" with arguments '{args}'"
        if kwargs:
            msg += f" with keyword arguments '{kwargs}'"
        msg += "."
        logger.debug(msg)
        return func(self, *args, **kwargs)

    return wrapper


class CountCalls:
    """
    Class decorator to count function calls.

    Examples
    --------
    >>> @CountCalls
    ... def my_function(x):
    ...     return x**2
    >>> my_function(1)
    1
    >>> my_function(2)
    4
    >>> my_function.counts
    2
    """

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.counts = 0

    def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        self.counts += 1
        return result
