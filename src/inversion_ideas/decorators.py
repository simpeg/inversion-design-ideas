"""
Decorators for functions, methods and classes.
"""

import functools
import hashlib

from ._utils import array_to_str
from .utils import get_logger
from .wires import ModelSlice, MultiSlice


def slice_model(func):
    """
    Slice the input model applying the ``model_slice`` attribute in the instance.

    Use this decorator on methods that take ``model`` as the first parameter. The
    decorator will use the ``model_slice`` attribute of the object to extract the
    relevant slice of the model and pass it to the decorated method.

    .. important::

        Use this decorator only for methods that take the ``model`` as the first
        argument.

    .. important::

        The instance needs to have a ``model_slice`` attribute. If it's None, then the
        method will run without any modification.
    """

    @functools.wraps(func)
    def wrapper(self, model, *args, **kwargs):
        if not hasattr(self, "model_slice"):
            msg = (
                f"Object '{self}' doesn't have a 'model_slice' attribute. "
                "Cannot slice the model without it."
            )
            raise AttributeError(msg)

        # Get model slice
        model_slice: ModelSlice | MultiSlice = self.model_slice

        # Don't slice the model if the object has model_slice as None
        if model_slice is None:
            return func(self, model, *args, **kwargs)

        # Don't slice the model if it's already reduced
        if model.size != model_slice.full_size:
            if model.size != model_slice.size:
                msg = (
                    f"Invalid model of size '{model.size}'. "
                    f"It should be the full model "
                    f"(size of {model_slice.full_size}) "
                    f"or the reduced model (size of {model_slice.size})."
                )
                raise ValueError(msg)
            return func(self, model, *args, **kwargs)

        # Slice the model and call the method with it
        model_reduced = model_slice.extract(model)
        return func(self, model_reduced, *args, **kwargs)

    return wrapper


def expand_output(func):
    """
    Expand output of method using the ``model_slice`` attribute in the instance.

    Use this decorator on any method that needs to expand its output based on the
    ``model_slice`` attribute of the instance.
    If it's a 1D array, it'll expand the array to fill the extra elements with zeros.
    If it's a square matrix or a square linear operator, it'll fill the extra blocks
    with zeros.

    .. important::

        The instance needs to have a ``model_slice`` attribute. If it's None, then the
        method will run without any modification.

    .. important::

        Use this decorator only for methods that return a 1D array, or a 2D array,
        sparse matrix or ``LinearOperator``.

    """

    @functools.wraps(func)
    def wrapper(self, model, *args, **kwargs):
        if not hasattr(self, "model_slice"):
            msg = (
                f"Object '{self}' doesn't have a 'model_slice' attribute. "
                f"Cannot expand the output of '{func}' without it."
            )
            raise AttributeError(msg)

        # Get model slice
        model_slice: ModelSlice | MultiSlice = self.model_slice

        # Don't modify the output if the object has no model_slice
        if model_slice is None:
            return func(self, model, *args, **kwargs)

        result = func(self, model, *args, **kwargs)

        if not hasattr(result, "ndim"):
            msg = (
                f"Invalid object '{result}' of type '{type(result).__name__}' "
                f"returned by '{func}''. "
                "The output of the method should be an array, sparse matrix or "
                "LinearOperator to be able to expand it."
            )
            raise TypeError(msg)

        if result.ndim == 1:
            result = model_slice.expand_array(result)
        elif result.ndim == 2:
            result = model_slice.expand_matrix(result)
        else:
            msg = (
                f"Invalid object with {result.ndim} dimensions. "
                "It must be a 1D or 2D array-like object to be able to expand it."
            )
            raise ValueError(msg)
        return result

    return wrapper


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
    ...
    ...     def __init__(self):
    ...         self.cache = True
    ...
    ...     @cache_on_model
    ...     def squared(self, model) -> float:
    ...         return (model ** 2).sum()
    >>>
    >>> sq = MyClass()
    >>> model = np.array([1.0, 2.0, 3.0])
    >>> print(sq.squared(model))  # perform the computation
    14.0
    >>> print(sq.squared(model))  # access the cached result
    14.0

    >>> model_new = np.array([4.0, 5.0, 6.0])
    >>> print(sq.squared(model_new))  # perform a new computation
    77.0
    """
    # Define attribute name for the cached result using the hash of the function
    cache_attr = f"_cache_{hash(func)}"

    @functools.wraps(func)
    def wrapper(self, model, *args, **kwargs):
        if not hasattr(self, "cache"):
            msg = f"Missing 'cache' attribute in {self}"
            raise AttributeError(msg)

        if self.cache:
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

        # Return result without caching
        return func(self, model, *args, **kwargs)

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
