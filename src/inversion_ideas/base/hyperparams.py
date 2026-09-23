"""
Base classes for custom hyperparameter objects.
"""

from copy import deepcopy
from math import ceil, floor, trunc
from numbers import Real
from typing import Self

import numpy as np
import numpy.typing as npt

from ._utils import float_to_latex
from .objective_function import Objective


class Multiplier(Real):  # ruff: ignore[PLW1641] (ignore undefined __hash__ method)
    """
    Multiplier hyperparameter.

    Wraps a single float that can be used as a multiplier for objective functions.

    .. note:

        Inherit this class to create custom multiplier hyperparameters, for example to
        add them an ``update`` method that updates the value of the multiplier.

    Parameters
    ----------
    value : float
        Value of the wrapped multiplier.
    mutable : bool, optional
        If False, the wrapped value is immutable, i.e. we cannot change it through
        public properties and methods.
        If True, the wrapped value is mutable and can be modified.

    Examples
    --------
    A wrapped multiplier behaves like any regular float:

    >>> beta = Multiplier(5.0)
    >>> print(beta)
    Multiplier(5.0)

    >>> 3 * beta
    15.0

    By default the multiplier is immutable, i.e. we cannot change the value thorugh
    public methods.

    >>> beta *= 5  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: ...

    >>> beta.value = 10.0  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: ...

    We can make it mutable when defining the multiplier:

    >>> beta = Multiplier(5.0, mutable=True)
    >>> beta *= 5
    >>> print(beta)
    Multiplier(25.0)

    """

    def __init__(self, value: float, *, mutable=False):
        if not isinstance(value, Real):
            msg = (
                f"Invalid 'value' argument of type '{type(value)}'. "
                "It must be a float or similar."
            )
            raise TypeError(msg)
        if not isinstance(mutable, bool):
            msg = (
                f"Invalid 'mutable' argument of type '{type(mutable)}'."
                "It must be a bool."
            )
            raise TypeError(msg)
        self._value = value
        self._mutable = mutable

    def __str__(self):
        return f"{type(self).__name__}({self.value})"

    def _repr_latex_(self):
        value = float_to_latex(self.value)
        return r"$\text{" + f"{type(self).__name__}" + f"}}({value})$"

    def __format__(self, fmt):
        return format(self.value, fmt)

    @property
    def mutable(self):
        """Mutable flag."""
        return self._mutable

    @property
    def value(self):
        """Wrapped multiplier."""
        return self._value

    @value.setter
    def value(self, other):
        if not self.mutable:
            raise TypeError(self._mutability_error_msg())
        self._value = other

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return self.value + other

    def __iadd__(self, other):
        if not self.mutable:
            raise TypeError(self._mutability_error_msg())
        self._value += other
        return self

    def __mul__(self, other):
        # Allow multiplication by objective function.
        # In such cases, make the other object to handle the multiplication.
        if isinstance(other, Objective):
            return NotImplemented
        return self.value * other

    def __rmul__(self, other):
        # Allow multiplication by objective function.
        # In such cases, make the other object to handle the multiplication.
        if isinstance(other, Objective):
            return NotImplemented
        return other * self.value

    def __imul__(self, other):
        if not self.mutable:
            raise TypeError(self._mutability_error_msg())
        self._value *= other
        return self

    def __abs__(self):
        return abs(self.value)

    def __ceil__(self):
        return ceil(self.value)

    def __eq__(self, other):
        return self.value == other

    def __float__(self):
        return float(self.value)

    def __floor__(self):
        return floor(self.value)

    def __floordiv__(self, other):
        return self.value // other

    def __rfloordiv__(self, other):
        return other // self.value

    def __ifloordiv__(self, other):
        if not self.mutable:
            raise TypeError(self._mutability_error_msg())
        self._value //= other
        return self

    def __le__(self, other):
        return self.value <= other

    def __lt__(self, other):
        return self.value < other

    def __ge__(self, other):
        return other <= self.value

    def __gt__(self, other):
        return other < self.value

    def __mod__(self, other):
        return self.value % other

    def __rmod__(self, other):
        return other % self.value

    def __neg__(self):
        return -self.value

    def __pos__(self):
        raise NotImplementedError

    def __pow__(self, exponent):
        return self.value**exponent

    def __rpow__(self, base):
        return base**self.value

    def __round__(self, ndigits=None):
        return round(self.value, ndigits)

    def __truediv__(self, other):
        if isinstance(other, Objective):
            msg = (
                f"True division is not supported between '{self}' of type "
                f"'{type(self)}' and {other} of type '{type(other)}'."
            )
            raise TypeError(msg)
        return self.value / other

    def __rtruediv__(self, other):
        if isinstance(other, Objective):
            msg = (
                f"True division is not supported between '{self}' of type "
                f"'{type(self)}' and {other} of type '{type(other)}'."
            )
            raise TypeError(msg)
        return other / self.value

    def __itruediv__(self, other):
        if not self.mutable:
            raise TypeError(self._mutability_error_msg())
        self._value /= other
        return self

    def __trunc__(self):
        return trunc(self.value)

    def _mutability_error_msg(self) -> str:
        msg = (
            f"Cannot modify the value of '{self!r}' "
            "because it was defined as immutable."
        )
        return msg


class WrappedArray:  # ruff: ignore[PLW1641] (ignore undefined __hash__ method)
    """
    Wraps a Numpy array into a class.

    Wraps a single Numpy array into a class to create custom hyperparameter classes.

    .. note:

        Inherit this class to create custom hyperparameters, for example to create
        regularization weights that have an ``update`` method that can update its values
        based on a given model.

    Parameters
    ----------
    array : array
        Array to be wrapped.

    Examples
    --------
    Wrap an array using this class:

    >>> import numpy as np
    >>> a = np.array([1., 2., 3., 4., 5.])
    >>> array_wrapped = WrappedArray(a)
    >>> array_wrapped
    WrappedArray([1., 2., 3., 4., 5.])

    We can operate with this ``array_wrapped`` as with any other array:
    >>> array_wrapped * 2
    array([ 2.,  4.,  6.,  8., 10.])
    >>> array_wrapped @ array_wrapped
    np.float64(55.0)

    We can also pass it to Numpy functions:

    >>> np.mean(array_wrapped)
    np.float64(3.0)
    >>> np.abs(array_wrapped)
    array([1., 2., 3., 4., 5.])
    """

    def __init__(self, array: npt.NDArray):
        self.array = array

    def __repr__(self):
        array_str = repr(self.array).removeprefix("array")
        return f"{type(self).__name__}{array_str}"

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    def copy(self) -> Self:
        return deepcopy(self)

    @property
    def dtype(self) -> npt.DTypeLike:
        """The data-type for the wrapped array."""
        return self.array.dtype

    def __len__(self) -> int:
        return len(self.array)

    def __add__(self, other):
        return self.array + other

    def __radd__(self, other):
        return other + self.array

    def __mul__(self, other):
        return self.array * other

    def __rmul__(self, other):
        return other * self.array

    def __truediv__(self, other):
        return self.array / other

    def __rtruediv__(self, other):
        return other / self.array

    def __floordiv__(self, other):
        return self.array // other

    def __rfloordiv__(self, other):
        return other // self.array

    def __matmul__(self, other):
        return self.array @ other

    def __rmatmul__(self, other):
        return other @ self.array

    def __lt__(self, other):
        return self.array < other

    def __le__(self, other):
        return self.array <= other

    def __eq__(self, other):
        return self.array == other

    def __ne__(self, other):
        return self.array != other

    def __ge__(self, other):
        return self.array >= other

    def __gt__(self, other):
        return self.array > other

    def __not__(self):
        return not self.array

    def __bool__(self):
        return bool(self.array)

    def __abs__(self):
        return np.abs(self.array)

    def __neg__(self):
        return -self.array

    def __and__(self, other):
        return self.array & other

    def __or__(self, other):
        return self.array | other

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __contains__(self, key):
        return self.array.__contains__(key)

    def transpose(self) -> npt.NDArray:
        """
        Transpose.

        Returns
        -------
        array
        """
        return self.array.T

    @property
    def T(self) -> npt.NDArray:
        """
        Transpose.

        Returns
        -------
        array
        """
        return self.transpose()

    @property
    def size(self) -> int:
        """
        Total number of elements in the wrapped array.

        Returns
        -------
        int
        """
        return self.array.size

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the wrapped array.

        Returns
        -------
        tuple of int
        """
        return self.array.shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in the wrapped array.

        Returns
        -------
        int
        """
        return self.array.ndim

    def min(self, axis=None, out=None, **kwargs):
        """
        Minimum value in the wrapped array along a given axis.

        See Also
        --------
        numpy.min : upstream function
        """
        return self.array.min(axis=axis, out=out, **kwargs)

    def max(self, axis=None, out=None, **kwargs):
        """
        Maximum value in the wrapped array along a given axis.

        See Also
        --------
        numpy.max : upstream function
        """
        return self.array.max(axis=axis, out=out, **kwargs)
