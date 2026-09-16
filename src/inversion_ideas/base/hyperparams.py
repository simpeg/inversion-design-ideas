"""
Base classes for custom hyperparameter objects.
"""

from math import ceil, floor, trunc
from numbers import Number, Real


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
        return str(self.value)

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
        # Allow multiplication by objects that are not Number (e.g. objective function)
        # In such cases, make the other object to handle the multiplication.
        if not isinstance(other, Number):
            return other.__mul__(self.value)
        return self.value * other

    def __rmul__(self, other):
        # Allow multiplication by objects that are not Number (e.g. objective function)
        # In such cases, make the other object to handle the multiplication.
        if not isinstance(other, Number):
            return other.__rmul__(self.value)
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
        return self.value / other

    def __rtruediv__(self, other):
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
