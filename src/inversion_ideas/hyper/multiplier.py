"""
Multiplier class.
"""
from numbers import Number, Real


class CooledMultiplier(Real):
    """
    Multiplier hyperparameter that can be cooled.
    """
    # TODO: Write a base class for all multiplier classes.
    # TODO: Add a __repr__ and __str__ methods

    def __init__(self, initial_value: float, *, cooling_factor: float):
        self._initial_value = initial_value
        self.cooling_factor = cooling_factor
        self._value = initial_value

    def update(self, *args):  # ruff: ignore[ARG002]
        """Cool down the multiplier."""
        self._value /= self.cooling_factor

    @property
    def initial_value(self):
        return self._initial_value

    @property
    def value(self):
        return self._value

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return self.value + other

    def __mul__(self, other):
        # Allow multiplication by objects that are not Number (e.g. objective function)
        # In such cases, make the other object to handle the multiplication.
        if not isinstance(other, Number):
            return other.__mul__(self.value)
        return self.value * other

    def __imul__(self, other):
        self._value *= other

    def __abs__(self):
        raise NotImplementedError

    def __ceil__(self):
        raise NotImplementedError

    def __eq__(self, other):
        return self.value == other

    def __float__(self):
        raise NotImplementedError

    def __floor__(self):
        raise NotImplementedError

    def __floordiv__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __pos__(self):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError

    def __rfloordiv__(self, other):
        raise NotImplementedError

    def __rmod__(self, v):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __round__(self, ndigits=None):
        raise NotImplementedError

    def __rpow__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __trunc__(self):
        raise NotImplementedError
