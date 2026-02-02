"""
Model class.
"""

import numpy as np
import numpy.typing as npt


def _check_key_value(key, value):
    if not isinstance(key, str):
        msg = f"Invalid key '{key}' of type '{type(key)}'. It must be a str."
        raise TypeError(msg)
    if not isinstance(value, np.ndarray):
        msg = (
            f"Invalid value '{value}' of type '{type(value).__name__}'. "
            "It must be an array."
        )
        raise TypeError(msg)


class MultiModel(dict):
    """
    Represent an inversion model with multiple parts.

    Use this class to split the inversion model into multiple parts. For example,
    multiple physical properties, physical properties with multiple components, etc.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            _check_key_value(key, value)
        super().__init__(**kwargs)

    def __setitem__(self, key, value):
        _check_key_value(key, value)
        super().__setitem__(key, value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        msg = f"Cannot delete attributes in {type(self).__name__}"
        raise TypeError(msg)

    def ravel(self) -> npt.NDArray:
        return np.hstack(list(self.values()))

    @property
    def size(self) -> int:
        return sum([v.size for v in self.values()])
