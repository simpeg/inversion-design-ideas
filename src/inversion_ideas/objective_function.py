"""
Classes to represent objective functions.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


class Objective(ABC):
    """
    Abstract representation of an objective function.
    """

    @abstractmethod
    def __init__(self):
        self.name = "φ"
        self.name_latex = r"\phi"

    @property
    @abstractmethod
    def n_params(self) -> int:
        """
        Number of model parameters.
        """

    @abstractmethod
    def __call__(self, model: NDArray) -> float:
        """
        Evaluate the objective function for a given model.
        """

    @abstractmethod
    def gradient(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """

    @abstractmethod
    def hessian(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the hessian of the objective function for a given model.
        """

    def __repr__(self):  # noqa: D105
        return f"{self.name}(m)"

    def _latex_repr_(self):
        return f"${self.name_latex}(m)$"

    def __add__(self, other) -> "Combo":  # noqa: D105
        return Combo([self, other])

    def __radd__(self, other) -> "Combo":  # noqa: D105
        return Combo([other, self])

    def __mul__(self, value) -> "Scaled":  # noqa: D105
        return Scaled(value, self)

    def __rmul__(self, value):  # noqa: D105
        return self.__mul__(value)

    def __div__(self, denominator):  # noqa: D105
        return self * (1.0 / denominator)

    def __truediv__(self, denominator):  # noqa: D105
        return self * (1.0 / denominator)


class Scaled(Objective):
    """
    Scaled objective function.
    """

    def __init__(self, multiplier, function):
        self.multiplier = multiplier
        self.function = function

    @property
    def n_params(self) -> int:
        """
        Number of model parameters.
        """
        return self.function.n_params

    def __call__(self, model):
        """
        Evaluate the objective function.
        """
        return self.multiplier * self.function(model)

    def gradient(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """
        return self.multiplier * self.function.gradient(model)

    def hessian(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the hessian of the objective function for a given model.
        """
        return self.multiplier * self.function.hessian(model)

    def __repr__(self):  # noqa: D105
        fmt = ".2e" if np.abs(self.multiplier) > 1e3 else ".2f"
        phi_repr = f"{self.function}"
        if isinstance(self.function, Iterable):
            phi_repr = f"[{phi_repr}]"
        return f"{self.multiplier:{fmt}} {phi_repr}"

    def _latex_repr_(self):
        fmt = ".2e" if np.abs(self.multiplier) > 1e3 else ".2f"
        multiplier_str = f"{self.multiplier:{fmt}}"
        if "e" in multiplier_str:
            base, exp = multiplier_str.split("e")
            exp = str(int(exp))
            multiplier_str = rf"{base} \cdot 10^{{exp}}"
        return f"${self.multiplier:{fmt}} {self.name_latex}(m)$"


class Combo(Objective):
    """
    Sum of objective functions.
    """

    def __init__(self, functions):
        # Check if functions have the same n_params
        n_params_list = [f.n_params for f in functions]
        if not all(p == n_params_list[0] for p in n_params_list):
            msg = "Invalid objective functions with different n_params."
            raise ValueError(msg)

        self.functions = functions

    def __iter__(self):  # noqa: D105
        return (f for f in self.functions)

    def __getitem__(self, index):  # noqa: D105
        return self.functions[index]

    @property
    def n_params(self) -> int:
        """
        Number of model parameters.
        """
        n_params_list = [f.n_params for f in self.functions]
        n_params = n_params_list[0]
        if not all(p == n_params for p in n_params_list):
            msg = "Invalid objective functions with different n_params."
            raise ValueError(msg)
        return n_params

    def __call__(self, model):
        """
        Evaluate the objective function.
        """
        return sum(f(model) for f in self.functions)

    def gradient(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the gradient of the objective function for a given model.
        """
        return sum(f.gradient(model) for f in self.functions)

    def hessian(self, model: NDArray) -> NDArray[np.float64]:
        """
        Evaluate the hessian of the objective function for a given model.
        """
        return sum(f.hessian(model) for f in self.functions)

    def __repr__(self):  # noqa: D105
        return " + ".join(repr(f) for f in self.functions)

    def _latex_repr_(self):
        return " + ".join(repr(f) for f in self.functions)


def _unpack_combo(functions: list) -> list:
    """
    Unpack combo objective functions.
    """
    unpacked = []
    for f in functions:
        if isinstance(f, Iterable):
            unpacked.extend(_unpack_combo(f))
        else:
            unpacked.append(f)
    return unpacked
